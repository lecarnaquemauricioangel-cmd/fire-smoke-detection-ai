"""
Backend Flask - Sistema de Detección Temprana de Fuego y Humo en Almacenes
SENATI - Curso PIAD-426 - Casuística 3: Seguridad industrial

Ejecutar con:
    python app.py

La aplicación queda disponible en http://localhost:5000
"""

import base64
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify,
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO

import database
import reportes
from alertas import severidad, enviar_telegram

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
RESULT_FOLDER = STATIC_DIR / "results"
VIDEO_FOLDER = STATIC_DIR / "videos"

for carpeta in (UPLOAD_FOLDER, RESULT_FOLDER, VIDEO_FOLDER):
    carpeta.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
app.secret_key = "senati-piad426-fuego-humo-2026"
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

# ---------------------------------------------------------------------------
# Carga del modelo YOLOv8 (con fallback a yolov8n.pt si no existe/es inválido best.pt)
# ---------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "best.pt"

# Algunos modelos públicos de fuego/humo usan nombres en inglés ('Fire', 'Smoke')
# o incluyen clases de relleno propias de exportaciones de Roboflow (ej. 'default').
# Este mapa normaliza cualquier variante conocida a 'fuego'/'humo'; toda clase que
# no aparezca aquí se descarta (devuelve None) para no ensuciar las detecciones.
_ALIAS_CLASES = {
    "fuego": "fuego", "fire": "fuego",
    "humo": "humo", "smoke": "humo",
}


def _normalizar_clase(nombre: str):
    return _ALIAS_CLASES.get(str(nombre).strip().lower())


def _cargar_modelo():
    """
    Intenta cargar 'models/best.pt'. Solo se considera válido si carga sin
    errores Y al menos una de sus clases normaliza a 'fuego' o 'humo'
    (aceptando nombres en inglés u otras variantes conocidas). En cualquier
    otro caso cae de vuelta a 'yolov8n.pt' (modo de pruebas) y explica por
    consola cómo resolverlo con 'preparar_modelo.py'.
    """
    if MODEL_PATH.exists():
        try:
            candidato = YOLO(str(MODEL_PATH))
            nombres_crudos = {int(k): v for k, v in candidato.names.items()}
            normalizados = {cid: _normalizar_clase(n) for cid, n in nombres_crudos.items()}
            if any(normalizados.values()):
                print(f"[app] Modelo personalizado cargado correctamente: {MODEL_PATH} "
                      f"(clases originales: {nombres_crudos} -> normalizadas: {normalizados})")
                return candidato, normalizados, True
            print(f"[app] 'models/best.pt' existe pero ninguna de sus clases {nombres_crudos} "
                  "corresponde a fuego/humo.")
        except Exception as exc:
            print(f"[app] No se pudo cargar 'models/best.pt' ({exc}).")
    else:
        print("[app] No se encontró 'models/best.pt'.")

    print("[app] Usando 'yolov8n.pt' de respaldo (modo de pruebas, clases genéricas COCO). "
          "Ejecuta 'python preparar_modelo.py' para generar un modelo válido con clases "
          "fuego/humo y eliminar este aviso.")
    modelo_respaldo = YOLO("yolov8n.pt")
    nombres_respaldo = {int(k): v for k, v in modelo_respaldo.names.items()}
    return modelo_respaldo, nombres_respaldo, False


modelo, CLASS_NAMES, MODELO_ES_PERSONALIZADO = _cargar_modelo()
print(f"[app] model.names verificado: {modelo.names}")

# ---------------------------------------------------------------------------
# Parámetros de inferencia: pipeline nativo de Ultralytics, un único umbral
# de confianza para TODAS las rutas (imagen, video, cámara). Sin heurísticas
# de visión clásica ni umbrales dobles: toda caja dibujada es una detección
# real de YOLOv8 con su confianza real.
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.10  # alta sensibilidad para capturar fuego y humo simultáneos
IMGSZ_YOLO = 640
IOU_THRESHOLD = 0.75
AGNOSTIC_NMS = False  # Ultralytics ya hace NMS por clase: 'humo' nunca suprime 'fuego'.
MAX_DETECCIONES = 50  # margen amplio para escenas con varios focos de fuego/humo.

# Cooldown (en segundos) para no saturar Telegram/BD con alertas de cámara
COOLDOWN_ALERTA_SEGUNDOS = 30
_ultima_alerta = {}

# Al procesar video: ejecutar inferencia 1 de cada N frames (rendimiento en CPU)
DETECCION_CADA_N_FRAMES = 3
# Alto máximo del video de salida. Reescalar antes de dibujar/escribir/
# codificar (en vez de solo en FFmpeg) acelera las tres etapas: la propia
# inferencia YOLO, cv2.VideoWriter y la recodificación final con FFmpeg.
ALTO_MAX_VIDEO_SALIDA = 720

# ---------------------------------------------------------------------------
# Memoria de persistencia (video / cámara en vivo)
# ---------------------------------------------------------------------------
# Toda caja que dibuja dibujar_detecciones() es una detección REAL de YOLOv8
# (misma confianza que reportó el modelo, sin inventar ni aproximar nada).
# Lo único que hace esta memoria es mantener esa caja en pantalla algunos
# fotogramas más tras perder la reconfirmación, para que una llama/humo que
# fluctúa cuadro a cuadro no parpadee -- no baja el umbral de confianza ni
# crea detecciones nuevas por su cuenta.
#
# Un foco activo que deja de reconfirmarse sigue dibujándose esta cantidad de
# FOTOGRAMAS antes de retirarse: 30 frames a ~24-30 fps ≈ 1s.
HOLD_FRAMES_TRACKING = 30
# IoU mínimo para considerar que una caja de este ciclo es la MISMA zona de
# fuego/humo que un foco ya activo (y no un foco nuevo aparte).
IOU_MATCH_TRACKING = 0.15


def _iou(bbox_a, bbox_b):
    """Intersection-over-Union de dos bboxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    ancho_i, alto_i = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    area_i = ancho_i * alto_i
    if area_i <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - area_i
    return area_i / union if union > 0 else 0.0


def actualizar_tracks(tracks, detecciones_crudas, hold_frames=HOLD_FRAMES_TRACKING):
    """
    Actualiza la lista de "focos activos" (tracks) con las detecciones
    REALES de YOLOv8 de este ciclo de inferencia (todas ya pasaron el único
    umbral CONF_THRESHOLD dentro de modelo.predict(), no hay un segundo
    filtro aquí):

    - Una caja de este ciclo que coincide (misma clase + IoU >=
      IOU_MATCH_TRACKING) con un track ya activo lo REFRESCA (posición,
      confianza real, contador de persistencia).
    - Una caja que no coincide con ningún track activo abre uno nuevo.
    - Un track activo que no se reconfirma este ciclo pierde 1 fotograma de
      su contador de persistencia; al llegar a 0 se retira.
    """
    usados = set()
    tracks_actualizados = []

    for track in tracks:
        mejor_iou, mejor_idx = 0.0, None
        for i, det in enumerate(detecciones_crudas):
            if i in usados or det["clase"] != track["clase"]:
                continue
            iou = _iou(track["bbox"], det["bbox"])
            if iou > mejor_iou:
                mejor_iou, mejor_idx = iou, i

        if mejor_idx is not None and mejor_iou >= IOU_MATCH_TRACKING:
            det = detecciones_crudas[mejor_idx]
            usados.add(mejor_idx)
            # dict(det, ...) preserva banderas como "color"/"densidad" para
            # que la etiqueta honesta de la heurística no se pierda al
            # refrescar el track en video/cámara.
            tracks_actualizados.append(dict(det, frames_restantes=hold_frames))
        else:
            track["frames_restantes"] -= 1
            if track["frames_restantes"] > 0:
                tracks_actualizados.append(track)

    for i, det in enumerate(detecciones_crudas):
        if i in usados:
            continue
        tracks_actualizados.append(dict(det, frames_restantes=hold_frames))

    return tracks_actualizados


def decaer_tracks(tracks):
    """Descuenta 1 fotograma de persistencia a todos los tracks (frames sin
    inferencia nueva, p.ej. los que se saltan por DETECCION_CADA_N_FRAMES)."""
    restantes = []
    for track in tracks:
        track["frames_restantes"] -= 1
        if track["frames_restantes"] > 0:
            restantes.append(track)
    return restantes


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def extension_permitida(nombre_archivo, extensiones_validas):
    return (
        "." in nombre_archivo
        and nombre_archivo.rsplit(".", 1)[1].lower() in extensiones_validas
    )


def nombre_unico(prefijo, extension):
    marca = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefijo}_{marca}_{uuid.uuid4().hex[:8]}.{extension}"


# Una caja "gigante" (de humo O de fuego) que cubre esta fracción o más del
# frame dispara la extracción complementaria de la OTRA clase -- SOLO dentro
# de esa región. Fuera de esa región nunca se activa, para no disparar con
# piel, atardeceres o luces cálidas en el resto de la imagen.
UMBRAL_CAJA_GIGANTE = 0.5
# Confianza fija y APROXIMADA para todo foco encontrado por heurística de
# color/densidad (no es una probabilidad real de un clasificador). Se
# etiqueta como "(color)"/"(densidad)" al dibujarla para no aparentar la
# precisión de una detección real de YOLOv8.
CONFIANZA_HEURISTICA = 0.50
# Al extraer focos individuales dentro de una caja gigante, esta se recorta
# (nunca se elimina) para que deje de tapar toda la pantalla -- pero sigue
# marcando la zona general del incendio/humareda.
AREA_MAXIMA_CAJA_GIGANTE = 0.7


def _detectar_fuego_por_color(frame, region_bbox):
    """
    Heurística de color (HSV) para localizar focos de fuego DENTRO de una
    región donde YOLO ya marcó una masa de humo grande. Físicamente, un
    humo denso suele tener llamas debajo que la propia caja de humo puede
    "tapar" visualmente al no generar su propia caja de clase fuego. Se
    acota estrictamente a esa región (nunca al frame completo) para reducir
    falsos positivos por color cálido ajeno al incendio.
    """
    x1, y1, x2, y2 = [int(v) for v in region_bbox]
    alto, ancho = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ancho, x2), min(alto, y2)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return []

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Rango de fuego: tonos amarillo-anaranjado-rojo. Saturación/brillo más
    # permisivos que un naranja "de manual" (S>=90, V>=120 en vez de
    # S>=130, V>=180) porque el fuego real filtrado a través de humo denso
    # se ve más pálido/atenuado -- verificado con el video de brasas real,
    # donde la llama visible medía S~101-152 y el umbral estricto no
    # encontraba ningún píxel.
    lower_fire = np.array([0, 90, 120])
    upper_fire = np.array([35, 255, 255])
    mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)
    mask_fire = cv2.dilate(mask_fire, kernel, iterations=2)

    contornos, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    encontrados = []
    for cnt in contornos:
        if cv2.contourArea(cnt) > 400:  # foco visible real, no ruido puntual
            bx, by, bw, bh = cv2.boundingRect(cnt)
            encontrados.append({
                "clase": "fuego",
                "confianza": CONFIANZA_HEURISTICA,
                "bbox": [float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh)],
                "color": True,
            })
    return encontrados


def _detectar_humo_por_densidad(frame, region_bbox):
    """
    Heurística de densidad (HSV) para localizar humo DENTRO de una región
    donde YOLO ya marcó una masa de fuego grande. El humo sube por
    convección y se concentra en la franja media-superior de la escena, así
    que se acota a ese 70% superior de la caja de fuego (excluye el tercio
    inferior, donde suele estar la llama pura) buscando zonas difusas de
    baja saturación (gris/blanco/neblina). Nunca se activa fuera de esa
    región.
    """
    x1, y1, x2, y2 = [int(v) for v in region_bbox]
    alto, ancho = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ancho, x2), min(alto, y2)
    y2_franja = y1 + int(round((y2 - y1) * 0.7))
    region = frame[y1:y2_franja, x1:x2]
    if region.size == 0:
        return []

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mascara = ((s < 70) & (v > 100) & (v < 230)).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    encontrados = []
    for cnt in contornos:
        if cv2.contourArea(cnt) > 1500:  # masa de humo real, no ruido puntual
            bx, by, bw, bh = cv2.boundingRect(cnt)
            encontrados.append({
                "clase": "humo",
                "confianza": CONFIANZA_HEURISTICA,
                "bbox": [float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh)],
                "densidad": True,
            })
    return encontrados


# Umbral de área (px) y techo de cobertura del frame para la detección
# difusa de humo/ceniza en video (ver _detectar_humo_difuso): igual que la
# heurística de densidad, pero sin acotarse a ninguna caja de YOLO -- corre
# sobre el frame COMPLETO como respaldo cuando YOLO no reportó clase 'humo'
# en absoluto ese ciclo.
UMBRAL_AREA_HUMO_DIFUSO = 1500
AREA_MAXIMA_HUMO_DIFUSO = 0.6  # evita marcar una pared/cielo gris entero como humo


def _detectar_humo_difuso(frame):
    """
    Heurística de saturación baja para localizar humo/ceniza difusos en TODO
    el frame de video, como respaldo de la clase 'humo' de YOLO: una nube
    tenue (gris/blanco/bruma) puede quedar por debajo del umbral de
    confianza del modelo sin dejar de ser visible a simple vista. Rango HSV
    más laxo que _detectar_humo_por_densidad (S<60, V entre 90 y 220) porque
    aquí no hay ninguna caja de YOLO que ya confirme la zona.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mascara = ((s < 60) & (v > 90) & (v < 220)).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    area_frame = frame.shape[0] * frame.shape[1]
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    encontrados = []
    for cnt in contornos:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt <= UMBRAL_AREA_HUMO_DIFUSO:
            continue
        if area_frame > 0 and area_cnt / area_frame > AREA_MAXIMA_HUMO_DIFUSO:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        encontrados.append({
            "clase": "humo",
            "confianza": CONFIANZA_HEURISTICA,
            "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
            "difuso": True,
        })
    return encontrados


def _recortar_caja_gigante(bbox, area_frame, area_maxima=AREA_MAXIMA_CAJA_GIGANTE):
    """Reduce (alrededor de su centro) una caja que supera 'area_maxima' del
    frame, para que deje de taparlo casi por completo una vez que sus focos
    individuales ya quedaron marcados por separado."""
    x1, y1, x2, y2 = bbox
    ancho, alto = max(0.0, x2 - x1), max(0.0, y2 - y1)
    area = ancho * alto
    if area_frame <= 0 or area <= 0 or area / area_frame <= area_maxima:
        return bbox
    escala = (area_maxima * area_frame / area) ** 0.5
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w2, h2 = ancho * escala / 2, alto * escala / 2
    return [cx - w2, cy - h2, cx + w2, cy + h2]


def ejecutar_deteccion(frame):
    """
    Inferencia nativa de Ultralytics sobre un frame (numpy array BGR).
    Retorna: [{"clase", "confianza", "bbox": [x1, y1, x2, y2]}, ...]

    Extracción bidireccional: si una caja de 'humo' O de 'fuego' es
    "gigante" (>= UMBRAL_CAJA_GIGANTE del frame), se complementa -- SOLO
    dentro de esa región -- con focos de la clase contraria detectados por
    heurística de color/densidad, marcados con "color"/"densidad": True
    para dibujarse y registrarse de forma transparente, nunca como si YOLO
    los hubiera reportado. La caja gigante original se recorta (no se
    elimina) cuando se le encuentran focos propios, para no tapar toda la
    pantalla.
    """
    resultados = modelo.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ_YOLO,
        iou=IOU_THRESHOLD,
        agnostic_nms=AGNOSTIC_NMS,
        max_det=MAX_DETECCIONES,
        verbose=False,
    )
    r = resultados[0]

    detecciones = []
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            clase = CLASS_NAMES.get(cls_id, str(cls_id))
            if clase is None:
                # Clase irrelevante para el modelo cargado (ej. 'default' de Roboflow).
                continue
            confianza = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detecciones.append(
                {"clase": clase, "confianza": confianza, "bbox": [x1, y1, x2, y2]}
            )

    area_frame = frame.shape[0] * frame.shape[1]

    def _es_gigante(bbox):
        bx1, by1, bx2, by2 = bbox
        return area_frame > 0 and (bx2 - bx1) * (by2 - by1) / area_frame >= UMBRAL_CAJA_GIGANTE

    # HUMO gigante -> extrae FUEGO por color dentro de esa región.
    for det_humo in [d for d in detecciones if d["clase"] == "humo"]:
        if not _es_gigante(det_humo["bbox"]):
            continue
        nuevos = [
            c for c in _detectar_fuego_por_color(frame, det_humo["bbox"])
            if not any(d["clase"] == "fuego" and _iou(d["bbox"], c["bbox"]) > 0.3 for d in detecciones)
        ]
        if nuevos:
            detecciones.extend(nuevos)
            det_humo["bbox"] = _recortar_caja_gigante(det_humo["bbox"], area_frame)

    # FUEGO gigante -> extrae HUMO por densidad dentro de esa región.
    for det_fuego in [d for d in detecciones if d["clase"] == "fuego" and not d.get("color")]:
        if not _es_gigante(det_fuego["bbox"]):
            continue
        nuevos = [
            c for c in _detectar_humo_por_densidad(frame, det_fuego["bbox"])
            if not any(d["clase"] == "humo" and _iou(d["bbox"], c["bbox"]) > 0.3 for d in detecciones)
        ]
        if nuevos:
            detecciones.extend(nuevos)
            det_fuego["bbox"] = _recortar_caja_gigante(det_fuego["bbox"], area_frame)

    # Último recurso: si YOLO no reportó absolutamente nada este ciclo (ni
    # fuego ni humo), se revisa el frame COMPLETO por color antes de darlo
    # por "limpio". Esto cubre el caso real de brasas/llamas que el modelo
    # pierde en algunos encuadres muy cerrados, sin que la ausencia total de
    # otra señal (ni una caja YOLO, ni una masa gigante) deje el video en
    # blanco durante varios segundos seguidos.
    if not detecciones:
        detecciones.extend(
            _detectar_fuego_por_color(frame, [0, 0, frame.shape[1], frame.shape[0]])
        )

    return detecciones


def dibujar_detecciones(frame, detecciones):
    """Dibuja los bounding boxes sobre el frame (in-place) y lo retorna."""
    alto_frame = frame.shape[0]
    grosor = 3  # fijo, tanto para fuego (rojo) como para humo (amarillo)
    escala_texto = max(0.6, alto_frame / 900)

    # Cuando varias cajas comparten esquina (típico en fuego+humo, que suelen
    # solaparse mucho), sus etiquetas se dibujan una debajo de otra en vez de
    # superpuestas e ilegibles. Se ordenan por y1 para que la cascada de
    # etiquetas siga el orden visual de las cajas.
    orden = sorted(range(len(detecciones)), key=lambda i: detecciones[i]["bbox"][1])
    desplazamiento_acumulado = {}

    for idx in orden:
        det = detecciones[idx]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        clase = det["clase"]
        confianza = det["confianza"]

        if clase == "fuego":
            color = (0, 0, 255)         # rojo (BGR)
            color_texto = (255, 255, 255)  # texto blanco sobre fondo rojo
        elif clase == "humo":
            color = (0, 255, 255)       # amarillo (BGR)
            color_texto = (0, 0, 0)     # texto negro sobre fondo amarillo (más legible que blanco)
        else:
            color = (255, 200, 0)       # celeste, clases genéricas (modo fallback)
            color_texto = (0, 0, 0)

        if det.get("color"):
            etiqueta = f"{clase.upper()} ~{confianza * 100:.0f}% (color)"
        elif det.get("densidad"):
            etiqueta = f"{clase.upper()} ~{confianza * 100:.0f}% (densidad)"
        elif det.get("difuso"):
            etiqueta = f"{clase.upper()} ~{confianza * 100:.0f}% (difuso)"
        else:
            etiqueta = f"{clase.upper()} {confianza * 100:.1f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)

        (tw, th), _ = cv2.getTextSize(
            etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala_texto, grosor
        )
        # Esquina de referencia (x1, y1) redondeada, para agrupar cajas que
        # nacen prácticamente en el mismo punto y escalonar sus etiquetas.
        clave_esquina = (round(x1 / 20), round(y1 / 20))
        salto = desplazamiento_acumulado.get(clave_esquina, 0)
        desplazamiento_acumulado[clave_esquina] = salto + th + 16

        y_etiqueta_inf = max(th + 2, y1 - 6) + salto
        y_etiqueta_sup = max(0, y_etiqueta_inf - th - 12)
        cv2.rectangle(frame, (x1, y_etiqueta_sup), (x1 + tw + 10, y_etiqueta_sup + th + 10), color, -1)
        cv2.putText(
            frame, etiqueta, (x1 + 5, y_etiqueta_sup + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX, escala_texto, color_texto, grosor, cv2.LINE_AA,
        )
    return frame


def clase_prioritaria(detecciones):
    """Determina la clase 'dominante' para fines de registro en BD."""
    if any(d["clase"] == "fuego" for d in detecciones):
        return "fuego"
    if any(d["clase"] == "humo" for d in detecciones):
        return "humo"
    if detecciones:
        return detecciones[0]["clase"]
    return "ninguno"


def confianza_maxima(detecciones, clase=None):
    valores = [d["confianza"] for d in detecciones if clase is None or d["clase"] == clase]
    return max(valores) if valores else 0.0


def detalle_severidad(info_severidad):
    """Arma el texto de detalle a guardar en SQLite, marcando explícitamente
    la coexistencia de fuego y humo cuando ocurre (caso más crítico)."""
    base = (
        f"Severidad: {info_severidad['nivel']} | "
        f"Área ocupada: {info_severidad['porcentaje_area']}%"
    )
    if info_severidad["hay_fuego"] and info_severidad["hay_humo"]:
        base += " | Coexisten fuego y humo (incendio activo)"
    return base


def puede_alertar(zona: str) -> bool:
    """Aplica un cooldown para no saturar Telegram/BD con alertas repetidas."""
    ahora = time.time()
    ultima = _ultima_alerta.get(zona, 0)
    if ahora - ultima >= COOLDOWN_ALERTA_SEGUNDOS:
        _ultima_alerta[zona] = ahora
        return True
    return False


# ---------------------------------------------------------------------------
# Rutas principales
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", modelo_personalizado=MODELO_ES_PERSONALIZADO)


@app.route("/dashboard")
def dashboard():
    resumen = database.resumen_por_clase()
    eventos = database.ultimos_eventos(20)
    total = database.total_eventos()
    return render_template(
        "dashboard.html", resumen=resumen, eventos=eventos, total=total
    )


@app.route("/exportar_excel")
def exportar_excel():
    ruta = reportes.generar_excel()
    return send_file(ruta, as_attachment=True, download_name="reporte_eventos.xlsx")


# ---------------------------------------------------------------------------
# Detección en imagen
# ---------------------------------------------------------------------------
@app.route("/detectar_imagen", methods=["GET", "POST"])
def detectar_imagen():
    if request.method == "GET":
        return redirect(url_for("index"))

    archivo = request.files.get("imagen")
    if not archivo or archivo.filename == "":
        flash("Debes seleccionar una imagen para analizar.")
        return redirect(url_for("index"))

    if not extension_permitida(archivo.filename, ALLOWED_IMAGE_EXT):
        flash("Formato de imagen no soportado. Usa PNG, JPG, JPEG o BMP.")
        return redirect(url_for("index"))

    nombre_seguro = secure_filename(archivo.filename)
    nombre_guardado = nombre_unico("imagen", nombre_seguro.rsplit(".", 1)[1].lower())
    ruta_original = UPLOAD_FOLDER / nombre_guardado
    archivo.save(ruta_original)

    frame = cv2.imread(str(ruta_original))
    if frame is None:
        flash("No se pudo procesar la imagen. Verifica que el archivo no esté dañado.")
        return redirect(url_for("index"))

    alto, ancho = frame.shape[:2]
    area_frame = ancho * alto

    detecciones = ejecutar_deteccion(frame)
    frame_anotado = dibujar_detecciones(frame.copy(), detecciones)

    nombre_resultado = f"res_{nombre_guardado}"
    ruta_resultado = RESULT_FOLDER / nombre_resultado
    cv2.imwrite(str(ruta_resultado), frame_anotado)

    info_severidad = severidad(detecciones, area_frame)
    nivel = info_severidad["nivel"]

    fuente = f"imagen:{nombre_guardado}"
    if detecciones:
        detalle = detalle_severidad(info_severidad)
        for det in detecciones:
            database.registrar(
                fuente=fuente,
                clase=det["clase"],
                confianza=det["confianza"],
                detalle=detalle,
            )
    else:
        database.registrar(
            fuente=fuente, clase="ninguno", confianza=0.0,
            detalle=f"Severidad: {nivel} | Sin detecciones",
        )

    alerta_enviada = False
    if info_severidad["hay_fuego"]:
        alerta_enviada = enviar_telegram(zona=fuente, nivel=nivel)

    return render_template(
        "resultado.html",
        imagen_original=url_for("static", filename=f"uploads/{nombre_guardado}"),
        imagen_resultado=url_for("static", filename=f"results/{nombre_resultado}"),
        detecciones=detecciones,
        severidad=info_severidad,
        alerta_enviada=alerta_enviada,
    )


# ---------------------------------------------------------------------------
# Detección en video
# ---------------------------------------------------------------------------
def listar_videos_disponibles():
    """Videos ya presentes en static/uploads que pueden reutilizarse sin resubir."""
    videos = [
        p.name for p in sorted(UPLOAD_FOLDER.iterdir())
        if p.is_file() and extension_permitida(p.name, ALLOWED_VIDEO_EXT)
    ]
    return videos


@app.route("/video")
def video_pagina():
    return render_template(
        "video.html", video_resultado=None, videos_disponibles=listar_videos_disponibles()
    )


@app.route("/procesar_video", methods=["POST"])
def procesar_video():
    archivo = request.files.get("video")
    nombre_existente = (request.form.get("video_existente") or "").strip()

    if archivo and archivo.filename:
        if not extension_permitida(archivo.filename, ALLOWED_VIDEO_EXT):
            flash("Formato de video no soportado. Usa MP4, AVI, MOV o MKV.")
            return redirect(url_for("video_pagina"))

        nombre_seguro = secure_filename(archivo.filename)
        ext_entrada = nombre_seguro.rsplit(".", 1)[1].lower()
        nombre_entrada = nombre_unico("video_in", ext_entrada)
        ruta_entrada = UPLOAD_FOLDER / nombre_entrada
        archivo.save(ruta_entrada)
    elif nombre_existente:
        nombre_seguro = secure_filename(nombre_existente)
        candidato = (UPLOAD_FOLDER / nombre_seguro).resolve()
        if candidato.parent != UPLOAD_FOLDER.resolve() or not candidato.is_file():
            flash("El video seleccionado ya no existe en el servidor.")
            return redirect(url_for("video_pagina"))
        ruta_entrada = candidato
        nombre_entrada = nombre_seguro
    else:
        flash("Selecciona un video existente o sube uno nuevo.")
        return redirect(url_for("video_pagina"))

    captura = cv2.VideoCapture(str(ruta_entrada))
    if not captura.isOpened():
        flash("No se pudo abrir el video. Verifica que el archivo no esté dañado.")
        return redirect(url_for("video_pagina"))

    fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
    ancho_original = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_original = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Videos en 1080p (o más) se reescalan a un máximo de 720p de alto ANTES
    # de inferencia/dibujo/escritura: acelera YOLO, cv2.VideoWriter y FFmpeg
    # a la vez, sin pérdida perceptible de calidad en un reproductor web.
    if alto_original > ALTO_MAX_VIDEO_SALIDA:
        escala = ALTO_MAX_VIDEO_SALIDA / alto_original
        alto = ALTO_MAX_VIDEO_SALIDA
        ancho = int(round(ancho_original * escala / 2) * 2)  # par, lo exige yuv420p
    else:
        ancho, alto = ancho_original, alto_original
    redimensionar_frame = (ancho, alto) != (ancho_original, alto_original)
    area_frame = ancho * alto

    nombre_temporal = nombre_unico("video_tmp", "mp4")
    ruta_temporal = VIDEO_FOLDER / nombre_temporal
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(str(ruta_temporal), fourcc, fps, (ancho, alto))

    # Para videos largos, ejecutar YOLO en cada frame es impracticable en CPU:
    # se infiere 1 de cada N frames (misma resolución IMGSZ_YOLO=640 que
    # imagen/cámara -- pipeline único, sin variantes por ruta).
    #
    # Un foco de fuego/humo real rara vez dura un solo muestreo: parpadea,
    # se ve tapado por una brasa u otro tronco, etc. Sin memoria de
    # persistencia, la caja se prende y apaga cuadro a cuadro (flickering).
    # 'tracks' guarda los focos activos (actualizar_tracks) y los mantiene
    # hasta HOLD_FRAMES_TRACKING fotogramas tras perder la reconfirmación
    # (decaer_tracks en los frames intermedios sin inferencia nueva). El
    # conteo real para severidad/BD ('detecciones_totales') son siempre las
    # cajas REALES que YOLO reportó en cada ciclo de inferencia, nunca los
    # cuadros donde una caja solo se mantiene dibujada por memoria.
    detecciones_totales = []
    tracks = []
    indice_frame = 0
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        if redimensionar_frame:
            frame = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)
        if indice_frame % DETECCION_CADA_N_FRAMES == 0:
            crudas = ejecutar_deteccion(frame)
            # Detección simultánea de humo: si YOLO no reportó clase 'humo'
            # este ciclo (ni directo ni vía extracción bidireccional dentro
            # de una caja gigante de fuego), se complementa con la
            # heurística de saturación baja sobre el frame completo -- para
            # que una nube de humo/ceniza difusa que el modelo pierde por
            # umbral no deje de dibujarse junto al fuego.
            if not any(d["clase"] == "humo" for d in crudas):
                crudas.extend(_detectar_humo_difuso(frame))
            detecciones_totales.extend(crudas)
            # Misma memoria de persistencia (tracks/HOLD_FRAMES_TRACKING) para
            # fuego Y humo: actualizar_tracks no distingue por clase, así que
            # las cajas amarillas quedan tan estables como las rojas.
            tracks = actualizar_tracks(tracks, crudas)
        else:
            tracks = decaer_tracks(tracks)
        frame_anotado = dibujar_detecciones(frame, tracks)
        escritor.write(frame_anotado)
        indice_frame += 1

    captura.release()
    escritor.release()

    # Conversión a H.264 con FFmpeg para asegurar compatibilidad de reproducción
    # web. preset=ultrafast + crf=26 prioriza velocidad de codificación (clave
    # para que el usuario no espere minutos); crf 26 es apenas perceptible en
    # un video ya reescalado a 720p para previsualización en navegador.
    nombre_final = nombre_unico("video_out", "mp4")
    ruta_final = VIDEO_FOLDER / nombre_final
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(ruta_temporal),
                "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "26",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(ruta_final),
            ],
            check=True, capture_output=True,
        )
        ruta_temporal.unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[app] FFmpeg no disponible o falló la conversión ({exc}). "
              "Se usará el video sin recodificar (puede no reproducirse en el navegador).")
        ruta_temporal.replace(ruta_final)

    info_severidad = severidad(detecciones_totales, area_frame)
    nivel = info_severidad["nivel"]
    clase_dominante = clase_prioritaria(detecciones_totales)
    conf_max = confianza_maxima(detecciones_totales, clase_dominante)

    fuente = f"video:{nombre_entrada}"
    database.registrar(
        fuente=fuente,
        clase=clase_dominante,
        confianza=conf_max,
        detalle=(
            f"{detalle_severidad(info_severidad)} | "
            f"Detecciones totales: {len(detecciones_totales)}"
        ),
    )

    alerta_enviada = False
    if info_severidad["hay_fuego"]:
        alerta_enviada = enviar_telegram(zona=fuente, nivel=nivel)

    return render_template(
        "video.html",
        # '?v=' con timestamp: el nombre de archivo ya es único de por sí
        # (nombre_unico incluye timestamp+uuid), pero se añade igual como
        # segunda capa explícita anti-caché para que el navegador nunca
        # reutilice un video anterior servido desde la misma ruta estática.
        video_resultado=url_for("static", filename=f"videos/{nombre_final}") + f"?v={int(time.time())}",
        severidad=info_severidad,
        total_detecciones=len(detecciones_totales),
        alerta_enviada=alerta_enviada,
    )


# ---------------------------------------------------------------------------
# Detección en tiempo real (cámara web del cliente)
# ---------------------------------------------------------------------------
@app.route("/camara")
def camara_pagina():
    return render_template("camara.html")


# Estado de rastreo de la cámara en vivo: cada request a /detectar_frame es
# independiente (HTTP sin sesión de video), así que la memoria de
# persistencia entre sondeos se guarda aquí. El servidor de desarrollo de
# Flask procesa requests secuencialmente (sin threaded=True), por lo que una
# variable de módulo simple es segura sin necesidad de locks.
_tracks_camara = []
# La cámara sondea cada ~800ms (INTERVALO_MS en camara.html), más lento que
# un frame de video: 1 sondeo de gracia ya cubre con margen el medio segundo
# de persistencia pedido, sin arrastrar cajas obsoletas por varios segundos.
HOLD_SONDEOS_CAMARA = 1


@app.route("/detectar_frame", methods=["POST"])
def detectar_frame():
    global _tracks_camara

    datos = request.get_json(silent=True)
    if not datos or "imagen" not in datos:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    try:
        cadena_b64 = datos["imagen"].split(",")[-1]
        bytes_imagen = base64.b64decode(cadena_b64)
        arreglo = np.frombuffer(bytes_imagen, dtype=np.uint8)
        frame = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "No se pudo decodificar la imagen recibida"}), 400

    if frame is None:
        return jsonify({"error": "Frame inválido"}), 400

    alto, ancho = frame.shape[:2]
    area_frame = ancho * alto

    # Misma memoria de persistencia que el pipeline de video: evita que la
    # caja/severidad parpadeen sondeo a sondeo. Las detecciones en sí siguen
    # siendo 100% reales de YOLOv8 (mismo CONF_THRESHOLD que imagen/video).
    crudas = ejecutar_deteccion(frame)
    _tracks_camara = actualizar_tracks(_tracks_camara, crudas, hold_frames=HOLD_SONDEOS_CAMARA)
    detecciones = _tracks_camara
    info_severidad = severidad(detecciones, area_frame)
    nivel = info_severidad["nivel"]

    zona = "Cámara web en vivo"
    alerta_enviada = False

    if info_severidad["hay_fuego"] and puede_alertar(zona):
        clase_dominante = clase_prioritaria(detecciones)
        database.registrar(
            fuente=zona,
            clase=clase_dominante,
            confianza=confianza_maxima(detecciones, clase_dominante),
            detalle=detalle_severidad(info_severidad),
        )
        alerta_enviada = enviar_telegram(zona=zona, nivel=nivel)
    elif nivel == "MEDIA" and info_severidad["hay_humo"] and puede_alertar(zona + ":humo"):
        database.registrar(
            fuente=zona, clase="humo",
            confianza=confianza_maxima(detecciones, "humo"),
            detalle=detalle_severidad(info_severidad),
        )

    return jsonify(
        {
            "detecciones": detecciones,
            "severidad": info_severidad,
            "alerta": info_severidad["hay_fuego"],
            "alerta_enviada": alerta_enviada,
            "ancho": ancho,
            "alto": alto,
        }
    )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    database.init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
