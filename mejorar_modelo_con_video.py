"""
Calibración adicional de models/best.pt usando fotogramas REALES extraídos
de un video propio del usuario (por defecto, static/uploads/prueba_fuego.mp4).

Por qué existe este script
---------------------------
El modelo generado por defecto en 'preparar_modelo.py' se entrena únicamente
con siluetas sintéticas (elipses de color) y generaliza mal a fotografías
reales de incendios. Para mejorar esto SIN depender de pesos de terceros no
verificados (que representan un riesgo real de seguridad: un archivo .pt es
un pickle de PyTorch que puede ejecutar código arbitrario al cargarse si
proviene de un origen no confiable), este script extrae fotogramas reales
de un video de incendio que el propio usuario ya posee localmente, genera
etiquetas YOLO de forma automática mediante heurísticas de color/diferencia
de fondo, y los añade al dataset de entrenamiento antes de recalibrar el
modelo. Todo el contenido usado es data (imágenes), nunca código ejecutable
de un tercero.

Limitación honesta: al provenir de un único video/escenario, el modelo
puede aprender también pistas del fondo (esa habitación específica) además
de la apariencia del fuego/humo. Sigue siendo una mejora real frente a
partir de cero con siluetas sintéticas, pero no reemplaza un dataset
diverso de imágenes reales de distintos almacenes/escenarios.

Uso:
    python mejorar_modelo_con_video.py
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
MODELO_ACTUAL = MODELS_DIR / "best.pt"
RUNS_DIR = BASE_DIR / "runs_calibracion_real"

VIDEO_FUENTE = BASE_DIR / "static" / "uploads" / "prueba_fuego.mp4"
SEGUNDO_BASELINE = 1.0
TIMESTAMPS_MUESTRA = list(range(10, 240, 8))  # cada 8s desde 10s hasta 240s

NOMBRES_CLASES = {0: "fuego", 1: "humo"}
IMG_SIZE = 640
EPOCHS_FINETUNE = 20
BATCH = 8
AREA_MINIMA_CONTORNO = 250  # px^2, descarta ruido pequeño


def leer_frame_en_segundo(captura: cv2.VideoCapture, segundo: float):
    captura.set(cv2.CAP_PROP_POS_MSEC, segundo * 1000)
    ok, frame = captura.read()
    return frame if ok else None


ALTO_FRANJA_RELOJ = 45  # px superiores donde la cámara imprime fecha/hora; se ignora
FRACCION_MAX_CAJA = 0.55  # descarta cajas que cubren casi toda la escena (ruido, no un foco localizado)


def segmentar_fuego_humo(frame: np.ndarray, baseline: np.ndarray):
    """
    Heurística de auto-etiquetado sobre cámara fija:
    - Se calcula la diferencia respecto al fondo limpio (baseline) para
      aislar únicamente lo que cambió en la escena (el incendio y su humo).
    - Dentro de esa región de cambio, los píxeles cálidos, saturados y
      brillantes se clasifican como 'fuego' (se exige saturación mínima
      para no confundir una ventana/luz blanca sobreexpuesta con la llama).
    - El resto de la región cambiada, grisácea y difusa, se clasifica
      como 'humo'.
    Retorna una lista de (clase, x1, y1, x2, y2) en píxeles.
    """
    alto, ancho = frame.shape[:2]

    gris_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris_base = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gris_frame, gris_base)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mascara_cambio = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    mascara_cambio = cv2.dilate(mascara_cambio, np.ones((7, 7), np.uint8), iterations=2)
    mascara_cambio[:ALTO_FRANJA_RELOJ, :] = 0  # ignora el overlay de fecha/hora

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # NOTA: en este video CCTV, la sobreexposición del núcleo de la llama
    # produce artefactos de color (franjas violetas/azuladas), por lo que
    # el matiz (H) NO es confiable para aislar el fuego. En su lugar se usa
    # brillo extremo + cambio temporal fuerte (parpadeo real de la llama,
    # a diferencia de un objeto estático que solo varía por auto-exposición).
    brillo_extremo = (v >= 215).astype(np.uint8) * 255
    cambio_fuerte = (diff >= 35).astype(np.uint8) * 255
    mascara_fuego = cv2.bitwise_and(brillo_extremo, cambio_fuerte)
    mascara_fuego = cv2.bitwise_and(mascara_fuego, mascara_cambio)
    mascara_fuego = cv2.morphologyEx(mascara_fuego, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mascara_fuego = cv2.dilate(mascara_fuego, np.ones((7, 7), np.uint8), iterations=1)

    # Humo: región que cambió, con brillo moderado (ni el núcleo
    # sobreexpuesto del fuego ni una sombra oscura) y difusa.
    brillo_moderado = ((v >= 55) & (v <= 210)).astype(np.uint8) * 255
    mascara_humo = cv2.bitwise_and(mascara_cambio, brillo_moderado)
    mascara_humo = cv2.bitwise_and(mascara_humo, cv2.bitwise_not(cv2.dilate(mascara_fuego, np.ones((15, 15), np.uint8))))
    mascara_humo = cv2.morphologyEx(mascara_humo, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    objetos = []
    area_frame = ancho * alto
    for clase, mascara in ((0, mascara_fuego), (1, mascara_humo)):
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = [c for c in contornos if cv2.contourArea(c) >= AREA_MINIMA_CONTORNO]
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:2]
        for c in contornos:
            x, y, w, h_ = cv2.boundingRect(c)
            if w >= FRACCION_MAX_CAJA * ancho and h_ >= FRACCION_MAX_CAJA * alto:
                continue  # caja casi del tamaño completo del frame: ruido, no un foco localizado
            if (w * h_) / area_frame < 0.0015:
                continue  # demasiado pequeña para ser una detección útil
            if clase == 0 and float(diff[y:y + h_, x:x + w].mean()) < 28:
                continue  # cambio temporal débil: probablemente un objeto estático (p. ej. una ventana), no llama
            objetos.append((clase, x, y, x + w, y + h_))

    return objetos


def extraer_y_etiquetar():
    if not VIDEO_FUENTE.exists():
        print(f"[mejorar_modelo] No se encontró el video fuente: {VIDEO_FUENTE}")
        return 0

    captura = cv2.VideoCapture(str(VIDEO_FUENTE))
    if not captura.isOpened():
        print(f"[mejorar_modelo] No se pudo abrir el video: {VIDEO_FUENTE}")
        return 0

    baseline = leer_frame_en_segundo(captura, SEGUNDO_BASELINE)
    if baseline is None:
        print("[mejorar_modelo] No se pudo leer el frame base.")
        return 0

    dir_img_train = DATASET_DIR / "images" / "train"
    dir_lbl_train = DATASET_DIR / "labels" / "train"
    dir_img_val = DATASET_DIR / "images" / "val"
    dir_lbl_val = DATASET_DIR / "labels" / "val"
    for d in (dir_img_train, dir_lbl_train, dir_img_val, dir_lbl_val):
        d.mkdir(parents=True, exist_ok=True)
        for viejo in d.glob("real_*"):
            viejo.unlink()

    total = 0
    for i, seg in enumerate(TIMESTAMPS_MUESTRA):
        frame = leer_frame_en_segundo(captura, seg)
        if frame is None:
            continue
        objetos = segmentar_fuego_humo(frame, baseline)
        if not objetos:
            continue  # solo se conservan frames con al menos una detección útil

        alto, ancho = frame.shape[:2]
        es_val = (i % 6 == 0)  # ~1 de cada 6 va a validación
        dir_img = dir_img_val if es_val else dir_img_train
        dir_lbl = dir_lbl_val if es_val else dir_lbl_train

        nombre = f"real_{seg:04d}s"
        cv2.imwrite(str(dir_img / f"{nombre}.jpg"), frame)
        with open(dir_lbl / f"{nombre}.txt", "w") as f:
            for clase, x1, y1, x2, y2 in objetos:
                xc = ((x1 + x2) / 2) / ancho
                yc = ((y1 + y2) / 2) / alto
                w_n = (x2 - x1) / ancho
                h_n = (y2 - y1) / alto
                f.write(f"{clase} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")
        total += 1

    captura.release()
    print(f"[mejorar_modelo] {total} fotogramas reales etiquetados y añadidos al dataset "
          f"(prefijo 'real_') a partir de '{VIDEO_FUENTE.name}'.")
    return total


def recalibrar_modelo():
    modelo_base = str(MODELO_ACTUAL) if MODELO_ACTUAL.exists() else "yolov8n.pt"
    print(f"[mejorar_modelo] Recalibrando desde '{modelo_base}' con {EPOCHS_FINETUNE} epochs adicionales...")

    modelo = YOLO(modelo_base)
    resultados = modelo.train(
        data=str(DATA_YAML),
        epochs=EPOCHS_FINETUNE,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS_DIR),
        name="fuego_humo_real",
        exist_ok=True,
        patience=0,
        seed=42,
        workers=0,
        verbose=False,
    )
    return Path(resultados.save_dir) / "weights" / "best.pt"


def verificar(modelo_path: Path) -> bool:
    try:
        m = YOLO(str(modelo_path))
        nombres = {int(k): v for k, v in m.names.items()}
        return nombres == NOMBRES_CLASES
    except Exception as exc:
        print(f"[mejorar_modelo] Verificación fallida: {exc}")
        return False


def main():
    total = extraer_y_etiquetar()
    if total == 0:
        print("[mejorar_modelo] No se generaron fotogramas reales etiquetados; se aborta la recalibración.")
        return

    ruta_best = recalibrar_modelo()
    if not ruta_best.exists() or not verificar(ruta_best):
        print("[mejorar_modelo] ERROR: el modelo recalibrado no superó la verificación. "
              "Se conserva el 'models/best.pt' anterior sin cambios.")
        return

    respaldo = MODELS_DIR / "best_anterior.pt"
    if MODELO_ACTUAL.exists():
        shutil.copyfile(MODELO_ACTUAL, respaldo)
    shutil.copyfile(ruta_best, MODELO_ACTUAL)
    print(f"[mejorar_modelo] 'models/best.pt' actualizado con la calibración sobre datos reales.")
    print(f"[mejorar_modelo] Copia del modelo anterior guardada en '{respaldo.name}' por si se desea revertir.")


if __name__ == "__main__":
    main()
