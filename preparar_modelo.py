"""
Script de preparación del modelo YOLOv8 para detección de fuego y humo.
SENATI - Curso PIAD-426 - Casuística 3: Seguridad industrial

Objetivo
--------
Garantizar que exista un archivo `models/best.pt` válido, cargable con
`YOLO('models/best.pt')` y cuyas clases sean exactamente:
    {0: "fuego", 1: "humo"}

De modo que `app.py` deje de mostrar la advertencia de "modelo de pruebas"
y utilice siempre el modelo correcto.

Estrategia (en orden de prioridad)
-----------------------------------
1. Si `models/best.pt` ya existe y es válido (clases correctas), no hace
   nada y termina.
2. Si la variable de entorno `FIRE_SMOKE_WEIGHTS_URL` apunta a unos pesos
   YOLOv8 propios (por ejemplo, alojados por el usuario tras entrenar con
   `entrenar.py` y subirlos a su propio storage), los descarga y verifica.
   Esto es opcional: por defecto no se descarga nada de fuentes externas
   no verificadas.
3. En su defecto (caso por omisión, sin configuración adicional):
   - Genera automáticamente un dataset sintético de calibración con
     siluetas de fuego y humo sobre fondos tipo almacén industrial
     (imágenes + etiquetas en formato YOLO).
   - Entrena un modelo YOLOv8n breve (pocas épocas) partiendo de los
     pesos preentrenados `yolov8n.pt`, usando `dataset/data.yaml`
     (0: fuego, 1: humo).
   - Copia el `best.pt` resultante a `models/best.pt`.
4. Verifica el resultado final cargando el modelo y comparando sus clases.

IMPORTANTE: el modelo generado en el paso 3 se entrena con datos
SINTÉTICOS y pocas épocas. Sirve para validar el pipeline completo
end-to-end (carga, inferencia, dibujo, severidad, alertas) con las
clases correctas, pero NO sustituye un entrenamiento real. Para
precisión de producción, ejecuta `entrenar.py` con un dataset real de
imágenes de fuego y humo en almacenes.

Uso:
    python preparar_modelo.py
"""

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELO_DESTINO = MODELS_DIR / "best.pt"
RUNS_DIR = BASE_DIR / "runs_preparacion"

NOMBRES_CLASES = {0: "fuego", 1: "humo"}

# Parámetros del dataset sintético de calibración
IMG_SIZE = 320
N_TRAIN = 90
N_VAL = 20
EPOCHS = 25
BATCH = 16
SEED = 42

URL_PESOS_PERSONALIZADOS = os.environ.get("FIRE_SMOKE_WEIGHTS_URL", "").strip()


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------
def modelo_ya_valido() -> bool:
    if not MODELO_DESTINO.exists():
        return False
    try:
        modelo = YOLO(str(MODELO_DESTINO))
        nombres = {int(k): v for k, v in modelo.names.items()}
        return nombres == NOMBRES_CLASES
    except Exception as exc:
        print(f"[preparar_modelo] 'models/best.pt' existe pero no es válido ({exc}).")
        return False


# ---------------------------------------------------------------------------
# Paso opcional: descarga de pesos propios del usuario
# ---------------------------------------------------------------------------
def intentar_descarga_personalizada() -> bool:
    if not URL_PESOS_PERSONALIZADOS:
        return False

    print(f"[preparar_modelo] Variable FIRE_SMOKE_WEIGHTS_URL detectada. "
          f"Intentando descargar pesos personalizados desde:\n  {URL_PESOS_PERSONALIZADOS}")
    try:
        import requests

        respuesta = requests.get(URL_PESOS_PERSONALIZADOS, timeout=60)
        respuesta.raise_for_status()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        MODELO_DESTINO.write_bytes(respuesta.content)

        if modelo_ya_valido():
            print("[preparar_modelo] Pesos personalizados descargados y verificados correctamente.")
            return True

        print("[preparar_modelo] Los pesos descargados no tienen las clases esperadas "
              f"{NOMBRES_CLASES}. Se descarta el archivo y se continúa con el plan de respaldo.")
        MODELO_DESTINO.unlink(missing_ok=True)
        return False
    except Exception as exc:
        print(f"[preparar_modelo] No se pudo descargar/verificar el modelo personalizado: {exc}")
        MODELO_DESTINO.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Generación de dataset sintético de calibración (fuego / humo)
# ---------------------------------------------------------------------------
def fondo_almacen(ancho: int, alto: int, rng: np.random.Generator) -> np.ndarray:
    base = int(rng.integers(45, 70))
    frame = np.full((alto, ancho, 3), (base, base - 5, base - 10), dtype=np.uint8)

    piso_y = int(alto * 0.72)
    cv2.rectangle(frame, (0, piso_y), (ancho, alto), (70, 90, 100), -1)

    x = int(rng.integers(5, 20))
    while x < ancho - 40:
        w_rack = int(rng.integers(45, 90))
        if rng.random() < 0.65:
            color = (
                int(rng.integers(35, 55)),
                int(rng.integers(45, 65)),
                int(rng.integers(55, 80)),
            )
            alto_rack = int(rng.integers(90, 170))
            cv2.rectangle(frame, (x, max(0, piso_y - alto_rack)), (x + w_rack, piso_y), color, -1)
            cv2.rectangle(frame, (x, max(0, piso_y - alto_rack)), (x + w_rack, piso_y), (25, 30, 35), 2)
        x += w_rack + int(rng.integers(15, 40))

    ruido = rng.normal(0, 4, frame.shape)
    frame = np.clip(frame.astype(np.int16) + ruido.astype(np.int16), 0, 255).astype(np.uint8)
    return frame


def dibujar_fuego(frame: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)

    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (0, 60, 220), -1)                        # rojo (BGR)
    cv2.ellipse(frame, (cx, cy + ay // 6), (int(ax * 0.65), int(ay * 0.70)), 0, 0, 360, (0, 130, 255), -1)   # naranja
    cv2.ellipse(frame, (cx, cy + ay // 3), (int(ax * 0.35), int(ay * 0.40)), 0, 0, 360, (40, 220, 255), -1)  # amarillo
    return frame


def dibujar_humo(frame: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)

    overlay = frame.copy()
    tono = 110
    cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, (tono, tono, tono), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    return frame


def generar_imagen(ancho: int, alto: int, rng: np.random.Generator):
    frame = fondo_almacen(ancho, alto, rng)
    objetos = []

    # ~15% de imágenes negativas (sin fuego ni humo) para reducir falsos positivos
    if rng.random() < 0.15:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame, objetos

    n_objetos = int(rng.integers(1, 4))
    intentos = 0
    while len(objetos) < n_objetos and intentos < 20:
        intentos += 1
        clase = int(rng.integers(0, 2))  # 0 fuego, 1 humo
        w = int(rng.integers(40, 100))
        h = int(rng.integers(50, 120))
        if w >= ancho - 10 or h >= alto - 10:
            continue
        cx = int(rng.integers(w // 2 + 5, ancho - w // 2 - 5))
        cy = int(rng.integers(h // 2 + 5, alto - h // 2 - 5))
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2

        if clase == 0:
            frame = dibujar_fuego(frame, (x1, y1, x2, y2))
        else:
            frame = dibujar_humo(frame, (x1, y1, x2, y2))
        objetos.append((clase, x1, y1, x2, y2))

    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame, objetos


def generar_dataset():
    rng = np.random.default_rng(SEED)

    for split, cantidad in (("train", N_TRAIN), ("val", N_VAL)):
        dir_img = DATASET_DIR / "images" / split
        dir_lbl = DATASET_DIR / "labels" / split
        dir_img.mkdir(parents=True, exist_ok=True)
        dir_lbl.mkdir(parents=True, exist_ok=True)

        # Limpia únicamente archivos sintéticos generados por este script en corridas previas
        for viejo in dir_img.glob("sint_*.jpg"):
            viejo.unlink()
        for viejo in dir_lbl.glob("sint_*.txt"):
            viejo.unlink()

        for i in range(cantidad):
            frame, objetos = generar_imagen(IMG_SIZE, IMG_SIZE, rng)
            nombre = f"sint_{split}_{i:04d}"
            cv2.imwrite(str(dir_img / f"{nombre}.jpg"), frame)
            with open(dir_lbl / f"{nombre}.txt", "w") as f:
                for clase, x1, y1, x2, y2 in objetos:
                    xc = ((x1 + x2) / 2) / IMG_SIZE
                    yc = ((y1 + y2) / 2) / IMG_SIZE
                    w_n = (x2 - x1) / IMG_SIZE
                    h_n = (y2 - y1) / IMG_SIZE
                    f.write(f"{clase} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")

    print(f"[preparar_modelo] Dataset sintético generado: {N_TRAIN} imágenes de entrenamiento "
          f"y {N_VAL} de validación en '{DATASET_DIR}'.")


# ---------------------------------------------------------------------------
# Entrenamiento breve del modelo base
# ---------------------------------------------------------------------------
def entrenar_modelo_base() -> Path:
    print(f"[preparar_modelo] Entrenando modelo base YOLOv8n ({EPOCHS} epochs, "
          f"imgsz={IMG_SIZE})... esto puede tardar varios minutos.")

    modelo = YOLO("yolov8n.pt")
    resultados = modelo.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS_DIR),
        name="fuego_humo_base",
        exist_ok=True,
        patience=0,
        seed=SEED,
        workers=0,
        verbose=False,
    )
    return Path(resultados.save_dir) / "weights" / "best.pt"


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if modelo_ya_valido():
        print(f"[preparar_modelo] 'models/best.pt' ya existe y es válido {NOMBRES_CLASES}. Nada que hacer.")
        return

    if intentar_descarga_personalizada():
        return

    print("[preparar_modelo] No hay pesos personalizados configurados (FIRE_SMOKE_WEIGHTS_URL vacío) "
          "o la descarga falló.")
    print("[preparar_modelo] Generando y entrenando un modelo base de calibración con dataset sintético...")

    generar_dataset()
    ruta_best = entrenar_modelo_base()

    if not ruta_best.exists():
        print(f"[preparar_modelo] ERROR: no se generó el archivo esperado en '{ruta_best}'.")
        sys.exit(1)

    shutil.copyfile(ruta_best, MODELO_DESTINO)

    if modelo_ya_valido():
        print(f"[preparar_modelo] 'models/best.pt' generado y verificado correctamente. Clases: {NOMBRES_CLASES}")
        print("[preparar_modelo] NOTA: este modelo se calibró con un dataset sintético reducido, pensado para")
        print("                   validar el pipeline completo (carga, inferencia, severidad, alertas) con las")
        print("                   clases correctas. Para precisión de producción, ejecuta 'entrenar.py' con un")
        print("                   dataset real de imágenes de fuego y humo en almacenes.")
    else:
        print("[preparar_modelo] ERROR: el modelo generado no superó la verificación final de clases.")
        sys.exit(1)


if __name__ == "__main__":
    main()
