"""
Script de entrenamiento y evaluación del modelo YOLOv8 para detección
temprana de fuego y humo en almacenes.

Uso:
    python entrenar.py

Requiere que el dataset esté organizado según `dataset/data.yaml`
(carpetas images/train, images/val, labels/train, labels/val, etc.).
"""

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
MODELOS_DIR = BASE_DIR / "models"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparámetros de entrenamiento
MODELO_BASE = "yolov8n.pt"   # modelo preentrenado usado como punto de partida
EPOCHS = 60
IMGSZ = 640
BATCH = 16
PROJECT_DIR = str(BASE_DIR / "runs")
NOMBRE_EXPERIMENTO = "fuego_humo_yolov8"


def entrenar():
    print("=" * 70)
    print("ENTRENAMIENTO YOLOv8 - Detección de fuego y humo en almacenes")
    print("=" * 70)
    print(f"Dataset: {DATA_YAML}")
    print(f"Modelo base: {MODELO_BASE}")
    print(f"Epochs: {EPOCHS} | imgsz: {IMGSZ} | batch: {BATCH}")
    print("-" * 70)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de dataset: {DATA_YAML}. "
            "Verifica que 'dataset/data.yaml' exista y esté configurado."
        )

    modelo = YOLO(MODELO_BASE)

    resultados_entrenamiento = modelo.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT_DIR,
        name=NOMBRE_EXPERIMENTO,
        patience=15,
        exist_ok=True,
    )

    mejor_modelo = Path(resultados_entrenamiento.save_dir) / "weights" / "best.pt"
    if mejor_modelo.exists():
        destino = MODELOS_DIR / "best.pt"
        destino.write_bytes(mejor_modelo.read_bytes())
        print(f"\nMejor modelo copiado a: {destino}")
    else:
        print(f"\n[AVISO] No se encontró el archivo best.pt en {mejor_modelo}")

    return modelo, mejor_modelo


def evaluar(modelo: YOLO):
    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL MODELO (conjunto de validación)")
    print("=" * 70)

    metricas = modelo.val(data=str(DATA_YAML), imgsz=IMGSZ)

    map50 = float(metricas.box.map50)
    map50_95 = float(metricas.box.map)
    precision = float(metricas.box.mp)
    recall = float(metricas.box.mr)

    print("-" * 70)
    print(f"mAP50      : {map50:.4f}")
    print(f"mAP50-95   : {map50_95:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print("-" * 70)

    return {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "Precision": precision,
        "Recall": recall,
    }


if __name__ == "__main__":
    modelo_entrenado, ruta_best = entrenar()
    evaluar(modelo_entrenado)
