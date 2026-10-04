<div align="center">

# 🔥 Fire & Smoke Detection AI

**Sistema de detección temprana de fuego y humo en almacenes con visión por computadora (YOLOv8) y panel web de monitoreo.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)

</div>

---

## 📋 Descripción

**Fire & Smoke Detection AI** es una aplicación web orientada a la **seguridad industrial** que analiza imágenes, videos y la cámara en vivo para identificar **fuego** y **humo** en entornos de almacén antes de que un incidente se propague.

El núcleo del sistema es un modelo **YOLOv8** entrenado con dos clases (`fuego`, `humo`), complementado con heurísticas de color y densidad en OpenCV para reforzar la detección de humo difuso y llamas parcialmente visibles. Cada detección se clasifica por **nivel de severidad** (BAJA / MEDIA / ALTA), se registra en una base de datos **SQLite**, dispara una **alarma sonora y por voz** en el navegador y, opcionalmente, envía una **notificación a Telegram**.

Todo se gestiona desde un **panel web en Flask** con estadísticas, historial de eventos y exportación de reportes a Excel.

> Proyecto desarrollado en el marco del curso **PIAD-426 (SENATI)** — Casuística 3: Seguridad industrial.

---

## ✨ Características clave

### 🎯 Detección en tiempo real
- Análisis de **imágenes** (`png`, `jpg`, `jpeg`, `bmp`), **videos** (`mp4`, `avi`, `mov`, `mkv`) y **cámara web en vivo** desde el navegador.
- Modelo **YOLOv8** con umbral de confianza ajustado para alta sensibilidad.
- **Seguimiento de detecciones** entre fotogramas (tracking por IoU) para reducir parpadeos y falsos negativos en video.
- Heurísticas complementarias de **color (fuego)** y **densidad/difusión (humo)** con OpenCV.

### 📊 Panel de control
- Dashboard con **total de eventos**, **resumen por clase** y **últimos eventos** registrados.
- Visualización de resultados con *bounding boxes* y nivel de confianza por detección.
- Historial de videos procesados disponible para revisión.

### 🚨 Módulo de alertas
- Cálculo de **severidad** según el área ocupada por el fuego/humo y la confianza del modelo.
- **Alarma sonora** (Web Audio API) y **advertencia hablada en español** (Web Speech API).
- Notificaciones vía **bot de Telegram**, con control de *cooldown* por zona para evitar spam.

### 📑 Reportes
- Registro persistente de cada evento (fecha, fuente, clase, confianza, detalle) en **SQLite**.
- Exportación del historial completo a **Excel (`.xlsx`)** con formato y colores por severidad.

---

## 🛠️ Stack tecnológico

| Capa | Tecnologías |
|------|-------------|
| **Lenguaje** | Python 3 |
| **IA / Visión** | YOLOv8 (Ultralytics), OpenCV, NumPy, Pillow |
| **Backend** | Flask |
| **Frontend** | HTML5, CSS3, JavaScript (Jinja2, Web Audio API, Web Speech API, MediaDevices API) |
| **Base de datos** | SQLite |
| **Reportes** | openpyxl |
| **Notificaciones** | Telegram Bot API (`requests`) |

---

## 📁 Estructura del proyecto

```text
fire-smoke-detection-ai/
├── app.py                        # Servidor Flask: rutas, inferencia YOLOv8, tracking y heurísticas
├── alertas.py                    # Cálculo de severidad y envío de alertas a Telegram
├── database.py                   # Persistencia de eventos en SQLite
├── reportes.py                   # Exportación del historial a Excel (.xlsx)
├── entrenar.py                   # Entrenamiento y evaluación del modelo YOLOv8
├── preparar_modelo.py            # Genera/valida models/best.pt (dataset sintético + entrenamiento)
├── mejorar_modelo_con_video.py   # Recalibración del modelo con fotogramas reales de video
├── requirements.txt              # Dependencias del proyecto
├── historial.db                  # Base de datos SQLite de eventos
├── yolov8n.pt                    # Pesos base preentrenados de YOLOv8n
│
├── dataset/
│   ├── data.yaml                 # Configuración del dataset (0: fuego, 1: humo)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── models/
│   └── best.pt                   # Modelo entrenado usado por la aplicación
│
├── runs_preparacion/             # Métricas y curvas del entrenamiento base
├── runs_calibracion_real/        # Métricas de la recalibración con video real
│
├── static/
│   ├── js/
│   │   └── alarma.js             # Sirena + aviso por voz en el navegador
│   ├── uploads/                  # Archivos subidos por el usuario
│   ├── results/                  # Imágenes procesadas con detecciones
│   └── videos/                   # Videos procesados con detecciones
│
└── templates/
    ├── base.html                 # Plantilla base
    ├── index.html                # Inicio / carga de imágenes
    ├── resultado.html            # Resultado de la detección en imagen
    ├── video.html                # Procesamiento de video
    ├── camara.html               # Detección con cámara en vivo
    └── dashboard.html            # Panel de control y estadísticas
```

---

## 🚀 Instalación

### Requisitos previos
- **Python 3.10 – 3.12** (recomendado)
- `pip` y `git`
- *(Opcional)* GPU NVIDIA con CUDA para acelerar inferencia y entrenamiento
- *(Opcional)* **FFmpeg** en el `PATH` para mejor compatibilidad de los videos procesados en el navegador

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/lecarnaquemauricioangel-cmd/fire-smoke-detection-ai.git
cd fire-smoke-detection-ai

# 2. Crear y activar un entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Configuración de alertas por Telegram (opcional)

Define las variables de entorno con el token de tu bot y el ID del chat de destino:

```bash
# Windows (PowerShell)
$env:TELEGRAM_TOKEN   = "tu_token_de_bot"
$env:TELEGRAM_CHAT_ID = "tu_chat_id"

# Linux / macOS
export TELEGRAM_TOKEN="tu_token_de_bot"
export TELEGRAM_CHAT_ID="tu_chat_id"
```

Si no se configuran, el sistema funciona con normalidad y simplemente omite el envío de notificaciones.

---

## ▶️ Uso

### 1. Iniciar la aplicación

```bash
python app.py
```

Abre el navegador en **http://localhost:5000**.

### 2. Módulos disponibles

| Ruta | Función |
|------|---------|
| `/` | Página de inicio y carga de imágenes para análisis |
| `/detectar_imagen` | Detección de fuego y humo sobre una imagen |
| `/video` | Carga y procesamiento de videos con detecciones anotadas |
| `/camara` | Detección en vivo desde la cámara web del navegador |
| `/dashboard` | Panel de control con estadísticas e historial de eventos |
| `/exportar_excel` | Descarga del reporte de eventos en formato `.xlsx` |

### 3. Entrenamiento y mejora del modelo (opcional)

```bash
# Generar/validar models/best.pt (usa dataset sintético si no existe un modelo válido)
python preparar_modelo.py

# Entrenar el modelo con el dataset definido en dataset/data.yaml
python entrenar.py

# Recalibrar el modelo con fotogramas reales extraídos de un video propio
python mejorar_modelo_con_video.py
```

> 💡 Para mejores resultados en producción, entrena con un dataset amplio y variado de imágenes reales de fuego y humo en entornos industriales.

---

## 🧠 ¿Cómo funciona?

1. **Entrada** → imagen, video o fotograma de cámara.
2. **Inferencia** → YOLOv8 detecta las clases `fuego` y `humo`; las heurísticas de OpenCV refuerzan casos difíciles.
3. **Severidad** → se evalúa según el porcentaje del área afectada y la confianza (BAJA / MEDIA / ALTA).
4. **Registro** → el evento se guarda en SQLite.
5. **Alerta** → alarma sonora + voz en el navegador y notificación a Telegram.
6. **Análisis** → consulta en el dashboard y exportación a Excel.

---

## 👤 Autor

**Angel Mauricio Lecarnaque**<br>
GitHub: [@lecarnaquemauricioangel-cmd](https://github.com/lecarnaquemauricioangel-cmd)<br>
Proyecto académico — SENATI · PIAD-426

---

<div align="center">

⭐ Si este proyecto te resultó útil, ¡no olvides darle una estrella!

</div>
