"""
Módulo de evaluación de severidad y notificación de alertas (Telegram)
para el sistema de detección temprana de fuego y humo en almacenes.
"""

import os
import requests

# ---------------------------------------------------------------------------
# Configuración de Telegram (puede sobreescribirse con variables de entorno)
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "TU_TOKEN_DE_BOT_AQUI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "TU_CHAT_ID_AQUI")
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Umbrales de severidad (porcentaje de área del frame ocupada por detecciones)
UMBRAL_MEDIA = 5.0   # % de área ocupada a partir del cual se considera MEDIA
UMBRAL_ALTA = 15.0   # % de área ocupada a partir del cual se considera ALTA


def _area_deteccion(bbox):
    """Calcula el área (en píxeles) de un bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    ancho = max(0.0, x2 - x1)
    alto = max(0.0, y2 - y1)
    return ancho * alto


def severidad(detecciones, area_frame):
    """
    Calcula el nivel de severidad de un conjunto de detecciones.

    Parámetros
    ----------
    detecciones : list[dict]
        Lista de detecciones, cada una con al menos las claves:
        {"clase": "fuego"|"humo", "confianza": float, "bbox": [x1, y1, x2, y2]}
    area_frame : float
        Área total del frame/imagen analizada (ancho * alto en píxeles).

    Retorna
    -------
    dict:
        {
            "nivel": "BAJA" | "MEDIA" | "ALTA",
            "porcentaje_area": float,
            "hay_fuego": bool,
            "hay_humo": bool,
        }
    """
    if not detecciones or area_frame <= 0:
        return {
            "nivel": "BAJA",
            "porcentaje_area": 0.0,
            "hay_fuego": False,
            "hay_humo": False,
        }

    area_total_detectada = sum(_area_deteccion(d["bbox"]) for d in detecciones)
    porcentaje_area = min(100.0, (area_total_detectada / area_frame) * 100.0)

    hay_fuego = any(d["clase"] == "fuego" for d in detecciones)
    hay_humo = any(d["clase"] == "humo" for d in detecciones)

    confianza_max_fuego = max(
        (d["confianza"] for d in detecciones if d["clase"] == "fuego"),
        default=0.0,
    )

    # La coexistencia de fuego y humo en la misma escena es siempre crítica
    # (incendio activo ya generando humo denso): severidad ALTA inmediata,
    # sin depender de área ni confianza.
    if hay_fuego and hay_humo:
        nivel = "ALTA"
    # La presencia de fuego (aunque sea sin humo) siempre es, como mínimo, MEDIA.
    elif hay_fuego and (porcentaje_area >= UMBRAL_ALTA or confianza_max_fuego >= 0.75):
        nivel = "ALTA"
    elif hay_fuego:
        nivel = "MEDIA"
    elif hay_humo and porcentaje_area >= UMBRAL_ALTA:
        nivel = "ALTA"
    elif hay_humo and porcentaje_area >= UMBRAL_MEDIA:
        nivel = "MEDIA"
    else:
        nivel = "BAJA"

    return {
        "nivel": nivel,
        "porcentaje_area": round(porcentaje_area, 2),
        "hay_fuego": hay_fuego,
        "hay_humo": hay_humo,
    }


def enviar_telegram(zona: str, nivel: str) -> bool:
    """
    Envía una alerta de emergencia a un chat de Telegram mediante la Bot API.

    Parámetros
    ----------
    zona : str
        Identificador de la zona/fuente donde se detectó el evento
        (por ejemplo: "Almacén A - Cámara 1", nombre de archivo, etc).
    nivel : str
        Nivel de severidad ("BAJA", "MEDIA", "ALTA").

    Retorna
    -------
    bool
        True si el mensaje se envió correctamente, False en caso contrario.
    """
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "TU_TOKEN_DE_BOT_AQUI":
        print("[alertas] TELEGRAM_TOKEN no configurado. Alerta no enviada.")
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "TU_CHAT_ID_AQUI":
        print("[alertas] TELEGRAM_CHAT_ID no configurado. Alerta no enviada.")
        return False

    mensaje = (
        "🔥 *ALERTA DE EMERGENCIA - FUEGO DETECTADO* 🔥\n\n"
        f"*Zona:* {zona}\n"
        f"*Severidad:* {nivel}\n"
        "Se ha detectado presencia de fuego en el almacén monitoreado. "
        "Verifique la zona de inmediato y active los protocolos de seguridad."
    )

    url = TELEGRAM_API_URL.format(token=TELEGRAM_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown",
    }

    try:
        respuesta = requests.post(url, data=payload, timeout=8)
        if respuesta.status_code == 200:
            return True
        print(f"[alertas] Error Telegram ({respuesta.status_code}): {respuesta.text}")
        return False
    except requests.RequestException as exc:
        print(f"[alertas] Excepción al enviar Telegram: {exc}")
        return False
