"""
Capa de acceso a datos (SQLite) para el sistema de detección temprana
de fuego y humo en almacenes.

Tabla `eventos`:
    id         INTEGER PRIMARY KEY AUTOINCREMENT
    fecha      TEXT       (formato ISO: YYYY-MM-DD HH:MM:SS)
    fuente     TEXT       (imagen | video | camara + nombre/identificador)
    clase      TEXT       (fuego | humo | ninguno)
    confianza  REAL       (0.0 - 1.0)
    detalle    TEXT       (severidad y observaciones adicionales)
"""

import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "historial.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crea la base de datos y la tabla `eventos` si no existen."""
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eventos (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha     TEXT NOT NULL,
                fuente    TEXT NOT NULL,
                clase     TEXT NOT NULL,
                confianza REAL NOT NULL,
                detalle   TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def registrar(fuente: str, clase: str, confianza: float, detalle: str = ""):
    """Inserta un nuevo evento de detección en la base de datos."""
    conn = get_conn()
    try:
        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO eventos (fecha, fuente, clase, confianza, detalle) "
            "VALUES (?, ?, ?, ?, ?)",
            (fecha, fuente, clase, float(confianza), detalle),
        )
        conn.commit()
    finally:
        conn.close()


def resumen_por_clase():
    """
    Devuelve un resumen agregado por clase, útil para las tarjetas de
    métricas y los gráficos del dashboard.

    Retorna una lista de dicts:
        [{"clase": "fuego", "total": 12, "confianza_promedio": 0.87}, ...]
    """
    conn = get_conn()
    try:
        filas = conn.execute(
            """
            SELECT clase,
                   COUNT(*) AS total,
                   AVG(confianza) AS confianza_promedio
            FROM eventos
            GROUP BY clase
            ORDER BY total DESC
            """
        ).fetchall()
        return [
            {
                "clase": f["clase"],
                "total": f["total"],
                "confianza_promedio": round(f["confianza_promedio"] or 0.0, 3),
            }
            for f in filas
        ]
    finally:
        conn.close()


def ultimos_eventos(n: int = 20):
    """Devuelve los últimos `n` eventos registrados, del más al menos reciente."""
    conn = get_conn()
    try:
        filas = conn.execute(
            "SELECT id, fecha, fuente, clase, confianza, detalle "
            "FROM eventos ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(f) for f in filas]
    finally:
        conn.close()


def obtener_todos():
    """
    Devuelve todos los eventos registrados (orden cronológico ascendente),
    utilizado por el módulo de reportes para exportar a Excel.
    """
    conn = get_conn()
    try:
        filas = conn.execute(
            "SELECT id, fecha, fuente, clase, confianza, detalle "
            "FROM eventos ORDER BY id ASC"
        ).fetchall()
        return [dict(f) for f in filas]
    finally:
        conn.close()


def total_eventos():
    """Cuenta total de eventos registrados en el historial."""
    conn = get_conn()
    try:
        fila = conn.execute("SELECT COUNT(*) AS total FROM eventos").fetchone()
        return fila["total"]
    finally:
        conn.close()
