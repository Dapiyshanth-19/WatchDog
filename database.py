"""
WatchDog – SQLite persistence layer.
"""

import sqlite3
import json
from datetime import datetime
from config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create tables if they don't exist."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                source_url TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id        INTEGER  PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER  NOT NULL DEFAULT 1,
                timestamp TEXT     NOT NULL,
                type      TEXT     NOT NULL,
                details   TEXT
            );

            CREATE TABLE IF NOT EXISTS counts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL DEFAULT 1,
                timestamp TEXT    NOT NULL,
                count     INTEGER NOT NULL
            );
        """)
        # Seed a default camera row if empty
        if not c.execute("SELECT 1 FROM cameras").fetchone():
            c.execute("INSERT INTO cameras(name, source_url) VALUES (?, ?)",
                      ("Camera 1", "0"))


def log_alert(camera_id: int, event_type: str, details: dict | None = None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = json.dumps(details) if details else ""
    with _conn() as c:
        c.execute(
            "INSERT INTO alerts(camera_id, timestamp, type, details) VALUES (?,?,?,?)",
            (camera_id, ts, event_type, info),
        )


def log_count(camera_id: int, count: int):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        c.execute(
            "INSERT INTO counts(camera_id, timestamp, count) VALUES (?,?,?)",
            (camera_id, ts, count),
        )


def get_alerts(limit: int = 20) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_cameras() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM cameras").fetchall()
    return [dict(r) for r in rows]
