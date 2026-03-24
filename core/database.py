"""
WatchDog – SQLite persistence layer.
"""

import json
import os
import sqlite3
from datetime import datetime

from core.config import DB_PATH


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create all tables if they don't exist."""
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

            -- Face recognition: registered users
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                embedding  TEXT    NOT NULL,   -- JSON list of floats
                image_path TEXT    NOT NULL DEFAULT ''
            );

            -- Squid Game: persistent player states
            CREATE TABLE IF NOT EXISTS players (
                track_id  INTEGER PRIMARY KEY,
                name      TEXT    NOT NULL DEFAULT 'Unknown',
                status    TEXT    NOT NULL DEFAULT 'alive'  -- 'alive' | 'eliminated'
            );
        """)

        # Seed a default camera row if empty
        if not c.execute("SELECT 1 FROM cameras").fetchone():
            c.execute(
                "INSERT INTO cameras(name, source_url) VALUES (?, ?)",
                ("Camera 1", "0"),
            )


# ── Original alert / count helpers ────────────────────────────────────────────
def log_alert(camera_id: int, event_type: str, details: dict | None = None):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    result = []
    for r in rows:
        d   = dict(r)
        raw = d.get("details", "") or ""
        try:
            parsed        = json.loads(raw)
            d["details_en"] = parsed.get("details", "")
            d["details_ta"] = parsed.get("details_ta", "")
        except (json.JSONDecodeError, TypeError):
            d["details_en"] = raw
            d["details_ta"] = ""
        result.append(d)
    return result


def get_cameras() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM cameras").fetchall()
    return [dict(r) for r in rows]


# ── Face recognition helpers ───────────────────────────────────────────────────
def save_user(name: str, embedding: list, image_path: str = "") -> int:
    """Insert or update a registered face.  Returns the row id."""
    with _conn() as c:
        # If a user with this name already exists, append their embedding
        existing = c.execute(
            "SELECT id, embedding FROM users WHERE name = ?", (name,)
        ).fetchone()
        
        if existing:
            try:
                current_emb = json.loads(existing["embedding"])
                # Handle legacy 1D array
                if current_emb and isinstance(current_emb[0], float):
                    all_embs = [current_emb]
                else:
                    all_embs = current_emb
            except (json.JSONDecodeError, IndexError, TypeError):
                all_embs = []
            
            # Append new embedding
            all_embs.append(embedding)
            emb_json = json.dumps(all_embs)
            
            c.execute(
                "UPDATE users SET embedding=?, image_path=? WHERE id=?",
                (emb_json, image_path, existing["id"]),
            )
            return existing["id"]
        
        # New user: store as list of lists [[embedding]]
        emb_json = json.dumps([embedding])
        cur = c.execute(
            "INSERT INTO users(name, embedding, image_path) VALUES (?,?,?)",
            (name, emb_json, image_path),
        )
        return cur.lastrowid


def get_users() -> list[dict]:
    """Return all registered users (name + embedding JSON)."""
    with _conn() as c:
        rows = c.execute("SELECT id, name, embedding, image_path FROM users").fetchall()
    return [dict(r) for r in rows]


def delete_user(name: str) -> bool:
    """Delete a registered user by name.  Returns True if a row was deleted."""
    with _conn() as c:
        cur = c.execute("DELETE FROM users WHERE name = ?", (name,))
        return cur.rowcount > 0


# ── Squid Game player helpers ──────────────────────────────────────────────────
def upsert_player(track_id: int, name: str, status: str = "alive"):
    """Insert or update a player record."""
    with _conn() as c:
        c.execute(
            "INSERT INTO players(track_id, name, status) VALUES (?,?,?) "
            "ON CONFLICT(track_id) DO UPDATE SET name=excluded.name, status=excluded.status",
            (track_id, name, status),
        )


def get_db_players() -> list[dict]:
    """Return all player records stored in the DB."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM players").fetchall()
    return [dict(r) for r in rows]
