# -*- coding: utf-8 -*-
"""
WatchDog – Squid Game engine v2.

Features
────────
• Auto-assign "Player 1 / 2 / 3 …" to anyone not in the face database
• Face-recognised players keep their registered name (overrides Player N)
• Configurable game: total duration, movement duration, freeze duration
• Background timer thread cycles GREEN ↔ RED automatically
• Finish-line / win zone: any player whose bbox bottom crosses win_zone_y wins
• Target player: named person highlighted; special winner flag when they win
• Winners list with name, how they won, timestamp
"""

import threading
import time

import numpy as np

from config import FREEZE_MOVEMENT_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# Game configuration (set via configure() before start_game())
# ══════════════════════════════════════════════════════════════════════════════
class _Config:
    total_duration:    float = 0.0    # seconds, 0 = unlimited
    movement_duration: float = 10.0   # green-light (moving) phase length
    freeze_duration:   float = 5.0    # red-light  (freeze) phase length
    target_name:       str   = ""     # registered player name to track as VIP
    win_zone_y:        int   = 0      # finish-line Y pixel (0 = disabled)
    win_zone_x1:       int   = 0      # left bound  (0 = full width)
    win_zone_x2:       int   = 0      # right bound (0 = full width)

_cfg = _Config()


# ══════════════════════════════════════════════════════════════════════════════
# Global state
# ══════════════════════════════════════════════════════════════════════════════
freeze_mode:   bool  = False
game_active:   bool  = False
current_phase: str   = "idle"    # "idle" | "moving" | "frozen" | "ended"
phase_ends_at: float = 0.0       # epoch when current phase ends
game_ends_at:  float = 0.0       # epoch when whole game ends (0 = unlimited)

# players[track_id] = {
#     "name":          str   – "Player N" or face-recognised name
#     "player_number": int   – sequential number (1, 2, 3 …)
#     "status":        str   – "alive" | "eliminated" | "winner"
#     "last_position": (cx, cy)
#     "movement":      float
# }
players: dict = {}
winners: list = []               # accumulated winner records this game

_player_counter: int = 0         # increments to produce Player 1, 2, 3 …

_lock         = threading.Lock()
_timer_thread: threading.Thread | None = None
_stop_timer   = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════
def _center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _next_player() -> tuple[int, str]:
    """Return (number, display_name) for the next unregistered player."""
    global _player_counter
    _player_counter += 1
    return _player_counter, f"Player {_player_counter}"


def _end_game_inner():
    """Mark all surviving players as winners.  Must be called under _lock."""
    global game_active, freeze_mode, current_phase
    game_active   = False
    freeze_mode   = False
    current_phase = "ended"
    for tid, p in players.items():
        if p["status"] == "alive":
            p["status"] = "winner"
            winners.append({
                "track_id":      tid,
                "name":          p["name"],
                "player_number": p["player_number"],
                "win_type":      "survived",
                "is_target":     p["name"] == _cfg.target_name,
                "timestamp":     time.strftime("%H:%M:%S"),
            })


# ══════════════════════════════════════════════════════════════════════════════
# Background timer thread
# ══════════════════════════════════════════════════════════════════════════════
def _timer_loop():
    """Cycles: GREEN (movement_duration) → RED (freeze_duration) → repeat."""
    global freeze_mode, current_phase, phase_ends_at

    while not _stop_timer.is_set():

        # ── Green-light phase ──────────────────────────────────────────────
        with _lock:
            if not game_active:
                break
            freeze_mode   = False
            current_phase = "moving"
            phase_ends_at = time.time() + _cfg.movement_duration

        _stop_timer.wait(timeout=_cfg.movement_duration)
        if _stop_timer.is_set():
            break

        if _cfg.total_duration > 0 and time.time() >= game_ends_at:
            with _lock:
                _end_game_inner()
            break

        # ── Red-light (freeze) phase ───────────────────────────────────────
        with _lock:
            if not game_active:
                break
            freeze_mode   = True
            current_phase = "frozen"
            phase_ends_at = time.time() + _cfg.freeze_duration

        _stop_timer.wait(timeout=_cfg.freeze_duration)
        if _stop_timer.is_set():
            break

        if _cfg.total_duration > 0 and time.time() >= game_ends_at:
            with _lock:
                _end_game_inner()
            break


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def configure(
    total_duration:    float = 0.0,
    movement_duration: float = 10.0,
    freeze_duration:   float = 5.0,
    target_name:       str   = "",
    win_zone_y:        int   = 0,
    win_zone_x1:       int   = 0,
    win_zone_x2:       int   = 0,
):
    """
    Set all game parameters.
    Call this BEFORE start_game().  Safe to call while game is idle.
    """
    _cfg.total_duration    = max(0.0, float(total_duration))
    _cfg.movement_duration = max(1.0, float(movement_duration))
    _cfg.freeze_duration   = max(1.0, float(freeze_duration))
    _cfg.target_name       = str(target_name).strip()
    _cfg.win_zone_y        = int(win_zone_y)
    _cfg.win_zone_x1       = int(win_zone_x1)
    _cfg.win_zone_x2       = int(win_zone_x2)


def start_game():
    """
    Activate the game and launch the cycling timer thread.
    Resets winners; existing players are all set to alive.
    """
    global game_active, game_ends_at, freeze_mode, current_phase, winners
    global _timer_thread

    # Stop any previous timer cleanly
    _stop_timer.set()
    if _timer_thread and _timer_thread.is_alive():
        _timer_thread.join(timeout=3)
    _stop_timer.clear()

    with _lock:
        game_active   = True
        freeze_mode   = False
        current_phase = "moving"
        winners       = []
        game_ends_at  = (time.time() + _cfg.total_duration
                         if _cfg.total_duration > 0 else 0.0)
        # All tracked players restart as alive
        for p in players.values():
            p["status"] = "alive"

    _timer_thread = threading.Thread(
        target=_timer_loop, daemon=True, name="game-timer"
    )
    _timer_thread.start()


def set_freeze(enabled: bool):
    """Manually override the current phase (won't break the timer cycle)."""
    global freeze_mode, current_phase
    with _lock:
        freeze_mode   = enabled
        current_phase = "frozen" if enabled else "moving"


def reset():
    """
    Full reset: stop timer, clear winners, clear player counter,
    set all players alive, game becomes inactive.
    """
    global freeze_mode, game_active, current_phase, winners, _player_counter

    _stop_timer.set()
    if _timer_thread and _timer_thread.is_alive():
        _timer_thread.join(timeout=3)
    _stop_timer.clear()

    with _lock:
        freeze_mode     = False
        game_active     = False
        current_phase   = "idle"
        winners         = []
        _player_counter = 0
        for p in players.values():
            p["status"]        = "alive"
            p["movement"]      = 0.0
            p["player_number"] = 0      # will be re-assigned on next update


def assign_name(track_id: int, name: str):
    """
    Called by face_engine when a face is recognised.
    Updates the player's display name; keeps their player_number.
    """
    if not name or name == "Unknown":
        return
    with _lock:
        if track_id in players:
            players[track_id]["name"] = name


def get_config() -> dict:
    return {
        "total_duration":    _cfg.total_duration,
        "movement_duration": _cfg.movement_duration,
        "freeze_duration":   _cfg.freeze_duration,
        "target_name":       _cfg.target_name,
        "win_zone_y":        _cfg.win_zone_y,
        "win_zone_x1":       _cfg.win_zone_x1,
        "win_zone_x2":       _cfg.win_zone_x2,
    }


def get_status() -> dict:
    with _lock:
        now = time.time()
        remaining_phase = max(0.0, phase_ends_at - now) if phase_ends_at > 0 else 0.0
        remaining_game  = max(0.0, game_ends_at  - now) if game_ends_at  > 0 else 0.0
        return {
            "game_active":     game_active,
            "freeze_mode":     freeze_mode,
            "current_phase":   current_phase,
            "remaining_phase": round(remaining_phase, 1),
            "remaining_game":  round(remaining_game,  1),
            "total_duration":  _cfg.total_duration,
            "movement_duration": _cfg.movement_duration,
            "freeze_duration":   _cfg.freeze_duration,
            "target_name":     _cfg.target_name,
            "win_zone_y":      _cfg.win_zone_y,
            "players": {
                str(tid): {
                    "name":          p["name"],
                    "player_number": p["player_number"],
                    "status":        p["status"],
                    "movement":      round(p["movement"], 2),
                    "is_target":     p["name"] == _cfg.target_name,
                }
                for tid, p in players.items()
            },
            "winners": list(winners),
        }


def update(tracks: np.ndarray, frame_w: int = 640, frame_h: int = 480) -> list[dict]:
    """
    Process one frame of tracks through game logic.

    Parameters
    ----------
    tracks  : np.ndarray (N, 5)  [x1, y1, x2, y2, track_id]
    frame_w : frame width  (used for full-width finish line)
    frame_h : frame height (unused currently, reserved)

    Returns
    -------
    list of event dicts (Eliminated / Winner)
    """
    global players

    events:     list[dict] = []
    active_ids: set        = set()

    with _lock:
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            active_ids.add(tid)
            cx, cy      = _center(x1, y1, x2, y2)
            bbox_bottom = float(y2)

            # ── Register new player ────────────────────────────────────────
            if tid not in players:
                num, pname = _next_player()
                players[tid] = {
                    "name":          pname,
                    "player_number": num,
                    "status":        "alive",
                    "last_position": (cx, cy),
                    "movement":      0.0,
                }
                continue

            p  = players[tid]
            lx, ly = p["last_position"]
            dist = float(np.hypot(cx - lx, cy - ly))

            p["movement"]      = dist
            p["last_position"] = (cx, cy)

            # ── Freeze-mode elimination ────────────────────────────────────
            if freeze_mode and game_active and p["status"] == "alive":
                if dist > FREEZE_MOVEMENT_THRESHOLD:
                    p["status"] = "eliminated"
                    events.append({
                        "type":       "Eliminated",
                        "track_id":   tid,
                        "name":       p["name"],
                        "details_en": f"{p['name']} eliminated – moved during freeze!",
                        "details_ta": f"{p['name']} நீக்கப்பட்டார் – உறைவில் நகர்ந்தார்!",
                    })
                    continue   # skip win-zone check this frame

            # ── Finish-line / win-zone crossing ───────────────────────────
            if (
                game_active
                and p["status"] == "alive"
                and _cfg.win_zone_y > 0
                and not freeze_mode
                and bbox_bottom >= _cfg.win_zone_y
            ):
                # Optional X bounds check
                x_ok = True
                if _cfg.win_zone_x1 > 0 and _cfg.win_zone_x2 > 0:
                    x_ok = _cfg.win_zone_x1 <= cx <= _cfg.win_zone_x2

                if x_ok:
                    p["status"] = "winner"
                    entry = {
                        "track_id":      tid,
                        "name":          p["name"],
                        "player_number": p["player_number"],
                        "win_type":      "crossed_line",
                        "is_target":     p["name"] == _cfg.target_name,
                        "timestamp":     time.strftime("%H:%M:%S"),
                    }
                    winners.append(entry)
                    events.append({
                        "type":       "Winner",
                        "track_id":   tid,
                        "name":       p["name"],
                        "details_en": f"{p['name']} reached the finish line!",
                        "details_ta": f"{p['name']} இலக்கை அடைந்தார்!",
                    })

        # Prune disappeared tracks
        for gone in list(players.keys()):
            if gone not in active_ids:
                del players[gone]

    return events
