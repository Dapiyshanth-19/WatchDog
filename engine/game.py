# -*- coding: utf-8 -*-
"""
WatchDog – Squid Game engine v3.

TWO MODES
─────────
• Normal mode  (game_mode_enabled = False)
    – Face recognition still labels people by name
    – Unrecognised people are shown as ID{tid}  (no Player N)
    – No elimination, no finish line, no game timer
    – All behaviour/vision features work normally

• Game mode  (game_mode_enabled = True, game_active = True)
    – Unrecognised people auto-get "Player 1 / 2 / 3 …"
    – Green ↔ Red light cycles automatically (timer thread)
    – Moving during Red = ELIMINATED
    – Crossing finish line = WINNER
    – Winners list, target player highlight

RE-SPAWN PROTECTION (game mode only)
─────────────────────────────────────
SORT assigns new track_ids when a person briefly leaves frame.
We prevent eliminated players "coming back to life" two ways:

  1. Spatial re-id  – remember the last position of every eliminated /
     winner player for _RESPAWN_GRACE seconds.  Any new track that
     appears within _RESPAWN_DIST pixels of a remembered position
     immediately inherits the old status.

  2. Face re-id  – if face_engine later recognises a track as "Alice"
     and Alice was previously eliminated, assign_name() restores her
     eliminated status from _status_by_name.
"""

import threading
import time

import numpy as np

from core.config import FREEZE_MOVEMENT_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# Re-spawn protection constants
# ══════════════════════════════════════════════════════════════════════════════
_RESPAWN_GRACE = 6.0    # seconds to remember a recently-lost eliminated track
_RESPAWN_DIST  = 90     # pixel radius for spatial re-identification


# ══════════════════════════════════════════════════════════════════════════════
# Game configuration
# ══════════════════════════════════════════════════════════════════════════════
class _Config:
    total_duration:    float = 0.0
    movement_duration: float = 10.0
    freeze_duration:   float = 5.0
    target_name:       str   = ""
    win_zone_y:        int   = 0
    win_zone_x1:       int   = 0
    win_zone_x2:       int   = 0

_cfg = _Config()


# ══════════════════════════════════════════════════════════════════════════════
# Global state
# ══════════════════════════════════════════════════════════════════════════════
game_mode_enabled: bool  = False   # Normal vs Game mode switch
freeze_mode:       bool  = False
game_active:       bool  = False
current_phase:     str   = "idle"
phase_ends_at:     float = 0.0
game_ends_at:      float = 0.0

# Active players currently visible in the frame
# players[track_id] = {
#     "name":          str
#     "player_number": int   – 0 in normal mode
#     "status":        str   – "alive" | "eliminated" | "winner"
#     "last_position": (cx, cy)
#     "movement":      float
# }
players: dict = {}
winners: list = []

# Re-spawn protection state
# _recently_gone[old_tid] = {pos, time, name, player_number, status}
_recently_gone: dict = {}

# Face-based re-id: name → {player_number, status}
# Updated whenever a named player's track disappears while eliminated/winner
_status_by_name: dict = {}

_player_counter: int = 0

_lock         = threading.Lock()
_timer_thread: threading.Thread | None = None
_stop_timer   = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════
def _center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _next_player() -> tuple[int, str]:
    global _player_counter
    _player_counter += 1
    return _player_counter, f"Player {_player_counter}"


def _end_game_inner():
    """Mark surviving players as winners.  Must be called while holding _lock."""
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


def _expire_old_gone():
    """Remove re-spawn records older than _RESPAWN_GRACE.  Call under _lock.

    Eliminated and winner records are NEVER expired during a game — they must
    stay in _recently_gone for the entire game so that a player who was
    eliminated and leaves the frame cannot return as a fresh 'alive' Player N.
    Only 'alive' temporary entries (e.g. brief tracking gaps) are pruned.
    """
    now = time.time()
    for k in [k for k, v in _recently_gone.items()
              if v["status"] not in ("eliminated", "winner")
              and now - v["time"] > _RESPAWN_GRACE]:
        del _recently_gone[k]


def _try_restore_by_position(cx: float, cy: float) -> dict | None:
    """
    Look for a recently-gone eliminated/winner player near (cx, cy).
    Returns the matching record and removes it, or None.
    Must be called under _lock.
    """
    _expire_old_gone()
    best_dist = _RESPAWN_DIST
    best_key  = None
    for k, rec in _recently_gone.items():
        ox, oy = rec["pos"]
        d = float(np.hypot(cx - ox, cy - oy))
        if d < best_dist:
            best_dist = d
            best_key  = k
    if best_key is not None:
        rec = _recently_gone.pop(best_key)
        return rec
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Timer thread
# ══════════════════════════════════════════════════════════════════════════════
def _timer_loop():
    global freeze_mode, current_phase, phase_ends_at

    while not _stop_timer.is_set():

        # Green-light phase
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

        # Red-light (freeze) phase
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
def set_game_mode(enabled: bool):
    """
    Switch between Normal surveillance mode and Game mode.
    Switching to Normal stops the timer and disables game logic.
    """
    global game_mode_enabled, game_active, freeze_mode, current_phase
    if not enabled:
        # Stop any running game
        _stop_timer.set()
        if _timer_thread and _timer_thread.is_alive():
            _timer_thread.join(timeout=2)
        _stop_timer.clear()
        with _lock:
            game_mode_enabled = False
            game_active       = False
            freeze_mode       = False
            current_phase     = "idle"
    else:
        with _lock:
            game_mode_enabled = True


def configure(
    total_duration:    float = 0.0,
    movement_duration: float = 10.0,
    freeze_duration:   float = 5.0,
    target_name:       str   = "",
    win_zone_y:        int   = 0,
    win_zone_x1:       int   = 0,
    win_zone_x2:       int   = 0,
):
    _cfg.total_duration    = max(0.0, float(total_duration))
    _cfg.movement_duration = max(1.0, float(movement_duration))
    _cfg.freeze_duration   = max(1.0, float(freeze_duration))
    _cfg.target_name       = str(target_name).strip()
    _cfg.win_zone_y        = int(win_zone_y)
    _cfg.win_zone_x1       = int(win_zone_x1)
    _cfg.win_zone_x2       = int(win_zone_x2)


def start_game():
    """Launch the game (game_mode must already be enabled)."""
    global game_active, game_ends_at, freeze_mode, current_phase, winners
    global _timer_thread, _player_counter

    if not game_mode_enabled:
        set_game_mode(True)

    _stop_timer.set()
    if _timer_thread and _timer_thread.is_alive():
        _timer_thread.join(timeout=3)
    _stop_timer.clear()

    with _lock:
        game_active     = True
        freeze_mode     = False
        current_phase   = "moving"
        winners         = []
        _player_counter = 0
        _recently_gone.clear()
        _status_by_name.clear()
        game_ends_at = (time.time() + _cfg.total_duration
                        if _cfg.total_duration > 0 else 0.0)
        for p in players.values():
            p["status"] = "alive"

    _timer_thread = threading.Thread(
        target=_timer_loop, daemon=True, name="game-timer"
    )
    _timer_thread.start()


def set_freeze(enabled: bool):
    global freeze_mode, current_phase
    with _lock:
        freeze_mode   = enabled
        current_phase = "frozen" if enabled else "moving"


def reset():
    """Full reset — stops timer, clears all game state, keeps mode setting."""
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
        _recently_gone.clear()
        _status_by_name.clear()
        for p in players.values():
            p["status"]        = "alive"
            p["player_number"] = 0
            p["movement"]      = 0.0
            p["freeze_anchor"] = None


def assign_name(track_id: int, name: str):
    """
    Called by face_engine when a face is recognised.

    Works in BOTH modes.
    In game mode, also checks _status_by_name so that returning players
    (who got a new track_id) keep their eliminated / winner status.
    """
    if not name or name == "Unknown":
        return
    with _lock:
        if track_id not in players:
            return
        players[track_id]["name"] = name

        # Face-based re-id: restore previous game status for this person
        if game_mode_enabled and name in _status_by_name:
            saved = _status_by_name[name]
            # Only restore if the status is a "permanent" game outcome
            if saved["status"] in ("eliminated", "winner"):
                players[track_id]["player_number"] = saved["player_number"]
                players[track_id]["status"]         = saved["status"]


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
            "game_mode_enabled": game_mode_enabled,
            "game_active":       game_active,
            "freeze_mode":       freeze_mode,
            "current_phase":     current_phase,
            "remaining_phase":   round(remaining_phase, 1),
            "remaining_game":    round(remaining_game,  1),
            "total_duration":    _cfg.total_duration,
            "movement_duration": _cfg.movement_duration,
            "freeze_duration":   _cfg.freeze_duration,
            "target_name":       _cfg.target_name,
            "win_zone_y":        _cfg.win_zone_y,
            "players": {
                str(tid): {
                    "name":          p["name"] or f"ID{tid}",
                    "player_number": p["player_number"],
                    "status":        p["status"],
                    "movement":      round(p["movement"], 2),
                    "is_target":     bool(p["name"] == _cfg.target_name and _cfg.target_name),
                }
                for tid, p in players.items()
            },
            "winners": list(winners),
        }


def update(tracks: np.ndarray, frame_w: int = 640, frame_h: int = 480) -> list[dict]:
    """
    Process one frame.

    In Normal mode:  just tracks positions for face-name display; no game logic.
    In Game mode:    full elimination + win-zone + re-spawn protection.
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

            # ── Register or restore player ────────────────────────────────
            if tid not in players:
                restored = None

                # 1. Spatial re-id (game mode only — in normal mode there's
                #    nothing to restore)
                if game_mode_enabled:
                    restored = _try_restore_by_position(cx, cy)

                bw_init = float(x2 - x1)
                bh_init = float(y2 - y1)
                if restored:
                    players[tid] = {
                        "name":          restored["name"],
                        "player_number": restored["player_number"],
                        "status":        restored["status"],
                        "last_position": (cx, cy),
                        "movement":      0.0,
                        "freeze_anchor": None,
                        "last_bw": bw_init, "last_bh": bh_init,
                    }
                elif game_mode_enabled:
                    num, pname = _next_player()
                    players[tid] = {
                        "name":          pname,
                        "player_number": num,
                        "status":        "alive",
                        "last_position": (cx, cy),
                        "movement":      0.0,
                        "freeze_anchor": None,
                        "last_bw": bw_init, "last_bh": bh_init,
                    }
                else:
                    # Normal mode – blank name until face_engine fills it in
                    players[tid] = {
                        "name":          "",
                        "player_number": 0,
                        "status":        "alive",
                        "last_position": (cx, cy),
                        "movement":      0.0,
                        "freeze_anchor": None,
                        "last_bw": bw_init, "last_bh": bh_init,
                    }
                continue

            p  = players[tid]
            lx, ly = p["last_position"]
            dist = float(np.hypot(cx - lx, cy - ly))

            # ── Filter YOLO bbox size flicker (same as behavior.py) ───
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            prev_bw = p.get("last_bw", bw)
            prev_bh = p.get("last_bh", bh)
            size_change = abs(bw - prev_bw) + abs(bh - prev_bh)
            avg_dim = max((bw + bh + prev_bw + prev_bh) / 4, 1)
            if size_change / avg_dim > 0.25:
                dist = 0.0   # bbox size jumped → detection noise, not real movement
            p["last_bw"] = bw
            p["last_bh"] = bh

            p["movement"]      = dist
            p["last_position"] = (cx, cy)

            # ── Game logic (game mode only) ───────────────────────────────
            if not game_mode_enabled or not game_active:
                p["freeze_anchor"] = None
                continue

            if freeze_mode and p["status"] == "alive":
                # Set anchor the first frame freeze starts (FPS-independent measurement)
                if p["freeze_anchor"] is None:
                    p["freeze_anchor"] = (cx, cy)

                # Total displacement from freeze-start position
                ax, ay = p["freeze_anchor"]
                total_disp = float(np.hypot(cx - ax, cy - ay))

                # Normalize threshold by bbox diagonal so close-up cameras
                # don't trigger false eliminations from YOLO jitter
                bbox_diag = max(float(np.hypot(bw, bh)), 1.0)
                norm_disp = total_disp / bbox_diag
                # FREEZE_MOVEMENT_THRESHOLD (12px) ÷ typical bbox_diag (~200px) ≈ 0.06
                # For close-up (bbox_diag ~700px), 12px raw = 0.017 norm (no elimination)
                # Real movement at close-up: 50+px = 0.07+ norm (elimination)
                effective_threshold = max(FREEZE_MOVEMENT_THRESHOLD, bbox_diag * 0.06)

                if total_disp > effective_threshold:
                    p["status"] = "eliminated"
                    p["freeze_anchor"] = None
                    # Save for face re-id
                    if p["name"] and not p["name"].startswith("Player"):
                        _status_by_name[p["name"]] = {
                            "player_number": p["player_number"],
                            "status": "eliminated",
                        }
                    events.append({
                        "type":       "Eliminated",
                        "track_id":   tid,
                        "name":       p["name"],
                        "details_en": f"{p['name']} eliminated – moved during freeze!",
                        "details_ta": f"{p['name']} நீக்கப்பட்டார் – உறைவில் நகர்ந்தார்!",
                    })
                    continue
            else:
                # Not in freeze — clear anchor so it resets for next freeze phase
                p["freeze_anchor"] = None

            # Finish-line crossing
            if (
                p["status"] == "alive"
                and _cfg.win_zone_y > 0
                and not freeze_mode
                and bbox_bottom >= _cfg.win_zone_y
            ):
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
                    if p["name"] and not p["name"].startswith("Player"):
                        _status_by_name[p["name"]] = {
                            "player_number": p["player_number"],
                            "status": "winner",
                        }
                    events.append({
                        "type":       "Winner",
                        "track_id":   tid,
                        "name":       p["name"],
                        "details_en": f"{p['name']} reached the finish line!",
                        "details_ta": f"{p['name']} இலக்கை அடைந்தார்!",
                    })

        # ── Prune tracks that left the frame ──────────────────────────────
        for gone_tid in list(players.keys()):
            if gone_tid not in active_ids:
                p = players[gone_tid]
                # In game mode: save eliminated/winner tracks for re-spawn
                if game_mode_enabled and p["status"] in ("eliminated", "winner"):
                    _recently_gone[gone_tid] = {
                        "pos":           p["last_position"],
                        "time":          time.time(),
                        "name":          p["name"],
                        "player_number": p["player_number"],
                        "status":        p["status"],
                    }
                del players[gone_tid]

    return events
