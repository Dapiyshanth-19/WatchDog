"""
WatchDog – Behaviour analysis engine.

Each frame, call `analyzer.update(tracks, frame_time)` to receive a list of
triggered events.  Events are plain dicts ready to be logged or sent to the UI.
"""

import time
import numpy as np
from config import (
    RUNNING_SPEED_THRESHOLD,
    LOITERING_PIXEL_THRESHOLD,
    LOITERING_TIME_THRESHOLD,
    CROWD_THRESHOLD,
)


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class BehaviorAnalyzer:
    def __init__(self):
        # {track_id: {"last_center": (x,y), "still_since": float | None}}
        self._state: dict[int, dict] = {}
        self._crowd_alerted = False   # simple cooldown flag

    # ------------------------------------------------------------------ #
    def update(self, tracks: np.ndarray, now: float | None = None) -> list[dict]:
        """
        Parameters
        ----------
        tracks : np.ndarray shape (N, 5) – [x1, y1, x2, y2, track_id]
        now    : float timestamp (seconds).  Defaults to time.time().

        Returns
        -------
        list of event dicts  {"type": str, "track_id": int, "details": str}
        """
        if now is None:
            now = time.time()

        events: list[dict] = []
        active_ids = set()

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            active_ids.add(tid)
            cx, cy = _center((x1, y1, x2, y2))

            prev = self._state.get(tid)

            if prev is None:
                self._state[tid] = {"last_center": (cx, cy), "still_since": now}
                continue

            lx, ly = prev["last_center"]
            dist = np.hypot(cx - lx, cy - ly)

            # ── Running ──────────────────────────────────────────────────
            if dist > RUNNING_SPEED_THRESHOLD:
                events.append({
                    "type": "Running",
                    "track_id": tid,
                    "details": f"Person {tid} running (speed ≈ {dist:.0f} px/frame)",
                })

            # ── Loitering ─────────────────────────────────────────────────
            if dist < LOITERING_PIXEL_THRESHOLD:
                if prev["still_since"] is None:
                    prev["still_since"] = now
                elapsed = now - prev["still_since"]
                if elapsed >= LOITERING_TIME_THRESHOLD:
                    events.append({
                        "type": "Loitering",
                        "track_id": tid,
                        "details": f"Person {tid} loitering ({elapsed:.0f}s stationary)",
                    })
            else:
                prev["still_since"] = None

            prev["last_center"] = (cx, cy)

        # ── Crowd density alert ───────────────────────────────────────────
        n = len(tracks)
        if n >= CROWD_THRESHOLD and not self._crowd_alerted:
            events.append({
                "type": "CrowdLimit",
                "track_id": -1,
                "details": f"Crowd limit reached: {n} people detected (threshold={CROWD_THRESHOLD})",
            })
            self._crowd_alerted = True
        elif n < CROWD_THRESHOLD:
            self._crowd_alerted = False

        # Prune state for tracks that disappeared
        for gone in list(self._state.keys()):
            if gone not in active_ids:
                del self._state[gone]

        return events
