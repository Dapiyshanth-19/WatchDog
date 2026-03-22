"""
WatchDog – Behaviour analysis engine (v2).

Tracks per-person activity and generates bilingual event alerts.
"""

import time
import numpy as np
from config import (
    RUNNING_SPEED_THRESHOLD,
    LOITERING_PIXEL_THRESHOLD,
    LOITERING_TIME_THRESHOLD,
    CROWD_THRESHOLD,
)
from translations import t


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


# Activity states
ACTIVITY_STANDING  = "standing"
ACTIVITY_WALKING   = "walking"
ACTIVITY_RUNNING   = "running"
ACTIVITY_LOITERING = "loitering"


class BehaviorAnalyzer:
    def __init__(self):
        # track_id → {"last_center", "still_since", "activity"}
        self._state: dict[int, dict] = {}
        self._crowd_alerted = False

    def get_activity(self, track_id: int) -> str:
        """Return current activity string for a track (English key)."""
        return self._state.get(track_id, {}).get("activity", ACTIVITY_STANDING)

    def get_activity_label(self, track_id: int, lang: str = "en") -> str:
        activity = self.get_activity(track_id)
        key = f"activity_{activity}"
        return t(key, lang)

    def update(self, tracks: np.ndarray, now: float | None = None) -> list[dict]:
        """
        Parameters
        ----------
        tracks : np.ndarray (N,5)  [x1,y1,x2,y2,track_id]
        now    : timestamp (seconds)

        Returns
        -------
        list of event dicts with keys: type, track_id, details_en, details_ta
        """
        if now is None:
            now = time.time()

        events: list[dict] = []
        active_ids: set[int] = set()

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            active_ids.add(tid)
            cx, cy = _center((x1, y1, x2, y2))

            prev = self._state.get(tid)
            if prev is None:
                self._state[tid] = {
                    "last_center": (cx, cy),
                    "still_since": now,
                    "activity": ACTIVITY_STANDING,
                }
                continue

            lx, ly = prev["last_center"]
            dist = np.hypot(cx - lx, cy - ly)

            # ── Determine activity ────────────────────────────────────────
            if dist > RUNNING_SPEED_THRESHOLD:
                activity = ACTIVITY_RUNNING
            elif dist > LOITERING_PIXEL_THRESHOLD:
                activity = ACTIVITY_WALKING
                prev["still_since"] = None
            else:
                # Barely moving
                if prev["still_since"] is None:
                    prev["still_since"] = now
                elapsed = now - prev["still_since"]
                activity = ACTIVITY_LOITERING if elapsed >= LOITERING_TIME_THRESHOLD else ACTIVITY_STANDING

            prev["activity"] = activity
            prev["last_center"] = (cx, cy)

            # ── Emit events ───────────────────────────────────────────────
            if activity == ACTIVITY_RUNNING:
                events.append({
                    "type":       "Running",
                    "track_id":   tid,
                    "details_en": t("alert_running", "en", id=tid),
                    "details_ta": t("alert_running", "ta", id=tid),
                })

            elif activity == ACTIVITY_LOITERING:
                elapsed = now - (prev["still_since"] or now)
                # Emit once per 10-second window to avoid spam
                if int(elapsed) % 10 == 0:
                    events.append({
                        "type":       "Loitering",
                        "track_id":   tid,
                        "details_en": t("alert_loitering", "en", id=tid, sec=int(elapsed)),
                        "details_ta": t("alert_loitering", "ta", id=tid, sec=int(elapsed)),
                    })

        # ── Crowd density ─────────────────────────────────────────────────
        n = len(tracks)
        if n >= CROWD_THRESHOLD and not self._crowd_alerted:
            events.append({
                "type":       "CrowdLimit",
                "track_id":   -1,
                "details_en": t("alert_crowd", "en", n=n, lim=CROWD_THRESHOLD),
                "details_ta": t("alert_crowd", "ta", n=n, lim=CROWD_THRESHOLD),
            })
            self._crowd_alerted = True
        elif n < CROWD_THRESHOLD:
            self._crowd_alerted = False

        # Prune gone tracks
        for gone in list(self._state.keys()):
            if gone not in active_ids:
                del self._state[gone]

        return events
