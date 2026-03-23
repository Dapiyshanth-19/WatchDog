"""
WatchDog – Behaviour analysis engine (v4 – normalized speed).

Key improvements over v3:
  • Speed normalized by bbox diagonal → works at ANY camera distance
    (close-up webcam, far CCTV, etc.)
  • Close-up detection: if bbox covers >50% of frame → person is at desk,
    suppress running/walking entirely
  • Longer smoothing window (12 frames) to absorb YOLO jitter
  • Alert cooldowns: same alert type per person throttled to once per 15s
  • Posture from aspect ratio: sitting, bending
"""

import time
import numpy as np
from collections import deque
from core.config import (
    LOITERING_TIME_THRESHOLD,
    CROWD_THRESHOLD,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)
from utils.translations import t


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_size(box):
    return (float(box[2] - box[0]), float(box[3] - box[1]))


# Activity states
ACTIVITY_STANDING  = "standing"
ACTIVITY_WALKING   = "walking"
ACTIVITY_RUNNING   = "running"
ACTIVITY_LOITERING = "loitering"
ACTIVITY_SITTING   = "sitting"
ACTIVITY_BENDING   = "bending"

# ── Tuning constants ────────────────────────────────────────────────
# Normalized speed = pixel_displacement / bbox_diagonal
# A person truly running covers ~15-25% of their body length per frame
# A person just sitting has YOLO jitter of ~0.5-2% of body length
NORM_RUNNING_THRESHOLD  = 0.12   # 12% of bbox diagonal per frame
NORM_WALKING_THRESHOLD  = 0.025  # 2.5% of bbox diagonal per frame
NORM_STILL_THRESHOLD    = 0.012  # below this = not moving at all

SMOOTH_WINDOW   = 12   # frames to average speed over
HYSTERESIS      = 5    # frames before switching activity label
ALERT_COOLDOWN  = 15.0 # seconds between same-type alerts per person

# If bbox area > this fraction of frame area, person is close-up (at desk)
CLOSEUP_AREA_RATIO = 0.35


class BehaviorAnalyzer:
    def __init__(self):
        self._state: dict[int, dict] = {}
        self._crowd_alerted = False
        self._alert_times: dict[str, float] = {}  # "type:tid" → last alert time
        self._frame_area = FRAME_WIDTH * FRAME_HEIGHT

    def get_activity(self, track_id: int) -> str:
        return self._state.get(track_id, {}).get("activity", ACTIVITY_STANDING)

    def get_activity_label(self, track_id: int, lang: str = "en") -> str:
        activity = self.get_activity(track_id)
        key = f"activity_{activity}"
        return t(key, lang)

    def _can_alert(self, alert_type: str, tid: int, now: float) -> bool:
        """Cooldown check: only allow one alert per type+person per ALERT_COOLDOWN."""
        key = f"{alert_type}:{tid}"
        last = self._alert_times.get(key, 0)
        if now - last < ALERT_COOLDOWN:
            return False
        self._alert_times[key] = now
        return True

    def update(self, tracks: np.ndarray, now: float | None = None) -> list[dict]:
        if now is None:
            now = time.time()

        events: list[dict] = []
        active_ids: set[int] = set()

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            active_ids.add(tid)
            cx, cy = _center((x1, y1, x2, y2))
            bw, bh = _box_size((x1, y1, x2, y2))
            bbox_diag = max(np.hypot(bw, bh), 1.0)
            bbox_area = bw * bh

            prev = self._state.get(tid)
            if prev is None:
                self._state[tid] = {
                    "last_center": (cx, cy),
                    "still_since": now,
                    "activity": ACTIVITY_STANDING,
                    "norm_speed_buf": deque(maxlen=SMOOTH_WINDOW),
                    "aspect_buf": deque(maxlen=SMOOTH_WINDOW),
                    "height_buf": deque(maxlen=SMOOTH_WINDOW),
                    "candidate": ACTIVITY_STANDING,
                    "candidate_frames": 0,
                    "last_bw": bw,
                    "last_bh": bh,
                }
                continue

            lx, ly = prev["last_center"]
            raw_dist = np.hypot(cx - lx, cy - ly)

            # ── Filter YOLO bbox size flicker ───────────────────────
            size_change = abs(bw - prev["last_bw"]) + abs(bh - prev["last_bh"])
            avg_dim = max((bw + bh + prev["last_bw"] + prev["last_bh"]) / 4, 1)
            if size_change / avg_dim > 0.30:
                norm_speed = 0.0  # bbox size jumped → detection noise
            else:
                # Normalize speed by bbox diagonal
                norm_speed = raw_dist / bbox_diag

            prev["norm_speed_buf"].append(norm_speed)
            prev["last_bw"] = bw
            prev["last_bh"] = bh
            prev["last_center"] = (cx, cy)

            # ── Smoothed normalized speed ───────────────────────────
            smooth_speed = float(np.mean(prev["norm_speed_buf"]))

            # ── Aspect ratio ────────────────────────────────────────
            aspect = bw / max(bh, 1)
            prev["aspect_buf"].append(aspect)
            smooth_aspect = float(np.mean(prev["aspect_buf"]))

            # ── Height tracking ─────────────────────────────────────
            prev["height_buf"].append(bh)
            avg_height = float(np.mean(prev["height_buf"]))

            # ── Close-up detection ──────────────────────────────────
            is_closeup = (bbox_area / self._frame_area) > CLOSEUP_AREA_RATIO

            # ── Classify activity ───────────────────────────────────
            candidate = self._classify(
                smooth_speed, smooth_aspect, avg_height, bh,
                is_closeup, now, prev
            )

            # ── Hysteresis ──────────────────────────────────────────
            if candidate == prev["candidate"]:
                prev["candidate_frames"] += 1
            else:
                prev["candidate"] = candidate
                prev["candidate_frames"] = 1

            if prev["candidate_frames"] >= HYSTERESIS:
                prev["activity"] = candidate

            activity = prev["activity"]

            # ── Still-time tracking ─────────────────────────────────
            if smooth_speed > NORM_STILL_THRESHOLD:
                prev["still_since"] = None
            elif prev["still_since"] is None:
                prev["still_since"] = now

            # ── Emit alert events (with cooldown) ───────────────────
            if activity == ACTIVITY_RUNNING and self._can_alert("Running", tid, now):
                events.append({
                    "type":       "Running",
                    "track_id":   tid,
                    "details_en": t("alert_running", "en", id=tid),
                    "details_ta": t("alert_running", "ta", id=tid),
                })

            elif activity == ACTIVITY_LOITERING:
                elapsed = now - (prev["still_since"] or now)
                if self._can_alert("Loitering", tid, now):
                    events.append({
                        "type":       "Loitering",
                        "track_id":   tid,
                        "details_en": t("alert_loitering", "en", id=tid, sec=int(elapsed)),
                        "details_ta": t("alert_loitering", "ta", id=tid, sec=int(elapsed)),
                    })

        # ── Crowd density ───────────────────────────────────────────
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

        # Prune old cooldown entries
        cutoff = now - 60
        self._alert_times = {k: v for k, v in self._alert_times.items() if v > cutoff}

        return events

    def _classify(self, speed, aspect, avg_height, cur_height,
                  is_closeup, now, prev) -> str:
        """
        Classify activity using normalized speed + posture.

        speed:      normalized (displacement / bbox_diagonal), smoothed
        aspect:     width / height of bbox
        is_closeup: bbox covers >35% of frame (person at desk / very close)
        """

        # ── Close-up: person is right in front of camera ────────────
        # They can NOT be running or walking — they're at their desk
        # or holding the camera. Only sitting/standing/bending apply.
        if is_closeup:
            if aspect > 0.75:
                return ACTIVITY_SITTING
            if avg_height > 0 and cur_height < avg_height * 0.75:
                return ACTIVITY_BENDING
            return ACTIVITY_STANDING

        # ── Sitting: wide bbox + barely moving ──────────────────────
        if aspect > 0.75 and speed < NORM_WALKING_THRESHOLD:
            return ACTIVITY_SITTING

        # ── Bending: height dropped significantly ───────────────────
        if avg_height > 0 and cur_height < avg_height * 0.7 and speed < NORM_WALKING_THRESHOLD:
            return ACTIVITY_BENDING

        # ── Running: fast normalized speed ──────────────────────────
        if speed >= NORM_RUNNING_THRESHOLD:
            return ACTIVITY_RUNNING

        # ── Walking: moderate speed ─────────────────────────────────
        if speed >= NORM_WALKING_THRESHOLD:
            return ACTIVITY_WALKING

        # ── Stationary: standing or loitering ───────────────────────
        still_since = prev.get("still_since")
        if still_since is not None:
            elapsed = now - still_since
            if elapsed >= LOITERING_TIME_THRESHOLD:
                return ACTIVITY_LOITERING

        return ACTIVITY_STANDING
