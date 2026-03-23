# -*- coding: utf-8 -*-
"""
WatchDog – Crowd Anomaly Detection Engine.

Detects abnormal crowd behaviors in real time:
  • Stampede / panic  – sudden collective high-speed movement
  • Dispersal         – crowd rapidly scattering from a point
  • Convergence       – people suddenly gathering to one spot
  • Density spike     – abnormal density increase in a zone
  • Flow reversal     – people moving against the dominant direction

Uses motion vectors from SORT tracks (no extra model needed).
"""

import time
import math
import numpy as np
from collections import deque

from core.config import CROWD_THRESHOLD


# ── Thresholds ────────────────────────────────────────────────────────────────
STAMPEDE_SPEED_THRESH   = 55      # px/frame avg speed to flag stampede
DISPERSAL_THRESH        = 0.7     # ratio of outward-moving people
CONVERGENCE_THRESH      = 0.7     # ratio of inward-moving people
DENSITY_SPIKE_FACTOR    = 2.0     # sudden count jump multiplier
FLOW_REVERSAL_THRESH    = 0.6     # ratio of reversed-direction people
MIN_TRACKS_FOR_ANOMALY  = 4       # need at least N people for crowd anomaly
ANOMALY_COOLDOWN        = 8.0     # seconds between same-type alerts


class AnomalyDetector:
    """Real-time crowd anomaly detection from tracking data."""

    def __init__(self):
        self._prev_centers: dict[int, tuple] = {}
        self._count_history: deque = deque(maxlen=30)  # last 30 frames
        self._last_alert: dict[str, float] = {}
        self._velocity_history: deque = deque(maxlen=10)

    def update(self, tracks: np.ndarray, now: float | None = None) -> list[dict]:
        """
        Analyze tracks for anomalies.

        Parameters
        ----------
        tracks : np.ndarray (N,5)  [x1,y1,x2,y2,track_id]
        now    : timestamp

        Returns
        -------
        list of anomaly event dicts
        """
        if now is None:
            now = time.time()

        events = []
        n = len(tracks)
        self._count_history.append(n)

        if n < MIN_TRACKS_FOR_ANOMALY:
            self._prev_centers.clear()
            return events

        # Compute current centers and velocities
        centers = {}
        velocities = {}
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers[tid] = (cx, cy)
            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                velocities[tid] = (cx - px, cy - py)

        speeds = [math.hypot(vx, vy) for vx, vy in velocities.values()]
        avg_speed = np.mean(speeds) if speeds else 0

        # Store velocity snapshot
        self._velocity_history.append(velocities)

        # ── Check: Stampede / Panic ──────────────────────────────────────
        if avg_speed > STAMPEDE_SPEED_THRESH and len(speeds) >= MIN_TRACKS_FOR_ANOMALY:
            ev = self._emit("Stampede", now,
                            f"STAMPEDE ALERT! {len(speeds)} people moving at dangerous speed ({avg_speed:.0f} px/f)",
                            f"நெருக்கடி எச்சரிக்கை! {len(speeds)} நபர்கள் ஆபத்தான வேகத்தில் நகர்கிறார்கள்",
                            severity="critical", speed=avg_speed)
            if ev:
                events.append(ev)

        # ── Check: Dispersal (scattering from center) ────────────────────
        if len(velocities) >= MIN_TRACKS_FOR_ANOMALY:
            crowd_cx = np.mean([c[0] for c in centers.values()])
            crowd_cy = np.mean([c[1] for c in centers.values()])

            outward = 0
            inward = 0
            for tid, (vx, vy) in velocities.items():
                if tid not in centers:
                    continue
                cx, cy = centers[tid]
                # Direction from center to person
                dx, dy = cx - crowd_cx, cy - crowd_cy
                # Dot product: positive = moving away, negative = moving toward
                dot = dx * vx + dy * vy
                if dot > 5:
                    outward += 1
                elif dot < -5:
                    inward += 1

            total_moving = outward + inward
            if total_moving > 0:
                out_ratio = outward / total_moving
                in_ratio = inward / total_moving

                if out_ratio > DISPERSAL_THRESH and total_moving >= MIN_TRACKS_FOR_ANOMALY:
                    ev = self._emit("Dispersal", now,
                                    f"CROWD DISPERSAL! {outward}/{total_moving} people scattering rapidly",
                                    f"கூட்டம் சிதறுகிறது! {outward}/{total_moving} நபர்கள் விரைவாக சிதறுகிறார்கள்",
                                    severity="high")
                    if ev:
                        events.append(ev)

                if in_ratio > CONVERGENCE_THRESH and total_moving >= MIN_TRACKS_FOR_ANOMALY:
                    ev = self._emit("Convergence", now,
                                    f"CROWD CONVERGENCE! {inward}/{total_moving} people rushing to one point",
                                    f"கூட்ட குவிப்பு! {inward}/{total_moving} நபர்கள் ஒரு இடத்தில் குவிகிறார்கள்",
                                    severity="high")
                    if ev:
                        events.append(ev)

        # ── Check: Density Spike ─────────────────────────────────────────
        if len(self._count_history) >= 10:
            recent_avg = np.mean(list(self._count_history)[-5:])
            older_avg = np.mean(list(self._count_history)[-10:-5])
            if older_avg > 0 and recent_avg > older_avg * DENSITY_SPIKE_FACTOR:
                ev = self._emit("DensitySpike", now,
                                f"DENSITY SPIKE! Count surged from {older_avg:.0f} to {recent_avg:.0f}",
                                f"அடர்த்தி உயர்வு! எண்ணிக்கை {older_avg:.0f} இலிருந்து {recent_avg:.0f} ஆக உயர்ந்தது",
                                severity="medium")
                if ev:
                    events.append(ev)

        # ── Check: Flow Reversal ─────────────────────────────────────────
        if len(self._velocity_history) >= 5 and len(velocities) >= MIN_TRACKS_FOR_ANOMALY:
            # Compare current dominant direction vs recent
            curr_vecs = list(velocities.values())
            curr_avg_vx = np.mean([v[0] for v in curr_vecs])
            curr_avg_vy = np.mean([v[1] for v in curr_vecs])

            old_vecs = []
            for old_v in list(self._velocity_history)[-5:-1]:
                old_vecs.extend(old_v.values())
            if old_vecs:
                old_avg_vx = np.mean([v[0] for v in old_vecs])
                old_avg_vy = np.mean([v[1] for v in old_vecs])
                # Dot product of average directions
                dot = curr_avg_vx * old_avg_vx + curr_avg_vy * old_avg_vy
                old_mag = math.hypot(old_avg_vx, old_avg_vy)
                curr_mag = math.hypot(curr_avg_vx, curr_avg_vy)
                if old_mag > 3 and curr_mag > 3 and dot < -old_mag * curr_mag * 0.5:
                    ev = self._emit("FlowReversal", now,
                                    "FLOW REVERSAL! Crowd direction suddenly reversed",
                                    "போக்கு மாற்றம்! கூட்டத்தின் திசை திடீரென மாறியது",
                                    severity="high")
                    if ev:
                        events.append(ev)

        # Update previous centers
        self._prev_centers = centers
        return events

    def _emit(self, anomaly_type: str, now: float,
              details_en: str, details_ta: str,
              severity: str = "medium", **extra) -> dict | None:
        """Emit anomaly event with cooldown."""
        last = self._last_alert.get(anomaly_type, 0)
        if now - last < ANOMALY_COOLDOWN:
            return None
        self._last_alert[anomaly_type] = now
        return {
            "type": f"Anomaly:{anomaly_type}",
            "track_id": -1,
            "details_en": details_en,
            "details_ta": details_ta,
            "severity": severity,
            "anomaly_type": anomaly_type,
            **extra,
        }

    def get_crowd_stats(self, tracks: np.ndarray) -> dict:
        """Return real-time crowd statistics for dashboard."""
        n = len(tracks)
        if n == 0:
            return {"count": 0, "avg_speed": 0, "density": 0,
                    "dominant_direction": "none", "spread": 0}

        centers = []
        speeds = []
        directions = []
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                vx, vy = cx - px, cy - py
                speed = math.hypot(vx, vy)
                speeds.append(speed)
                if speed > 2:
                    angle = math.degrees(math.atan2(vy, vx))
                    directions.append(angle)

        avg_speed = float(np.mean(speeds)) if speeds else 0
        spread = float(np.std([c[0] for c in centers]) + np.std([c[1] for c in centers])) if len(centers) > 1 else 0

        # Dominant direction
        dom_dir = "stationary"
        if directions:
            avg_angle = float(np.mean(directions))
            if -45 <= avg_angle < 45:
                dom_dir = "right"
            elif 45 <= avg_angle < 135:
                dom_dir = "down"
            elif -135 <= avg_angle < -45:
                dom_dir = "up"
            else:
                dom_dir = "left"

        return {
            "count": n,
            "avg_speed": round(avg_speed, 1),
            "density": round(n / max(1, spread) * 100, 1) if spread > 0 else 0,
            "dominant_direction": dom_dir,
            "spread": round(spread, 1),
        }
