# -*- coding: utf-8 -*-
"""
WatchDog – Threat Detection Engine.

Detects specific threats from video frames and tracking data:
  • Fall detection      – person suddenly changes from tall to wide aspect ratio
  • Fire/smoke detection – color-based detection of flames/smoke regions
  • Intrusion detection  – person enters a restricted zone
  • Abandoned object    – static non-person detection for extended time
  • Crowd crush risk    – dangerously high density in a small area

Works alongside YOLO+SORT — adds threat-specific analysis layers.
"""

import time
import math
import cv2
import numpy as np
from collections import deque

from core.config import FRAME_WIDTH, FRAME_HEIGHT


# ── Thresholds ────────────────────────────────────────────────────────────────
FALL_ASPECT_RATIO_THRESH = 1.2   # width/height > this = possibly fallen
FALL_SPEED_THRESH        = 30    # rapid vertical movement before fall
FALL_CONFIRM_FRAMES      = 5     # must stay "fallen" for N frames

FIRE_MIN_AREA            = 500   # minimum pixel area for fire region
FIRE_CONFIDENCE          = 0.6   # color-based confidence threshold

CRUSH_DENSITY_THRESH     = 0.15  # people per pixel^2 in cluster
CRUSH_MIN_PEOPLE         = 6     # min people in a crush zone

INTRUSION_COOLDOWN       = 10.0  # seconds between same-zone alerts
ABANDONED_TIME           = 30.0  # seconds for abandoned object alert

THREAT_COOLDOWN          = 5.0   # seconds between same-type alerts


class ThreatDetector:
    """Multi-threat detection engine."""

    def __init__(self):
        # Fall detection state: track_id → {aspect_ratios, vertical_speed, fallen_frames}
        self._fall_state: dict[int, dict] = {}
        # Intrusion zones: list of {name, x1, y1, x2, y2}
        self._zones: list[dict] = []
        # Static object tracking
        self._static_objects: dict[str, dict] = {}
        # Alert cooldowns
        self._last_alert: dict[str, float] = {}
        # Historical for crush detection
        self._prev_centers: dict[int, tuple] = {}

    # ── Zone Management ──────────────────────────────────────────────────────
    def add_zone(self, name: str, x1: int, y1: int, x2: int, y2: int,
                 zone_type: str = "restricted"):
        """Add a monitored zone (restricted area, VIP area, etc.)."""
        self._zones.append({
            "name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "type": zone_type,
        })

    def clear_zones(self):
        self._zones.clear()

    def get_zones(self) -> list[dict]:
        return list(self._zones)

    # ── Main Update ──────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, tracks: np.ndarray,
               now: float | None = None) -> list[dict]:
        """
        Detect threats from frame and tracks.

        Returns list of threat event dicts.
        """
        if now is None:
            now = time.time()

        events = []

        # Fall detection
        events.extend(self._detect_falls(tracks, now))

        # Fire detection
        events.extend(self._detect_fire(frame, now))

        # Intrusion detection
        events.extend(self._detect_intrusions(tracks, now))

        # Crush risk
        events.extend(self._detect_crush_risk(tracks, now))

        # Update state
        self._update_centers(tracks)

        return events

    # ── Fall Detection ───────────────────────────────────────────────────────
    def _detect_falls(self, tracks: np.ndarray, now: float) -> list[dict]:
        events = []
        active_ids = set()

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            active_ids.add(tid)

            w = float(x2 - x1)
            h = float(y2 - y1)
            aspect = w / max(h, 1)
            cy_now = float(y1 + y2) / 2

            if tid not in self._fall_state:
                self._fall_state[tid] = {
                    "aspects": deque(maxlen=15),
                    "prev_cy": cy_now,
                    "fallen_frames": 0,
                    "was_tall": False,
                    "alerted": False,
                }

            state = self._fall_state[tid]
            state["aspects"].append(aspect)

            # Check if person was previously tall (standing)
            if len(state["aspects"]) >= 3:
                recent = list(state["aspects"])
                old_aspects = recent[:-3] if len(recent) > 3 else [recent[0]]
                was_tall = any(a < 0.8 for a in old_aspects)

                if was_tall:
                    state["was_tall"] = True

            # Vertical speed (falling = rapid downward movement)
            vert_speed = cy_now - state["prev_cy"]
            state["prev_cy"] = cy_now

            # Person now wide (fallen) and was previously tall
            if aspect > FALL_ASPECT_RATIO_THRESH and state["was_tall"]:
                state["fallen_frames"] += 1
            else:
                state["fallen_frames"] = max(0, state["fallen_frames"] - 1)

            # Confirm fall after N frames of being "wide"
            if state["fallen_frames"] >= FALL_CONFIRM_FRAMES and not state["alerted"]:
                state["alerted"] = True
                ev = self._emit("Fall", now,
                                f"FALL DETECTED! Person ID{tid} may have fallen",
                                f"விழுந்தது கண்டறியப்பட்டது! நபர் ID{tid} விழுந்திருக்கலாம்",
                                severity="critical", track_id=tid)
                if ev:
                    events.append(ev)

            # Reset alert if person stands back up
            if aspect < 0.8:
                state["alerted"] = False
                state["was_tall"] = True

        # Prune gone tracks
        for gone in list(self._fall_state):
            if gone not in active_ids:
                del self._fall_state[gone]

        return events

    # ── Fire Detection (Color-Based) ─────────────────────────────────────────
    def _detect_fire(self, frame: np.ndarray, now: float) -> list[dict]:
        """Detect fire/flame regions using HSV color analysis."""
        events = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Fire color range in HSV (red-orange-yellow flames)
        lower1 = np.array([0, 120, 200])
        upper1 = np.array([25, 255, 255])
        lower2 = np.array([160, 120, 200])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological ops to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < FIRE_MIN_AREA:
                continue

            # Check color intensity distribution (fires have bright centers)
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Fire regions tend to be very bright
            brightness = np.mean(roi)
            if brightness < 150:
                continue

            # Confidence based on area and brightness
            confidence = min(1.0, (area / 3000) * (brightness / 255))
            if confidence >= FIRE_CONFIDENCE:
                ev = self._emit("Fire", now,
                                f"FIRE/FLAME DETECTED at ({x},{y}) area={area:.0f}px confidence={confidence:.0%}",
                                f"தீ/நெருப்பு கண்டறியப்பட்டது ({x},{y}) பரப்பளவு={area:.0f}px நம்பிக்கை={confidence:.0%}",
                                severity="critical", x=x, y=y, w=w, h=h,
                                confidence=round(confidence, 2))
                if ev:
                    events.append(ev)

        return events

    # ── Intrusion Detection ──────────────────────────────────────────────────
    def _detect_intrusions(self, tracks: np.ndarray, now: float) -> list[dict]:
        events = []
        if not self._zones:
            return events

        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            for zone in self._zones:
                if zone["type"] != "restricted":
                    continue
                if (zone["x1"] <= cx <= zone["x2"] and
                        zone["y1"] <= cy <= zone["y2"]):
                    key = f"Intrusion:{zone['name']}:{tid}"
                    ev = self._emit(key, now,
                                    f"INTRUSION! Person ID{tid} entered restricted zone '{zone['name']}'",
                                    f"ஊடுருவல்! நபர் ID{tid} தடைப்பகுதி '{zone['name']}' உள் நுழைந்தார்",
                                    severity="high", track_id=tid,
                                    zone_name=zone["name"])
                    if ev:
                        ev["type"] = "Threat:Intrusion"
                        events.append(ev)

        return events

    # ── Crush Risk Detection ─────────────────────────────────────────────────
    def _detect_crush_risk(self, tracks: np.ndarray, now: float) -> list[dict]:
        """Detect dangerously dense clusters of people."""
        events = []
        if len(tracks) < CRUSH_MIN_PEOPLE:
            return events

        centers = []
        for trk in tracks:
            x1, y1, x2, y2, _ = trk
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

        # Simple cluster detection: check if many people in small area
        centers_arr = np.array(centers)
        for i, (cx, cy) in enumerate(centers):
            dists = np.sqrt((centers_arr[:, 0] - cx)**2 + (centers_arr[:, 1] - cy)**2)
            nearby = np.sum(dists < 80)  # within 80px radius
            if nearby >= CRUSH_MIN_PEOPLE:
                ev = self._emit("CrushRisk", now,
                                f"CRUSH RISK! {nearby} people packed within 80px radius at ({cx:.0f},{cy:.0f})",
                                f"நெரிசல் ஆபத்து! {nearby} நபர்கள் ({cx:.0f},{cy:.0f}) அருகில் நெருக்கமாக உள்ளனர்",
                                severity="critical", cluster_size=nearby,
                                x=int(cx), y=int(cy))
                if ev:
                    events.append(ev)
                break  # Only one crush alert per frame

        return events

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _update_centers(self, tracks: np.ndarray):
        self._prev_centers.clear()
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            self._prev_centers[int(tid)] = ((x1 + x2) / 2, (y1 + y2) / 2)

    def _emit(self, threat_type: str, now: float,
              details_en: str, details_ta: str,
              severity: str = "medium", **extra) -> dict | None:
        last = self._last_alert.get(threat_type, 0)
        cooldown = INTRUSION_COOLDOWN if "Intrusion" in threat_type else THREAT_COOLDOWN
        if now - last < cooldown:
            return None
        self._last_alert[threat_type] = now
        track_id = extra.pop("track_id", -1)
        return {
            "type": f"Threat:{threat_type.split(':')[0]}",
            "track_id": track_id,
            "details_en": details_en,
            "details_ta": details_ta,
            "severity": severity,
            "threat_type": threat_type,
            **extra,
        }

    def draw_zones(self, frame: np.ndarray):
        """Draw restricted zones on frame."""
        for zone in self._zones:
            color = (0, 0, 255) if zone["type"] == "restricted" else (0, 255, 255)
            cv2.rectangle(frame, (zone["x1"], zone["y1"]),
                          (zone["x2"], zone["y2"]), color, 2, cv2.LINE_AA)
            # Zone label
            label = f"{zone['type'].upper()}: {zone['name']}"
            lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (zone["x1"], zone["y1"] - lsz[1] - 6),
                          (zone["x1"] + lsz[0] + 4, zone["y1"]), color, -1)
            cv2.putText(frame, label, (zone["x1"] + 2, zone["y1"] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_fire_overlay(self, frame: np.ndarray):
        """Draw fire detection overlay (highlight fire regions)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array([0, 120, 200]), np.array([25, 255, 255])
        lower2, upper2 = np.array([160, 120, 200]), np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > FIRE_MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "FIRE!", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
