# -*- coding: utf-8 -*-
"""
WatchDog – PHANTOM VISION engine.

Stacks visual effects on top of each annotated frame.
All effects are independent and can be toggled at runtime.
"""

import cv2
import numpy as np
from collections import deque

# ── Colour palette (one per track ID) ────────────────────────────────────────
_PALETTE = [
    (0, 255, 150), (255, 100,   0), (  0, 150, 255), (255,   0, 150),
    (150, 255,   0), (  0, 220, 255), (200,   0, 255), (255, 220,   0),
    (  0, 255, 255), (255,  80,  80), ( 80, 255,  80), ( 80,  80, 255),
]


def _tcolor(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


class VisionEngine:
    """
    Usage (inside pipeline loop)::

        engine = VisionEngine()
        ...
        frame = engine.apply(frame, tracks)
    """

    # ── Runtime flags (set these from outside) ────────────────────────────────
    mode          = "normal"   # normal | neon | thermal | xray
    trails_on     = True
    network_on    = True
    predict_on    = True
    heatmap_on    = False

    # ── Internal state ────────────────────────────────────────────────────────
    def __init__(self, trail_len: int = 30, frame_w: int = 640, frame_h: int = 480):
        self._trail_len = trail_len
        self._trails: dict[int, deque] = {}
        self._heatmap  = np.zeros((frame_h, frame_w), dtype=np.float32)
        self._ripple_frames = 0   # >0 while crowd-alarm ripple is playing
        self._w = frame_w
        self._h = frame_h

    # ── Main entry point ──────────────────────────────────────────────────────
    def apply(self, frame: np.ndarray, tracks: np.ndarray,
              crowd_alarm: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        frame       : BGR frame (already has boxes drawn by pipeline)
        tracks      : np.ndarray (N,5) [x1,y1,x2,y2,track_id]
        crowd_alarm : True the frame a crowd-limit event fired
        """
        self._update_trails(tracks)
        self._update_heatmap(tracks)

        if crowd_alarm:
            self._ripple_frames = 18   # play ripple for 18 frames

        # ── Base mode filter ──────────────────────────────────────────────
        if self.mode == "thermal":
            frame = self._thermal(frame)
        elif self.mode == "xray":
            frame = self._xray(frame, tracks)
        elif self.mode == "neon":
            frame = self._neon_bg(frame)

        # ── Overlay effects ───────────────────────────────────────────────
        if self.heatmap_on and self.mode != "thermal":
            frame = self._draw_heatmap(frame)

        if self.network_on:
            self._draw_cluster_network(frame, tracks)

        if self.trails_on:
            self._draw_trails(frame)

        if self.predict_on:
            self._draw_predictions(frame, tracks)

        if self._ripple_frames > 0:
            self._draw_ripple(frame)
            self._ripple_frames -= 1

        return frame

    # ── Trail management ──────────────────────────────────────────────────────
    def _update_trails(self, tracks):
        active = set()
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            tid = int(tid)
            active.add(tid)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if tid not in self._trails:
                self._trails[tid] = deque(maxlen=self._trail_len)
            self._trails[tid].append((cx, cy))
        for gone in list(self._trails):
            if gone not in active:
                del self._trails[gone]

    def _update_heatmap(self, tracks):
        h, w = self._heatmap.shape
        for trk in tracks:
            x1, y1, x2, y2, _ = trk
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(self._heatmap, (cx, cy), 35, 0.6, -1)
        self._heatmap *= 0.97   # slow decay

    # ── Visual effects ────────────────────────────────────────────────────────
    def _thermal(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Boost contrast
        gray = cv2.equalizeHist(gray)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    def _xray(self, frame: np.ndarray, tracks) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.25).astype(np.uint8)
        # Highlight person regions with a subtle blue glow
        for trk in tracks:
            x1, y1, x2, y2, tid = [int(v) for v in trk]
            color = _tcolor(int(tid))
            cv2.rectangle(dark, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            # Inner glow
            region = dark[max(0,y1):y2, max(0,x1):x2]
            glow   = np.full_like(region, [c // 12 for c in color], dtype=np.uint8)
            dark[max(0,y1):y2, max(0,x1):x2] = cv2.addWeighted(region, 1.0, glow, 1.0, 0)
        return dark

    def _neon_bg(self, frame: np.ndarray) -> np.ndarray:
        """Darken the background so neon trails pop."""
        return (frame * 0.25).astype(np.uint8)

    def _draw_heatmap(self, frame: np.ndarray) -> np.ndarray:
        hm  = np.clip(self._heatmap, 0, 1)
        hm8 = (hm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
        mask  = (hm > 0.08).astype(np.float32)[:, :, np.newaxis]
        blend = cv2.addWeighted(frame, 0.55, colored, 0.45, 0)
        return (frame * (1 - mask) + blend * mask).astype(np.uint8)

    def _draw_trails(self, frame: np.ndarray):
        for tid, trail in self._trails.items():
            pts = list(trail)
            if len(pts) < 2:
                continue
            color = _tcolor(tid)
            n = len(pts)
            for i in range(1, n):
                alpha = i / n                          # fades from 0 → 1 (tip is brightest)
                thickness = max(1, int(4 * alpha))
                c = tuple(int(ch * alpha) for ch in color)
                cv2.line(frame, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)
            # Bright glowing dot at tip
            cv2.circle(frame, pts[-1], 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[-1], 8, tuple(c // 3 for c in color), 1, cv2.LINE_AA)

    def _draw_cluster_network(self, frame: np.ndarray, tracks):
        """Draw animated neural-network lines between nearby people."""
        if len(tracks) < 2:
            return
        centers = []
        ids     = []
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            ids.append(int(tid))

        LINK_DIST = 180   # px — max distance to draw a link
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                dist = np.hypot(cx2 - cx1, cy2 - cy1)
                if dist > LINK_DIST:
                    continue
                # Opacity fades with distance
                alpha = max(0.0, 1.0 - dist / LINK_DIST)
                c1 = _tcolor(ids[i])
                c2 = _tcolor(ids[j])
                # Gradient line by blending midpoint
                mid = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
                col_a = tuple(int(ch * alpha * 0.8) for ch in c1)
                col_b = tuple(int(ch * alpha * 0.8) for ch in c2)
                cv2.line(frame, (cx1, cy1), mid,       col_a, 1, cv2.LINE_AA)
                cv2.line(frame, mid,       (cx2, cy2), col_b, 1, cv2.LINE_AA)
                # Small pulse dot at midpoint
                pulse_r = max(2, int(4 * alpha))
                mid_col = tuple((a + b) // 2 for a, b in zip(col_a, col_b))
                cv2.circle(frame, mid, pulse_r, mid_col, -1, cv2.LINE_AA)

    def _draw_predictions(self, frame: np.ndarray, tracks):
        """Project each track forward using its current velocity vector."""
        STEPS   = 8
        HORIZON = 4     # px per step — tune to camera scale

        for trk in tracks:
            tid = int(trk[4])
            trail = self._trails.get(tid)
            if not trail or len(trail) < 2:
                continue
            pts = list(trail)
            vx  = pts[-1][0] - pts[-2][0]
            vy  = pts[-1][1] - pts[-2][1]
            speed = np.hypot(vx, vy)
            if speed < 2:     # not moving enough to predict
                continue

            prev = pts[-1]
            color = _tcolor(tid)
            for s in range(1, STEPS + 1):
                px = int(pts[-1][0] + vx * s * HORIZON)
                py = int(pts[-1][1] + vy * s * HORIZON)
                alpha = max(0.0, 1.0 - s / STEPS)
                r = max(1, int(4 * alpha))
                c = tuple(int(ch * alpha * 0.7) for ch in color)
                cv2.circle(frame, (px, py), r, c, -1, cv2.LINE_AA)
                if s > 1:
                    cv2.line(frame, prev, (px, py), c, 1, cv2.LINE_AA)
                prev = (px, py)

    def _draw_ripple(self, frame: np.ndarray):
        """Red shockwave ripple effect on crowd-limit alert."""
        h, w = frame.shape[:2]
        progress = 1.0 - self._ripple_frames / 18.0   # 0 → 1
        cx, cy = w // 2, h // 2
        max_r  = int(np.hypot(cx, cy))
        r      = int(max_r * progress)
        alpha  = max(0.0, 1.0 - progress)
        thickness = max(2, int(8 * (1 - progress)))
        color  = (0, 0, int(220 * alpha))
        if r > 0:
            cv2.circle(frame, (cx, cy), r, color, thickness, cv2.LINE_AA)
        # Red vignette at start
        if progress < 0.3:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(frame, 1 - 0.35 * alpha, overlay, 0.35 * alpha, 0, frame)
