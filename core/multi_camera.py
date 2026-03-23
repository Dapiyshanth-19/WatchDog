# -*- coding: utf-8 -*-
"""
WatchDog – Multi-Camera Manager.

Manages multiple video feeds with independent pipelines:
  • Each camera runs its own YOLO + SORT + analysis
  • Global tracker reconciles IDs across cameras
  • Unified alert stream from all cameras
  • Aggregate analytics across all feeds

Architecture:
  MultiCameraManager
    ├── Camera 1 → mini-pipeline (detect + track + analyze)
    ├── Camera 2 → mini-pipeline
    └── Camera N → mini-pipeline
    └── Global aggregator (merge alerts, cross-cam tracking)
"""

import threading
import time
import cv2
import numpy as np

from core.config import (
    CONFIDENCE_THRESHOLD, FRAME_WIDTH, FRAME_HEIGHT,
    SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESH,
)
from core.sort import Sort
from engine.behavior import BehaviorAnalyzer


class CameraFeed:
    """A single camera's processing state."""

    def __init__(self, cam_id: int, name: str, source):
        self.cam_id = cam_id
        self.name = name
        self.source = source
        self.running = False
        self.cap = None
        self.tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                            iou_threshold=SORT_IOU_THRESH)
        self.analyzer = BehaviorAnalyzer()
        self.frame_jpg: bytes | None = None
        self.count = 0
        self.fps = 0.0
        self.tracks = []
        self.alerts = []
        self.error = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, model):
        """Start processing this camera feed."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(model,), daemon=True,
            name=f"cam-{self.cam_id}"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.running = False

    def _run(self, model):
        raw = str(self.source).strip()
        src = int(raw) if raw.isdigit() else raw
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            self.error = f"Cannot open: {raw}"
            return

        self.running = True
        self.error = None
        fps_avg = 0.0
        t_prev = time.time()

        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(src, int):
                    self.error = "Camera disconnected"
                    break
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Detection
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                            classes=[0])
            boxes = results[0].boxes
            dets = np.empty((0, 5))
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, confs])

            # Tracking
            tracks = self.tracker.update(dets)

            # Behaviour
            now = time.time()
            events = self.analyzer.update(tracks, now)

            # FPS
            t_now = time.time()
            fps_avg = 0.9 * fps_avg + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # Draw
            annotated = frame.copy()
            for trk in tracks:
                x1, y1, x2, y2, tid = [int(v) for v in trk]
                color = (0, 255, 100)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"C{self.cam_id}:ID{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # HUD
            cv2.putText(annotated, f"{self.name} | {len(tracks)} people | {fps_avg:.1f} FPS",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])

            with self._lock:
                self.frame_jpg = jpg.tobytes()
                self.count = len(tracks)
                self.fps = round(fps_avg, 1)
                self.tracks = [{"x1": float(t[0]), "y1": float(t[1]),
                                "x2": float(t[2]), "y2": float(t[3]),
                                "id": int(t[4]), "cam_id": self.cam_id}
                               for t in tracks]
                self.alerts = events

        if self.cap:
            self.cap.release()
        self.running = False

    def get_state(self) -> dict:
        with self._lock:
            return {
                "cam_id": self.cam_id,
                "name": self.name,
                "source": str(self.source),
                "running": self.running,
                "count": self.count,
                "fps": self.fps,
                "tracks": self.tracks,
                "alerts": self.alerts,
                "error": self.error,
                "has_frame": self.frame_jpg is not None,
            }

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self.frame_jpg


class MultiCameraManager:
    """Manages multiple camera feeds with cross-camera analytics."""

    def __init__(self):
        self._cameras: dict[int, CameraFeed] = {}
        self._model = None
        self._next_id = 1

    def set_model(self, model):
        """Set the shared YOLO model for all cameras."""
        self._model = model

    def add_camera(self, name: str, source) -> int:
        """Add a camera feed. Returns camera ID."""
        cam_id = self._next_id
        self._next_id += 1
        self._cameras[cam_id] = CameraFeed(cam_id, name, source)
        return cam_id

    def remove_camera(self, cam_id: int) -> bool:
        if cam_id in self._cameras:
            self._cameras[cam_id].stop()
            del self._cameras[cam_id]
            return True
        return False

    def start_camera(self, cam_id: int) -> bool:
        if cam_id in self._cameras and self._model:
            self._cameras[cam_id].start(self._model)
            return True
        return False

    def stop_camera(self, cam_id: int) -> bool:
        if cam_id in self._cameras:
            self._cameras[cam_id].stop()
            return True
        return False

    def start_all(self):
        for cam_id in self._cameras:
            self.start_camera(cam_id)

    def stop_all(self):
        for cam in self._cameras.values():
            cam.stop()

    def get_camera_state(self, cam_id: int) -> dict | None:
        cam = self._cameras.get(cam_id)
        return cam.get_state() if cam else None

    def get_camera_frame(self, cam_id: int) -> bytes | None:
        cam = self._cameras.get(cam_id)
        return cam.get_frame() if cam else None

    def get_all_states(self) -> list[dict]:
        return [cam.get_state() for cam in self._cameras.values()]

    def get_aggregate_stats(self) -> dict:
        """Cross-camera aggregate statistics."""
        total_count = 0
        total_alerts = []
        active_cams = 0

        for cam in self._cameras.values():
            state = cam.get_state()
            if state["running"]:
                active_cams += 1
                total_count += state["count"]
                for alert in state["alerts"]:
                    alert["cam_id"] = cam.cam_id
                    alert["cam_name"] = cam.name
                    total_alerts.append(alert)

        return {
            "total_cameras": len(self._cameras),
            "active_cameras": active_cams,
            "total_people": total_count,
            "total_alerts": total_alerts,
            "cameras": self.get_all_states(),
        }


# Global instance
manager = MultiCameraManager()
