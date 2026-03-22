"""
WatchDog – Core video processing pipeline.

Runs in a background thread; results are stored in the shared `state` dict
so the Flask server can read them without blocking.
"""

import threading
import time
import cv2
import numpy as np

from config import (
    CAMERA_SOURCE,
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    PERSON_CLASS_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
    SORT_IOU_THRESH,
    CROWD_THRESHOLD,
)
from sort import Sort
from behavior import BehaviorAnalyzer
import database as db

# ── Shared state (written by pipeline thread, read by Flask) ─────────────────
state = {
    "running": False,
    "fps": 0.0,
    "count": 0,
    "tracks": [],        # list of [x1,y1,x2,y2,id]
    "alerts": [],        # recent alert dicts
    "frame_jpg": None,   # latest annotated JPEG bytes
    "error": None,
}
_lock = threading.Lock()

_stop_event = threading.Event()
_thread: threading.Thread | None = None

# ── Colours ──────────────────────────────────────────────────────────────────
_PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128),
    (128, 255, 0), (0, 255, 128), (128, 0, 255), (255, 255, 0),
]

def _color(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


# ── Annotation helpers ───────────────────────────────────────────────────────
def _draw_tracks(frame, tracks, alerts_this_frame):
    alert_ids = {e["track_id"] for e in alerts_this_frame}
    for trk in tracks:
        x1, y1, x2, y2, tid = [int(v) for v in trk]
        color = (0, 0, 255) if tid in alert_ids else _color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_overlay(frame, count, fps, alerts_this_frame):
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    count_color = (0, 0, 255) if count >= CROWD_THRESHOLD else (0, 255, 80)
    cv2.putText(frame, f"People: {count}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Show active alerts on-frame
    y = 60
    for ev in alerts_this_frame[:3]:
        cv2.putText(frame, f"! {ev['type']}: {ev['details'][:50]}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)
        y += 22


# ── Pipeline thread ──────────────────────────────────────────────────────────
def _run(source):
    from ultralytics import YOLO  # lazy import so Flask starts fast

    with _lock:
        state["error"] = None
        state["running"] = True

    # ── Initialise components ─────────────────────────────────────────────
    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        with _lock:
            state["error"] = f"Failed to load model '{MODEL_NAME}': {exc}"
            state["running"] = False
        return

    tracker = Sort(
        max_age=SORT_MAX_AGE,
        min_hits=SORT_MIN_HITS,
        iou_threshold=SORT_IOU_THRESH,
    )
    analyzer = BehaviorAnalyzer()

    # VideoCapture accepts int or str
    try:
        cap_source = int(source) if str(source).isdigit() else source
    except (ValueError, TypeError):
        cap_source = source

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        with _lock:
            state["error"] = f"Cannot open video source: {source}"
            state["running"] = False
        return

    fps_avg = 0.0
    t_prev = time.time()
    count_log_interval = 5.0   # log crowd count every N seconds
    t_last_count_log = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # For video files, loop; for cameras, report error
            if isinstance(cap_source, int):
                with _lock:
                    state["error"] = "Camera disconnected."
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video file
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ── Detection ──────────────────────────────────────────────────
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                        classes=[PERSON_CLASS_ID])
        boxes = results[0].boxes

        dets = np.empty((0, 5))
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, conf])

        # ── Tracking ───────────────────────────────────────────────────
        tracks = tracker.update(dets)

        # ── Behaviour analysis ─────────────────────────────────────────
        now = time.time()
        events = analyzer.update(tracks, now)
        for ev in events:
            db.log_alert(1, ev["type"], {"details": ev["details"],
                                         "track_id": ev["track_id"]})

        # Periodic count logging
        if now - t_last_count_log >= count_log_interval:
            db.log_count(1, len(tracks))
            t_last_count_log = now

        # ── FPS ────────────────────────────────────────────────────────
        t_now = time.time()
        inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
        fps_avg = 0.9 * fps_avg + 0.1 * inst_fps
        t_prev = t_now

        # ── Annotate frame ─────────────────────────────────────────────
        annotated = frame.copy()
        _draw_tracks(annotated, tracks, events)
        _draw_overlay(annotated, len(tracks), fps_avg, events)

        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])

        # ── Write shared state ─────────────────────────────────────────
        with _lock:
            state["fps"]       = round(fps_avg, 1)
            state["count"]     = len(tracks)
            state["tracks"]    = tracks.tolist() if len(tracks) else []
            state["alerts"]    = events
            state["frame_jpg"] = jpg.tobytes()

    cap.release()
    with _lock:
        state["running"] = False
        state["frame_jpg"] = None


# ── Public control API ────────────────────────────────────────────────────────
def start(source=None):
    global _thread
    if source is None:
        source = CAMERA_SOURCE

    if _thread and _thread.is_alive():
        return False, "Already running"

    _stop_event.clear()
    _thread = threading.Thread(target=_run, args=(source,), daemon=True)
    _thread.start()
    return True, "Started"


def stop():
    global _thread
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
    with _lock:
        state["running"] = False
        state["frame_jpg"] = None
    return True, "Stopped"


def get_state() -> dict:
    with _lock:
        return dict(state)
