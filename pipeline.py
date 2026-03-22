"""
WatchDog – Core video processing pipeline (v2).

Supports:
  • Multi-class detection (people / objects / vehicles / both / all)
  • Per-track activity labels (EN + Tamil)
  • Runtime config switching without restart
"""

import threading
import time
import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, MODEL_NAME, CONFIDENCE_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT,
    SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESH,
    CROWD_THRESHOLD,
)
from sort import Sort
from behavior import BehaviorAnalyzer
from translations import DETECTION_MODES, class_name, COCO_CLASS_MAP
import database as db

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "running":    False,
    "fps":        0.0,
    "count":      0,
    "obj_count":  0,
    "tracks":     [],   # [{x1,y1,x2,y2,id,class_id,activity_en,activity_ta}]
    "alerts":     [],
    "frame_jpg":  None,
    "error":      None,
}

# Runtime config (hot-swappable)
_cfg = {
    "detection_mode": "people",   # people | objects | vehicles | both | all
    "source":         CAMERA_SOURCE,
}

_lock        = threading.Lock()
_stop_event  = threading.Event()
_thread: threading.Thread | None = None


# ── Colours ───────────────────────────────────────────────────────────────────
_PALETTE = [
    (0,255,100),(255,140,0),(0,140,255),(255,0,140),
    (140,255,0),(0,255,200),(140,0,255),(255,255,0),
    (0,200,255),(255,80,80),(80,255,80),(80,80,255),
]
def _color(tid: int): return _PALETTE[int(tid) % len(_PALETTE)]

_ALERT_COLOR  = (0, 60, 255)
_OBJECT_COLOR = (255, 200, 0)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_tracks(frame, track_data: list, lang: str = "en"):
    for td in track_data:
        x1, y1, x2, y2 = int(td["x1"]), int(td["y1"]), int(td["x2"]), int(td["y2"])
        tid      = td["id"]
        cls_id   = td.get("class_id", 0)
        activity = td.get(f"activity_{lang}", "")
        is_alert = td.get("is_alert", False)

        is_person = cls_id == 0
        color = _ALERT_COLOR if is_alert else (_color(tid) if is_person else _OBJECT_COLOR)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label bar
        label = f"ID{tid} {class_name(cls_id, 'en')}"
        lsz   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lsz[1] - 6), (x1 + lsz[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 1, cv2.LINE_AA)

        # Activity text below the box
        if activity and is_person:
            cv2.putText(frame, activity, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_overlay(frame, person_count, obj_count, fps, alerts_this_frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cc = (0, 0, 255) if person_count >= CROWD_THRESHOLD else (0, 255, 100)
    cv2.putText(frame, f"People: {person_count}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cc, 2, cv2.LINE_AA)
    if obj_count:
        cv2.putText(frame, f"Objects: {obj_count}", (180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2, cv2.LINE_AA)

    y = 65
    for ev in alerts_this_frame[:3]:
        cv2.putText(frame, f"! {ev['details_en'][:55]}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 2, cv2.LINE_AA)
        y += 20


# ── Pipeline thread ───────────────────────────────────────────────────────────
def _run(initial_source):
    from ultralytics import YOLO

    with _lock:
        state["error"] = None
        state["running"] = True

    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        with _lock:
            state["error"] = f"Failed to load model '{MODEL_NAME}': {exc}"
            state["running"] = False
        return

    tracker  = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESH)
    analyzer = BehaviorAnalyzer()

    source = initial_source
    raw = str(source)
    cap_source = int(raw) if raw.isdigit() else raw
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        with _lock:
            state["error"] = f"Cannot open source: {source}"
            state["running"] = False
        return

    fps_avg        = 0.0
    t_prev         = time.time()
    t_count_log    = time.time()
    COUNT_LOG_INT  = 5.0

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if isinstance(cap_source, int):
                with _lock:
                    state["error"] = "Camera disconnected."
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Snapshot of live config (no lock needed for reads on simple dict)
        mode = _cfg.get("detection_mode", "people")
        classes = DETECTION_MODES.get(mode, [0])

        # ── Detection ──────────────────────────────────────────────────
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                        classes=classes)
        boxes = results[0].boxes

        person_dets = np.empty((0, 5))
        all_dets    = []   # (x1,y1,x2,y2,conf,class_id)

        if boxes is not None and len(boxes):
            xyxy    = boxes.xyxy.cpu().numpy()
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1,y1,x2,y2 = xyxy[i]
                cf  = confs[i]
                cid = cls_ids[i]
                all_dets.append((x1,y1,x2,y2,cf,cid))
                if cid == 0:   # person
                    person_dets = np.vstack([person_dets, [x1,y1,x2,y2,cf]])

        # ── Tracking (persons only) ────────────────────────────────────
        tracks = tracker.update(person_dets)

        # ── Behaviour ─────────────────────────────────────────────────
        now    = time.time()
        events = analyzer.update(tracks, now)
        alert_ids = {ev["track_id"] for ev in events}

        for ev in events:
            db.log_alert(1, ev["type"], {
                "details": ev["details_en"],
                "details_ta": ev["details_ta"],
                "track_id": ev["track_id"],
            })

        if now - t_count_log >= COUNT_LOG_INT:
            db.log_count(1, len(tracks))
            t_count_log = now

        # ── Build track_data list ──────────────────────────────────────
        track_data = []
        for trk in tracks:
            x1,y1,x2,y2,tid = trk
            tid = int(tid)
            track_data.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "id": tid,
                "class_id":    0,
                "activity_en": analyzer.get_activity_label(tid, "en"),
                "activity_ta": analyzer.get_activity_label(tid, "ta"),
                "is_alert":    tid in alert_ids,
            })

        # Non-person objects (drawn but not tracked)
        obj_data = []
        for (x1,y1,x2,y2,cf,cid) in all_dets:
            if cid != 0:
                obj_data.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "id": -1, "class_id": int(cid),
                    "activity_en": class_name(cid, "en"),
                    "activity_ta": class_name(cid, "ta"),
                    "is_alert": False,
                })

        # ── FPS ────────────────────────────────────────────────────────
        t_now   = time.time()
        fps_avg = 0.9*fps_avg + 0.1/max(t_now - t_prev, 1e-6)
        t_prev  = t_now

        # ── Annotate ───────────────────────────────────────────────────
        annotated = frame.copy()
        _draw_tracks(annotated, track_data + obj_data)
        _draw_overlay(annotated, len(tracks), len(obj_data), fps_avg, events)

        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])

        with _lock:
            state["fps"]       = round(fps_avg, 1)
            state["count"]     = len(tracks)
            state["obj_count"] = len(obj_data)
            state["tracks"]    = track_data
            state["alerts"]    = events
            state["frame_jpg"] = jpg.tobytes()

    cap.release()
    with _lock:
        state["running"]  = False
        state["frame_jpg"] = None


# ── Public API ────────────────────────────────────────────────────────────────
def start(source=None):
    global _thread
    if source is None:
        source = CAMERA_SOURCE
    if _thread and _thread.is_alive():
        return False, "Already running"
    _cfg["source"] = source
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
        state["running"]   = False
        state["frame_jpg"] = None
    return True, "Stopped"


def set_detection_mode(mode: str):
    if mode in DETECTION_MODES:
        _cfg["detection_mode"] = mode
        return True, f"Mode set to {mode}"
    return False, f"Unknown mode: {mode}"


def get_state() -> dict:
    with _lock:
        return dict(state)
