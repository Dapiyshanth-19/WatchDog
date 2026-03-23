"""
WatchDog – Headless runner (OpenCV window, no browser needed).

Usage:
    python run_headless.py                   # default webcam (source 0)
    python run_headless.py 1                 # second webcam
    python run_headless.py sample.mp4        # video file
    python run_headless.py rtsp://...        # IP camera

Press  Q  in the window to quit.
"""

import sys
import cv2
import numpy as np
import time

from ultralytics import YOLO
from core.sort import Sort
from engine.behavior import BehaviorAnalyzer
from core import database as db
from core.config import (
    CAMERA_SOURCE, MODEL_PATH, CONFIDENCE_THRESHOLD, PERSON_CLASS_ID,
    FRAME_WIDTH, FRAME_HEIGHT, SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESH,
    CROWD_THRESHOLD,
)

db.init_db()

# ── Source ───────────────────────────────────────────────────────────────────
raw = sys.argv[1] if len(sys.argv) > 1 else str(CAMERA_SOURCE)
source = int(raw) if raw.isdigit() else raw

# ── Init ──────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model   = YOLO(MODEL_PATH)
tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESH)
analyzer = BehaviorAnalyzer()

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    sys.exit(f"Cannot open source: {source}")

PALETTE = [(0,255,0),(255,128,0),(0,128,255),(255,0,128),(128,255,0),(0,255,128)]

fps_avg  = 0.0
t_prev   = time.time()
count_log_t = time.time()

print("Running – press Q in the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        if isinstance(source, int):
            print("Camera disconnected."); break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video file
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Detection
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                    classes=[PERSON_CLASS_ID])
    boxes = results[0].boxes
    dets = np.empty((0, 5))
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        dets = np.hstack([xyxy, conf])

    # Tracking
    tracks = tracker.update(dets)

    # Behaviour
    now    = time.time()
    events = analyzer.update(tracks, now)
    for ev in events:
        print(f"  [ALERT] {ev['type']}: {ev['details_en']}")
        db.log_alert(1, ev["type"], {"details": ev["details_en"], "track_id": ev["track_id"]})

    if now - count_log_t >= 5.0:
        db.log_count(1, len(tracks))
        count_log_t = now

    # FPS
    t_now   = time.time()
    fps_avg = 0.9 * fps_avg + 0.1 / max(t_now - t_prev, 1e-6)
    t_prev  = t_now

    # Draw
    for trk in tracks:
        x1, y1, x2, y2, tid = [int(v) for v in trk]
        color = PALETTE[tid % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    count_col = (0, 0, 255) if len(tracks) >= CROWD_THRESHOLD else (0, 255, 80)
    cv2.putText(frame, f"People: {len(tracks)}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_col, 2)
    cv2.putText(frame, f"FPS: {fps_avg:.1f}", (FRAME_WIDTH-120, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y = 55
    for ev in events[:3]:
        cv2.putText(frame, f"! {ev['type']}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
        y += 22

    cv2.imshow("WatchDog", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
