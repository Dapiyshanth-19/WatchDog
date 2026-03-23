"""
WatchDog – Central configuration.
Edit these values to match your environment before running.
"""

import os

# Base directory = project root (one level up from core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Video source ─────────────────────────────────────────────────────────────
# 0          = default USB/built-in webcam
# 1, 2, …   = other USB cameras
# "path/to/video.mp4"                        = local video file (great for demo)
# "http://192.168.x.x:8080/video"            = IP Webcam (Android app, WiFi)
# "rtsp://user:pass@192.168.x.x:554/stream1" = RTSP IP camera
CAMERA_SOURCE = 0

# ── YOLO model ───────────────────────────────────────────────────────────────
# "yolo11n.pt"  – fast nano model (recommended for CPU / demo)
# "yolo11s.pt"  – small, slightly more accurate
# "yolo26n.pt"  – use when YOLOv26 weights become available
MODEL_NAME = "yolo26n.pt"

# MODEL_PATH can be changed at runtime via POST /model {"path": "yolo11n.pt"}
# Defaults to MODEL_NAME; a pipeline restart is required after switching.
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

# Only report detections above this confidence
CONFIDENCE_THRESHOLD = 0.45

# COCO class id for "person" is 0 — keep this unless you change models
PERSON_CLASS_ID = 0

# ── Frame processing ──────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Behaviour thresholds ──────────────────────────────────────────────────────
# Crowd alert: warn when this many people are in the frame at once
CROWD_THRESHOLD = 8

# Running: smoothed speed (avg over N frames) must exceed this
RUNNING_SPEED_THRESHOLD = 25   # pixels/frame (smoothed avg)

# Walking: smoothed speed must exceed this (below = stationary)
WALKING_SPEED_THRESHOLD = 5    # pixels/frame (smoothed avg)

# Loitering: flag if a track barely moves for longer than this duration
LOITERING_PIXEL_THRESHOLD = 4           # px; "barely moving" per frame
LOITERING_TIME_THRESHOLD  = 15.0        # seconds

# Speed smoothing: number of frames to average speed over (removes jitter)
SPEED_SMOOTH_WINDOW = 8

# ── Squid Game – Freeze Mode ──────────────────────────────────────────────────
# A player is ELIMINATED if they move more than this many pixels from
# the position they were at when the freeze phase started.
# This is total displacement (not per-frame), so it is FPS-independent.
# Raise if you get false eliminations; lower to catch subtler movement.
FREEZE_MOVEMENT_THRESHOLD = 12          # pixels (total displacement from freeze anchor)

# ── Face recognition ──────────────────────────────────────────────────────────
# SETUP: install ONE backend before enabling:
#   pip install deepface tf-keras          (recommended)
#   pip install face-recognition dlib      (alternative, needs C++ tools)

# Run face recognition every N pipeline frames (trade-off: speed vs latency)
FACE_RECOGNITION_INTERVAL = 7           # frames between recognition passes

# Minimum match score to accept a face as recognised (0.0 – 1.0)
# Lower = more permissive (more false positives)
# Higher = stricter (may miss valid matches)
FACE_SIMILARITY_THRESHOLD = 0.70

# Directory for storing uploaded face images
FACES_DIR = os.path.join(BASE_DIR, "data", "faces")

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "data", "watchdog.db")

# ── Web server ────────────────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# ── SORT tracker ──────────────────────────────────────────────────────────────
SORT_MAX_AGE   = 10   # frames to keep a lost track alive
SORT_MIN_HITS  = 2    # frames a detection must be seen before reporting
SORT_IOU_THRESH = 0.25
