"""
WatchDog – Central configuration
Edit these values to match your environment before running.
"""

# ── Video source ────────────────────────────────────────────────────────────
# 0          = default USB/built-in webcam
# 1, 2, …   = other USB cameras
# "path/to/video.mp4"                        = local video file (great for demo)
# "http://192.168.x.x:8080/video"            = IP Webcam (Android app, WiFi)
# "rtsp://user:pass@192.168.x.x:554/stream1" = RTSP IP camera
CAMERA_SOURCE = 0

# ── YOLO model ──────────────────────────────────────────────────────────────
# "yolo11n.pt"  – fast nano model (recommended for CPU / demo)
# "yolo11s.pt"  – small, slightly more accurate
# "yolo26n.pt"  – use when YOLOv26 weights become available
MODEL_NAME = "yolo11n.pt"

# Only report detections above this confidence
CONFIDENCE_THRESHOLD = 0.45

# COCO class id for "person" is 0 — keep this unless you change models
PERSON_CLASS_ID = 0

# ── Frame processing ─────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Behaviour thresholds ─────────────────────────────────────────────────────
# Crowd alert: warn when this many people are in the frame at once
CROWD_THRESHOLD = 8

# Running: flag if a track moves more than this many pixels between frames
RUNNING_SPEED_THRESHOLD = 40   # pixels/frame

# Loitering: flag if a track barely moves for longer than this duration
LOITERING_PIXEL_THRESHOLD = 15          # px; "barely moving" definition
LOITERING_TIME_THRESHOLD  = 15.0        # seconds

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH = "watchdog.db"

# ── Web server ────────────────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# ── SORT tracker ─────────────────────────────────────────────────────────────
SORT_MAX_AGE   = 10   # frames to keep a lost track alive
SORT_MIN_HITS  = 2    # frames a detection must be seen before reporting
SORT_IOU_THRESH = 0.25
