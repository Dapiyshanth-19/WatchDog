# WatchDog – AI Crowd Surveillance System

Real-time crowd monitoring using **YOLOv11/v26 + SORT tracking + behaviour rules**, served through a live Flask web dashboard.

---

## Quick Start

### 1. Install dependencies
```bash
# Windows
setup.bat

# Linux / macOS
pip install -r requirements.txt
```

### 2. Run (Web dashboard)
```bash
python app.py
```
Open **http://localhost:5000** in your browser, press **Start**.

### 3. Run (Headless – OpenCV window)
```bash
python run_headless.py            # default webcam
python run_headless.py sample.mp4 # video file
python run_headless.py 1          # second webcam
python run_headless.py rtsp://user:pass@192.168.1.20:554/stream1
```
Press **Q** to quit.

---

## Video Sources

| Source | Value to use |
|--------|-------------|
| Default webcam | `0` |
| Second webcam | `1` |
| Video file | `/path/to/video.mp4` |
| Android IP Webcam (WiFi) | `http://192.168.x.x:8080/video` |
| DroidCam (USB) | `http://127.0.0.1:4747/video` |
| RTSP IP camera | `rtsp://user:pass@192.168.x.x:554/stream1` |

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_SOURCE` | `0` | Default video source |
| `MODEL_NAME` | `yolo11n.pt` | YOLO model to use |
| `CONFIDENCE_THRESHOLD` | `0.45` | Detection confidence |
| `CROWD_THRESHOLD` | `8` | People count to trigger crowd alert |
| `RUNNING_SPEED_THRESHOLD` | `40` | px/frame to classify as running |
| `LOITERING_TIME_THRESHOLD` | `15.0` | Seconds stationary = loitering |

---

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/video_feed` | GET | MJPEG live stream |
| `/status` | GET | `{running, fps, count, error}` |
| `/alerts?limit=N` | GET | Recent alert log |
| `/cameras` | GET | Configured cameras |
| `/start` | POST | `{"source": 0}` – start pipeline |
| `/stop` | POST | Stop pipeline |
| `/alert` | POST | `{"type":"Manual","details":"..."}` |

---

## Architecture

```
Camera ──► Frame Capture ──► YOLOv11/26 ──► SORT Tracker
                                                  │
                              ┌───────────────────┘
                              ▼
                      Behaviour Analyser
                      (running / loitering / crowd)
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
                 SQLite DB        Flask API
                 (alerts)         (JSON + MJPEG)
                                       │
                                  Web Dashboard
```

---

## Privacy

- **No face recognition** – only anonymous bounding boxes are processed.
- Frames are **not stored** to disk.
- Alerts contain only track IDs and timestamps – no PII.

---

## Upgrading to YOLOv26

When Ultralytics releases YOLOv26 weights, update `config.py`:
```python
MODEL_NAME = "yolo26n.pt"
```
No other changes needed – the pipeline uses the generic Ultralytics API.
