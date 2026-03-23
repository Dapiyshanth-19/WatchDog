# WatchDog - System Architecture

> **AI-Powered Real-Time Crowd Surveillance System**
> Built with YOLOv11/v26 + SORT Tracking + Flask + OpenCV

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Breakdown](#4-component-breakdown)
5. [Data Flow & Pipeline](#5-data-flow--pipeline)
6. [Threading Model](#6-threading-model)
7. [REST API Reference](#7-rest-api-reference)
8. [Database Schema](#8-database-schema)
9. [Configuration Reference](#9-configuration-reference)
10. [Detection & Tracking Engine](#10-detection--tracking-engine)
11. [Feature Modules](#11-feature-modules)
12. [Frontend Dashboard](#12-frontend-dashboard)
13. [Deployment Modes](#13-deployment-modes)
14. [External Integrations](#14-external-integrations)
15. [Tech Stack Summary](#15-tech-stack-summary)

---

## 1. Project Overview

**WatchDog** is an advanced real-time AI surveillance system that combines deep learning object detection, multi-object tracking, behaviour analysis, face recognition, and gamification (Squid Game mode) into a single, modular Python application.

### Core Capabilities

| Capability              | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| **People Detection**    | YOLOv11/v26 real-time object detection                        |
| **Multi-Object Tracking** | SORT algorithm with Kalman filtering                        |
| **Behaviour Analysis**  | Standing, walking, running, loitering detection               |
| **Anomaly Detection**   | Stampede, dispersal, convergence, density spike, flow reversal|
| **Threat Detection**    | Fall, fire/flame, intrusion zones, crush risk detection       |
| **Predictive Analytics**| Crowd trend forecasting, risk assessment, hourly patterns     |
| **Multi-Camera**        | Independent pipelines per camera with cross-cam analytics     |
| **Face Recognition**    | Async identification via DeepFace or face_recognition         |
| **Squid Game Mode**     | Red Light/Green Light with elimination & win detection        |
| **PHANTOM VISION**      | Neon trails, thermal, X-ray, heatmaps, network visualization |
| **AI Voice Alerts**     | Multilingual TTS (EN, Tamil, Hindi, Spanish, French, Arabic)  |
| **Live Dashboard**      | Real-time web UI with analytics panels, threat zones, charts  |
| **Multi-Source Input**  | Webcam, RTSP, HTTP IP cameras, video files                    |

---

## 2. Directory Structure

```
WatchDog/
│
├── app.py                      # Flask web server & REST API (main entry point)
├── run_headless.py             # CLI mode entry point (no browser)
│
├── core/                       # Core infrastructure
│   ├── __init__.py
│   ├── config.py               # Central configuration & constants
│   ├── pipeline.py             # Core video processing pipeline
│   ├── sort.py                 # SORT multi-object tracking (Kalman)
│   ├── database.py             # SQLite persistence layer
│   └── multi_camera.py         # Multi-camera manager (cross-cam analytics)
│
├── engine/                     # Feature modules
│   ├── __init__.py
│   ├── behavior.py             # Behaviour analysis engine
│   ├── anomaly.py              # Crowd anomaly detection (stampede, dispersal, etc.)
│   ├── threat.py               # Threat detection (fall, fire, intrusion, crush)
│   ├── predictor.py            # Predictive crowd analytics & forecasting
│   ├── game.py                 # Squid Game mode state machine
│   ├── face_engine.py          # Face recognition subsystem (async)
│   ├── vision.py               # PHANTOM VISION visual effects
│   └── voice.py                # AI voice alert system (pyttsx3)
│
├── utils/                      # Utilities
│   ├── __init__.py
│   └── translations.py         # Multilingual strings (EN/Tamil/Hindi/Spanish/French/Arabic)
│
├── models/                     # ML model weights
│   └── yolo11n.pt              # YOLO model weights (~5.6 MB)
│
├── data/                       # Runtime data (git-ignored)
│   ├── watchdog.db             # SQLite database (auto-created)
│   └── faces/                  # Uploaded face images
│
├── templates/
│   └── index.html              # Web dashboard (single-page application)
│
├── requirements.txt            # Python dependencies
├── setup.bat                   # Windows automated setup script
├── README.md                   # Project documentation
├── ARCHITECTURE.md             # This file
└── .gitignore                  # Git ignore rules
```

### Package Roles at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                             │
│  app.py (Web + API)          run_headless.py (CLI/OpenCV)       │
└──────────────┬───────────────────────────┬──────────────────────┘
               │                           │
               ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     core/ — CORE ENGINE                         │
│  pipeline.py ──► sort.py (tracking)                             │
│       │          config.py (settings)                           │
│       │          database.py (SQLite)                           │
│       │                                                         │
│       ├──────────────────────────────────────────────────┐      │
│       ▼                                                  ▼      │
│  ┌──────────────────────────────┐  ┌──────────────────────┐    │
│  │  engine/ — FEATURE MODULES   │  │  utils/ — UTILITIES  │    │
│  │  behavior.py  (activities)   │  │  translations.py     │    │
│  │  anomaly.py   (crowd anomaly)│  │  (6 languages)       │    │
│  │  threat.py    (threats)      │  └──────────────────────┘    │
│  │  predictor.py (forecasting)  │                               │
│  │  game.py      (Squid Game)   │                               │
│  │  face_engine.py (faces)      │                               │
│  │  vision.py    (effects)      │                               │
│  │  voice.py     (TTS)          │                               │
│  └──────────────────────────────┘                               │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  models/yolo11n.pt    data/watchdog.db    data/faces/           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. High-Level Architecture

```
                    ┌──────────────────────┐
                    │    VIDEO SOURCES      │
                    │  Webcam / RTSP / HTTP │
                    │  / Video File         │
                    └──────────┬───────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      PIPELINE (pipeline.py)                      │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ Frame    │──►│ YOLO     │──►│ SORT     │──►│ Behaviour   │  │
│  │ Capture  │   │ Detect   │   │ Track    │   │ Analysis    │  │
│  │ (OpenCV) │   │(ultralytics)│ │(Kalman)  │   │(behavior.py)│  │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘  │
│                                                       │          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │          │
│  │ Anomaly  │◄──┤ Threat   │◄──┤ Predict  │◄─────────┘          │
│  │ Detect   │   │ Detect   │   │ Analytics│                     │
│  │(anomaly) │   │(threat)  │   │(predictor)                     │
│  └────┬─────┘   └────┬─────┘   └──────────┘                     │
│       │               │                                          │
│  ┌────▼─────┐   ┌────▼─────┐   ┌──────────┐                    │
│  │ Face     │   │ Game     │   │ Draw     │                     │
│  │ Recog.   │   │ Logic    │   │ Annotate │                     │
│  │(async)   │   │(game.py) │   │+ Zones   │                     │
│  └──────────┘   └──────────┘   └────┬─────┘                     │
│                                      │                           │
│  ┌──────────┐   ┌──────────┐        │                           │
│  │ Vision   │◄──┤ Voice    │        │                           │
│  │ Effects  │   │ Alerts   │        │                           │
│  │(vision.py)   │(voice.py)│        │                           │
│  └────┬─────┘   └──────────┘        │                           │
│       │                              │                           │
│       ▼                              ▼                           │
│  ┌─────────────────────────────────────────┐                    │
│  │         MJPEG Frame Buffer              │                    │
│  │   (Thread-safe, JPEG-encoded)           │                    │
│  └─────────────────┬───────────────────────┘                    │
└─────────────────────┼────────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Web      │ │ REST     │ │ SQLite   │
   │ Dashboard│ │ API      │ │ Database │
   │ (HTML)   │ │ (Flask)  │ │          │
   └──────────┘ └──────────┘ └──────────┘
```

---

## 4. Component Breakdown

### 4.1 `app.py` (root) — Flask Web Server & REST API

**Role**: Main entry point. Serves the web dashboard and exposes REST endpoints for controlling the pipeline.

**Responsibilities**:
- Serve `index.html` dashboard
- Start/stop video pipeline via API
- Stream MJPEG frames to browser
- Proxy configuration changes to pipeline
- Handle face registration/management
- Handle Squid Game control commands

**Key Pattern**: Acts as a thin controller layer — delegates all heavy processing to `pipeline.py`.

---

### 4.2 `core/pipeline.py` — Core Video Processing Pipeline

**Role**: The heart of the system. Runs the continuous video processing loop in a background thread.

**Responsibilities**:
- Capture frames from video source (OpenCV VideoCapture)
- Run YOLO inference for object detection
- Feed detections into SORT tracker
- Coordinate behaviour analysis, game logic, face recognition
- Apply visual effects (PHANTOM VISION)
- Encode processed frames as JPEG for streaming
- Emit alerts to database

**Processing Loop (per frame)**:
```
1. cap.read()                    → Raw frame (BGR)
2. cv2.resize(640x480)           → Normalized frame
3. model.predict(conf=0.45)      → YOLO detections [x1,y1,x2,y2,conf,cls]
4. sort.update(detections)       → Tracked objects [x1,y1,x2,y2,track_id]
5. behavior.update(tracks)       → Activity labels + alerts
6. game.process(tracks, frame)   → Game overlays + eliminations (if enabled)
7. face_engine.recognize(frame)  → Name labels (async, non-blocking)
8. draw_annotations(frame)       → Bounding boxes, labels, stats
9. vision.apply(frame)           → Visual effects stack
10. cv2.imencode('.jpg')         → JPEG buffer → shared frame variable
```

---

### 4.3 `core/sort.py` — SORT Multi-Object Tracker

**Role**: Assigns persistent IDs to detected objects across frames.

**Algorithm**:
```
SORT (Simple Online and Realtime Tracking)
├── Kalman Filter     → Predict next position of each tracked object
├── Hungarian Algo    → Match predictions to new detections (IoU-based)
├── Track Management  → Create new tracks, retire lost tracks
└── Output            → [x1, y1, x2, y2, track_id] per object
```

**Key Parameters**:
| Parameter       | Default | Purpose                              |
|-----------------|---------|--------------------------------------|
| `max_age`       | 10      | Frames before unmatched track dies   |
| `min_hits`      | 2       | Detections before track is confirmed |
| `iou_threshold` | 0.25    | Minimum IoU for detection-track match|

---

### 4.4 `engine/behavior.py` — Behaviour Analysis Engine

**Role**: Classifies each tracked person's activity based on movement speed over time.

**Activity Classification**:
```
Speed (px/frame)    Activity
─────────────────   ──────────
< 5                 Standing
5 – 40              Walking
> 40                Running  → generates alert
```

**Loitering Detection**:
```
If a person remains "Standing" for > 15 seconds → Loitering alert
Repeat alert every 10 seconds while loitering continues
```

**Crowd Monitoring**:
```
If total tracked people ≥ crowd_threshold (default: 8) → CrowdLimit alert
```

---

### 4.5 `engine/game.py` — Squid Game Mode (State Machine)

**Role**: Implements "Red Light / Green Light" game mechanics as a state machine overlay on top of the tracking pipeline.

**Two Operating Modes**:

```
┌──────────────────────────────────┬──────────────────────────────────┐
│         NORMAL MODE              │          GAME MODE               │
├──────────────────────────────────┼──────────────────────────────────┤
│ Standard surveillance            │ Red Light / Green Light game     │
│ Face recognition labels by name  │ Auto-numbered: Player 1, 2...   │
│ No elimination / finish line     │ Elimination on freeze violation  │
│ Behaviour analysis active        │ Winner on finish line crossing   │
└──────────────────────────────────┴──────────────────────────────────┘
```

**Game State Machine**:
```
                    ┌─────────────┐
            ┌──────►│  GREEN      │◄──────┐
            │       │  (Movement  │       │
            │       │   allowed)  │       │
            │       └──────┬──────┘       │
            │              │ timer        │
            │              ▼              │
            │       ┌─────────────┐       │
            └───────┤  RED        ├───────┘
                    │  (Freeze!   │  timer
                    │   Move=Die) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ ELIMINATION │ if movement > 12px
                    │ (Player out)│ during RED phase
                    └─────────────┘
```

**Win Condition**: Player crosses the configurable finish line (Y-coordinate) during GREEN phase.

**Anti-Cheat (Re-spawn Protection)**:
- Spatial re-ID: Tracks if a new track appears near an eliminated player's last position
- Face re-ID: Recognizes eliminated players by face even if they get new track IDs

---

### 4.6 `engine/face_engine.py` — Face Recognition Subsystem

**Role**: Asynchronously identifies tracked people by matching faces against registered embeddings.

**Architecture**:
```
Pipeline Frame (every Nth frame)
        │
        ▼
┌───────────────────┐
│  Crop face region  │
│  from bounding box │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     ┌─────────────┐
│  Compute embedding │────►│  Compare vs  │
│  (DeepFace/dlib)   │     │  registered  │
└───────────────────┘     │  faces (DB)  │
                          └──────┬──────┘
                                 │
                          similarity > 0.70?
                          ┌──────┴──────┐
                          │ YES         │ NO
                          ▼             ▼
                    Return name    Return "Unknown"
```

**Backends** (auto-detected, graceful fallback):
1. **DeepFace** (recommended) — `deepface` + `tf-keras`
2. **face_recognition** — `face-recognition` + `dlib`
3. **None** — Feature silently disabled if no backend installed

**Key Design Decision**: Runs asynchronously (non-blocking) to avoid slowing down the main pipeline. Only processes every 7th frame by default.

---

### 4.7 `engine/vision.py` — PHANTOM VISION Effects Engine

**Role**: Applies stackable visual effects on annotated frames for enhanced visualization.

**Vision Modes** (mutually exclusive base layer):
```
┌──────────┬───────────────────────────────────────────────────┐
│ Mode     │ Effect                                            │
├──────────┼───────────────────────────────────────────────────┤
│ Normal   │ Standard view with tracking overlays              │
│ Thermal  │ Inferno colormap heat effect                      │
│ X-Ray    │ Dark background, highlighted person regions       │
│ Neon     │ Darkened background for neon trail visibility      │
└──────────┴───────────────────────────────────────────────────┘
```

**Overlay Effects** (toggleable, stackable on any mode):
```
┌─────────────┬─────────────────────────────────────────────────┐
│ Effect      │ Description                                     │
├─────────────┼─────────────────────────────────────────────────┤
│ Trails      │ Neon motion history lines per tracked person    │
│ Network     │ Neural-network-style lines between nearby people│
│ Predictions │ Forward trajectory projection arrows            │
│ Heatmap     │ Crowd density heat visualization                │
│ Ripple      │ Expanding shockwave ring on crowd alerts        │
└─────────────┴─────────────────────────────────────────────────┘
```

---

### 4.8 `engine/voice.py` — AI Voice Alert System

**Role**: Delivers text-to-speech alerts using offline TTS engine.

**Features**:
- Uses `pyttsx3` (no internet required)
- Bilingual: English + Tamil
- Queue-based, non-blocking delivery
- Configurable speech rate (145 WPM)
- Graceful fallback if TTS backend unavailable

---

### 4.9 `core/database.py` — SQLite Persistence Layer

**Role**: Manages all persistent data storage via SQLite.

See [Section 8: Database Schema](#8-database-schema) for full table definitions.

---

### 4.10 `utils/translations.py` — Internationalization

**Role**: Centralized bilingual string store for English and Tamil.

**Contents**:
- Activity labels (running, loitering, standing, walking)
- Alert message templates with `{placeholders}`
- COCO class names (person, car, bicycle, etc.)
- Detection mode definitions

---

## 5. Data Flow & Pipeline

### 5.1 Frame Processing Flow

```
Video Source
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                    PER-FRAME PIPELINE                           │
│                                                                │
│  [Capture] → [Resize] → [YOLO Detect] → [SORT Track]          │
│                                              │                 │
│                              ┌────────────────┤                │
│                              ▼                ▼                │
│                       [Behaviour]      [Game Logic]            │
│                       [Analysis ]      [if enabled]            │
│                              │                │                │
│                              ▼                ▼                │
│                    [Face Recognition]   [Eliminations]         │
│                    [  (async, Nth)  ]   [  /Winners  ]         │
│                              │                │                │
│                              └────────┬───────┘                │
│                                       ▼                        │
│                              [Draw Annotations]                │
│                                       │                        │
│                                       ▼                        │
│                              [Vision Effects]                  │
│                                       │                        │
│                                       ▼                        │
│                              [JPEG Encode]                     │
│                                       │                        │
│                              ┌────────┼────────┐               │
│                              ▼        ▼        ▼               │
│                          [Stream] [Database] [Voice]           │
│                          [Buffer] [  Write ] [Alert]           │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Alert Flow

```
Behaviour/Game Event Detected
        │
        ├──► database.py  →  INSERT into alerts table
        ├──► voice.py     →  TTS queue (speaks alert aloud)
        └──► app.py       →  GET /alerts returns to frontend
                                    │
                                    ▼
                            Dashboard Activity Feed
```

### 5.3 Frontend ↔ Backend Communication

```
┌──────────────┐         HTTP/REST          ┌──────────────┐
│   Browser    │ ◄─────────────────────────► │  Flask API   │
│  (index.html)│                             │  (app.py)    │
│              │  GET  /video_feed (MJPEG)   │              │
│  <img src>   │ ◄━━━━━━━━━━━━━━━━━━━━━━━━━ │  Generator   │
│              │                             │              │
│  JS polling  │  GET  /status (JSON)        │              │
│  setInterval │ ◄─────────────────────────► │  get_state() │
│              │                             │              │
│  Button click│  POST /start, /stop, etc.   │              │
│              │ ─────────────────────────►  │  Pipeline    │
└──────────────┘                             └──────────────┘
```

---

## 6. Threading Model

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS: python app.py                    │
│                                                             │
│  ┌─────────────────┐                                        │
│  │  MAIN THREAD    │  Flask HTTP server                     │
│  │  (app.py)       │  Handles REST API requests             │
│  └────────┬────────┘                                        │
│           │ spawns                                           │
│  ┌────────▼────────┐                                        │
│  │ PIPELINE THREAD │  Video capture + YOLO + SORT           │
│  │ (pipeline.py)   │  Continuous frame processing loop      │
│  └────────┬────────┘                                        │
│           │ spawns                                           │
│  ┌────────▼────────┐                                        │
│  │ FACE WORKER     │  Async face recognition                │
│  │ (face_engine.py)│  Processes every Nth frame             │
│  └─────────────────┘                                        │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ VOICE WORKER    │  TTS queue consumer                    │
│  │ (voice.py)      │  Speaks alerts asynchronously          │
│  └─────────────────┘                                        │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ GAME TIMER      │  Phase switching (Green ↔ Red)         │
│  │ (game.py)       │  threading.Timer for auto-cycle        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘

Thread Communication:
  - Pipeline → Flask:     Shared frame buffer (thread-safe dict with lock)
  - Flask → Pipeline:     Shared config dict (thread-safe updates)
  - Pipeline → Database:  Direct SQLite writes (serialized)
  - Pipeline → Voice:     Queue-based message passing
  - Pipeline → Face:      Async callback pattern
```

---

## 7. REST API Reference

### Control Endpoints

| Method | Endpoint          | Description                        | Body / Params                |
|--------|-------------------|------------------------------------|------------------------------|
| `GET`  | `/`               | Serve web dashboard                | —                            |
| `GET`  | `/video_feed`     | MJPEG live video stream            | —                            |
| `GET`  | `/status`         | Pipeline status + stats            | —                            |
| `GET`  | `/alerts`         | Recent alerts                      | `?limit=N` (default 50)     |
| `POST` | `/start`          | Start pipeline                     | `{ "source": "0" }`         |
| `POST` | `/stop`           | Stop pipeline                      | —                            |
| `POST` | `/config`         | Set detection mode                 | `{ "mode": "people" }`      |
| `POST` | `/vision`         | Toggle vision effects              | `{ "mode": "thermal", ... }`|
| `POST` | `/model`          | Switch YOLO model at runtime       | `{ "model": "yolo26n.pt" }` |

### Face Recognition Endpoints

| Method | Endpoint          | Description                        | Body / Params                |
|--------|-------------------|------------------------------------|------------------------------|
| `POST` | `/face/register`  | Register a new face                | `{ "name": "...", image }` |
| `GET`  | `/face/list`      | List registered faces              | —                            |
| `DELETE`| `/face/delete`   | Remove a registered face           | `{ "name": "..." }`         |

### Squid Game Endpoints

| Method | Endpoint           | Description                       | Body / Params                |
|--------|--------------------|-----------------------------------|------------------------------|
| `POST` | `/game/configure`  | Configure game settings           | `{ total_duration, ... }`   |
| `POST` | `/game/start`      | Start Squid Game mode             | —                            |
| `POST` | `/game/freeze`     | Force freeze (red light)          | —                            |
| `POST` | `/game/unfreeze`   | Force green light                 | —                            |
| `POST` | `/game/reset`      | Reset game                        | —                            |
| `GET`  | `/game/status`     | Game state + players              | —                            |
| `GET`  | `/game/config`     | Current game config               | —                            |
| `GET`  | `/game/winners`    | List of winners                   | —                            |
| `POST` | `/game/set_zone`   | Set finish line position          | `{ "win_zone_y": 380 }`     |

### AI Analytics Endpoints

| Method | Endpoint              | Description                        | Body / Params                |
|--------|-----------------------|------------------------------------|------------------------------|
| `GET`  | `/analytics/crowd`    | Real-time crowd stats + anomalies  | —                            |
| `GET`  | `/analytics/prediction`| Trend forecast + risk + hourly    | —                            |
| `GET`  | `/analytics/threats`  | Active threats + zone list         | —                            |
| `GET`  | `/analytics/full`     | All analytics in one call          | —                            |

### Threat Zone Endpoints

| Method | Endpoint              | Description                        | Body / Params                |
|--------|-----------------------|------------------------------------|------------------------------|
| `GET`  | `/zones`              | List restricted zones              | —                            |
| `POST` | `/zones`              | Add a restricted zone              | `{ name, x1, y1, x2, y2 }` |
| `POST` | `/zones/clear`        | Clear all zones                    | —                            |

### Multi-Camera Endpoints

| Method | Endpoint               | Description                       | Body / Params                |
|--------|------------------------|-----------------------------------|------------------------------|
| `POST` | `/multicam/add`        | Add a camera feed                 | `{ "name": "...", "source": "..." }` |
| `POST` | `/multicam/remove`     | Remove a camera                   | `{ "cam_id": 1 }`           |
| `POST` | `/multicam/start`      | Start camera(s)                   | `{ "cam_id": 1 }` or omit for all |
| `POST` | `/multicam/stop`       | Stop camera(s)                    | `{ "cam_id": 1 }` or omit for all |
| `GET`  | `/multicam/status`     | Aggregate stats across all cams   | —                            |
| `GET`  | `/multicam/feed/<id>`  | MJPEG stream for specific camera  | —                            |
| `POST` | `/multicam/set_model`  | Set shared YOLO model             | `{ "model_path": "..." }`   |

---

## 8. Database Schema

**File**: `watchdog.db` (SQLite, auto-created at runtime)

```sql
┌─────────────────────────────────────────────────┐
│                    cameras                       │
├──────────┬──────────┬───────────────────────────┤
│ id (PK)  │ name     │ source_url                │
│ INTEGER  │ TEXT     │ TEXT                       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                    alerts                        │
├──────────┬───────────┬──────────┬───────────────┤
│ id (PK)  │ camera_id │timestamp │ type          │
│ INTEGER  │ INTEGER   │ TEXT     │ TEXT          │
├──────────┴───────────┴──────────┴───────────────┤
│ details (TEXT/JSON)                              │
│ { "details_en": "...",                          │
│   "details_ta": "...",                          │
│   "track_id": 5 }                               │
├─────────────────────────────────────────────────┤
│ Types: Running, Loitering, CrowdLimit,          │
│   Eliminated, Winner, FaceRecognized,            │
│   Anomaly:Stampede, Anomaly:Dispersal,           │
│   Anomaly:Convergence, Anomaly:DensitySpike,     │
│   Anomaly:FlowReversal, Threat:Fall,             │
│   Threat:Fire, Threat:Intrusion,                 │
│   Threat:CrushRisk                               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                    counts                        │
├──────────┬───────────┬──────────┬───────────────┤
│ id (PK)  │ camera_id │timestamp │ count         │
│ INTEGER  │ INTEGER   │ TEXT     │ INTEGER       │
├─────────────────────────────────────────────────┤
│ Logged every ~5 seconds for historical charts   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                    users                         │
├──────────┬──────────┬───────────┬───────────────┤
│ id (PK)  │ name     │ embedding │ image_path    │
│ INTEGER  │ TEXT     │ TEXT/JSON │ TEXT          │
├─────────────────────────────────────────────────┤
│ embedding: JSON array of float values           │
│ Used for face similarity matching               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                    players                       │
├──────────────┬──────────┬───────────────────────┤
│ track_id(PK) │ name     │ status                │
│ INTEGER      │ TEXT     │ TEXT                   │
├─────────────────────────────────────────────────┤
│ status: "alive" | "eliminated" | "winner"       │
│ Used during Squid Game mode                     │
└─────────────────────────────────────────────────┘
```

### Entity Relationship

```
cameras 1──────M alerts
cameras 1──────M counts
users   (standalone - face embeddings)
players (standalone - game state, ephemeral)
```

---

## 9. Configuration Reference

**File**: `config.py`

### Detection & Model

| Parameter               | Default          | Description                          |
|-------------------------|------------------|--------------------------------------|
| `MODEL_NAME`            | `"yolo26n.pt"`  | YOLO model weights file              |
| `CONFIDENCE_THRESHOLD`  | `0.45`           | Min detection confidence             |
| `CAMERA_SOURCE`         | `0`              | Default video source                 |
| `FRAME_WIDTH`           | `640`            | Processing frame width               |
| `FRAME_HEIGHT`          | `480`            | Processing frame height              |

### Tracking (SORT)

| Parameter               | Default | Description                              |
|-------------------------|---------|------------------------------------------|
| `SORT_MAX_AGE`          | `10`    | Frames before lost track is removed      |
| `SORT_MIN_HITS`         | `2`     | Min detections to confirm a track        |
| `SORT_IOU_THRESH`       | `0.25`  | IoU threshold for detection-track match  |

### Behaviour Thresholds

| Parameter                  | Default | Description                           |
|----------------------------|---------|---------------------------------------|
| `RUNNING_SPEED_THRESHOLD`  | `40`    | px/frame speed to classify as running |
| `LOITERING_TIME_THRESHOLD` | `15.0`  | Seconds standing = loitering          |
| `CROWD_THRESHOLD`          | `8`     | People count to trigger crowd alert   |

### Face Recognition

| Parameter                     | Default | Description                        |
|-------------------------------|---------|-------------------------------------|
| `FACE_RECOGNITION_INTERVAL`   | `7`     | Process every Nth frame             |
| `FACE_SIMILARITY_THRESHOLD`   | `0.70`  | Min similarity for positive match   |

### Squid Game

| Parameter                     | Default | Description                        |
|-------------------------------|---------|-------------------------------------|
| `FREEZE_MOVEMENT_THRESHOLD`   | `12`    | px movement in RED = elimination    |

### Server

| Parameter      | Default       | Description                |
|----------------|---------------|----------------------------|
| `SERVER_HOST`  | `"0.0.0.0"`  | Flask bind address         |
| `SERVER_PORT`  | `5000`        | Flask port                 |
| `DB_PATH`      | `"watchdog.db"`| SQLite database file path |

---

## 10. Detection & Tracking Engine

### YOLO Detection Pipeline

```
Raw Frame (640x480 BGR)
        │
        ▼
┌──────────────────────────────┐
│    YOLO Model Inference      │
│    model.predict(frame,      │
│      conf=0.45,              │
│      classes=[0,1,2,...])    │
│                              │
│    Returns per detection:    │
│    [x1, y1, x2, y2,         │
│     confidence, class_id]    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Detection Modes           │
│    ┌─────────┬────────────┐  │
│    │ people  │ class 0    │  │
│    │ vehicles│ 2,3,5,7    │  │
│    │ objects │ all COCO   │  │
│    │ all     │ all classes │  │
│    └─────────┴────────────┘  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    SORT Tracker Update       │
│                              │
│    Input:  [[x1,y1,x2,y2,c]]│
│    Output: [[x1,y1,x2,y2,id]]│
│                              │
│    Kalman predict → IoU match│
│    → Hungarian assignment    │
│    → Update/Create/Remove    │
└──────────────────────────────┘
```

### Detection Modes

| Mode      | COCO Classes                     | Use Case                  |
|-----------|----------------------------------|---------------------------|
| `people`  | Person (0)                       | Crowd surveillance        |
| `vehicles`| Bicycle, Car, Bus, Truck (2,3,5,7)| Traffic monitoring       |
| `objects` | All 80 COCO classes              | General object detection  |
| `all`     | All classes                      | Full scene analysis       |

---

## 11. Feature Modules

### 11.1 Squid Game — Full Game Flow

```
┌───────────────────────────────────────────────────────────────┐
│                    GAME LIFECYCLE                              │
│                                                               │
│  [POST /game/start]                                           │
│        │                                                      │
│        ▼                                                      │
│  ┌──────────┐    timer    ┌──────────┐    timer               │
│  │  GREEN   │ ──────────► │   RED    │ ──────────► GREEN...   │
│  │  LIGHT   │             │  LIGHT   │                        │
│  │(move OK) │ ◄────────── │(FREEZE!) │                        │
│  └──────────┘             └────┬─────┘                        │
│                                │                              │
│                     ┌──────────┼──────────┐                   │
│                     ▼                     ▼                    │
│            Movement > 12px?        Cross finish line?         │
│            ┌─────┐                 ┌─────┐                    │
│            │ YES │                 │ YES │                     │
│            │     ▼                 │     ▼                     │
│         ELIMINATED              WINNER                        │
│         (bbox turns red)        (bbox turns gold)             │
│         (voice: "eliminated!")  (voice: "wins!")              │
│         (DB: status=eliminated) (DB: status=winner)           │
│                                                               │
│  [POST /game/stop] → Reset all state, back to Normal Mode    │
└───────────────────────────────────────────────────────────────┘
```

### 11.2 PHANTOM VISION — Effects Stack

```
Base Frame (annotated)
        │
        ▼
┌──────────────────────────────┐
│  BASE MODE (pick one):       │
│  ├── Normal  (passthrough)   │
│  ├── Thermal (inferno cmap)  │
│  ├── X-Ray   (dark + glow)  │
│  └── Neon    (dark bg)       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  OVERLAY STACK (toggle any): │
│  ├── Trails     (motion paths)│
│  ├── Network    (connections) │
│  ├── Predictions(trajectory)  │
│  ├── Heatmap    (density)    │
│  └── Ripple     (alert wave) │
└──────────────┬───────────────┘
               │
               ▼
        Final Frame → Stream
```

### 11.3 Anomaly Detection — Crowd Anomaly Engine

```
engine/anomaly.py — AnomalyDetector

Detects unusual crowd motion patterns from SORT track data:
  • Stampede        — high average speed across all tracks
  • Dispersal       — tracks spreading outward from center
  • Convergence     — tracks converging toward a single point
  • Density Spike   — sudden jump in people count
  • Flow Reversal   — dominant motion direction flips 180°

Uses motion vectors computed from consecutive track centers.
Cooldown-based deduplication prevents alert floods.
```

### 11.4 Threat Detection — Multi-Threat Engine

```
engine/threat.py — ThreatDetector

Detects specific safety threats from frames + tracks:
  • Fall Detection      — aspect ratio change (tall→wide = fallen)
                          Confirmed after N consecutive wide-ratio frames
  • Fire Detection      — HSV color analysis for flame-colored regions
                          Filters by area, brightness, confidence
  • Intrusion Detection — configurable restricted zones
                          Alerts when person center enters zone
  • Crush Risk          — spatial density clustering
                          N+ people within 80px radius

Provides draw_zones() and draw_fire_overlay() for annotation.
```

### 11.5 Predictive Analytics — Crowd Forecasting

```
engine/predictor.py — CrowdPredictor

Time-series crowd trend forecasting:
  • get_trend()         — current trend (rising/stable/declining),
                          rate of change, 5-min & 15-min forecasts,
                          confidence score, peak/avg today
  • get_hourly_forecast() — historical avg/peak/low per hour
  • get_risk_assessment() — risk level (low/moderate/high/critical)
                            with contributing factors and 0-100 score

Uses simple linear regression (no external ML framework needed).
```

### 11.6 Multi-Camera Manager

```
core/multi_camera.py — MultiCameraManager + CameraFeed

Manages multiple video feeds with independent pipelines:
  • Each camera runs its own YOLO + SORT + BehaviorAnalyzer
  • Thread-per-camera architecture
  • Per-camera MJPEG frame buffer
  • Cross-camera aggregate statistics
  • Shared YOLO model instance

API: add/remove/start/stop cameras, get individual or aggregate stats.
```

---

## 12. Frontend Dashboard

**File**: `templates/index.html` (~1064 lines, single-page application)

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: WatchDog — AI Crowd Surveillance                   │
│  [Language Toggle: EN / தமிழ்]                                │
├─────────────────────────────────────┬───────────────────────┤
│                                     │                       │
│  ┌───────────────────────────────┐  │  CONTROL PANEL        │
│  │                               │  │  ┌─────────────────┐  │
│  │     LIVE VIDEO STREAM         │  │  │ Source Input     │  │
│  │     (MJPEG <img> tag)         │  │  │ [Start] [Stop]  │  │
│  │                               │  │  └─────────────────┘  │
│  │     + Canvas overlay for      │  │                       │
│  │       finish line drawing     │  │  Detection Mode:      │
│  │                               │  │  [People][Vehicles]   │
│  └───────────────────────────────┘  │  [Objects][All]       │
│                                     │                       │
│  STATS BAR:                         │  Vision Mode:         │
│  [People][Objects][FPS]             │  [Normal][Neon]       │
│  [Alerts][Threats][Risk]            │  [Thermal][X-Ray]     │
├─────────────────────────────────────┤                       │
│                                     │  ACTIVITY FEED        │
│  VISION / DETECTION / GAME PANELS   │  PLAYER LIST          │
│                                     │  WINNERS PANEL        │
│  HISTORICAL CHART (Chart.js)        │                       │
│  ┌───────────────────────────────┐  │  THREAT ALERTS        │
│  │  People count over time 📈    │  │  ┌─────────────────┐  │
│  └───────────────────────────────┘  │  │ 🆘 Fall Detected │  │
│                                     │  │ 🔥 Fire Detected │  │
│  AI ANALYTICS GRID (2x2):          │  └─────────────────┘  │
│  ┌──────────────┬───────────────┐  │                       │
│  │ Crowd Trend  │ Risk Assess.  │  │  RESTRICTED ZONES     │
│  │ ▲ +2.1/min   │ MODERATE 45   │  │  [Add][Clear]         │
│  │ 5min: 12     │ Factors: ...  │  │                       │
│  │ 15min: 18    │               │  │  FACE REGISTRATION    │
│  ├──────────────┼───────────────┤  │                       │
│  │ Crowd Stats  │ Pred. Chart   │  │                       │
│  │ Speed/Dense  │ ┌──── 📈 ───┐│  │                       │
│  │ Spread/Dir   │ └───────────┘│  │                       │
│  └──────────────┴───────────────┘  │                       │
│                                     │                       │
│                                     │  FACE MANAGEMENT      │
│                                     │  [Register] [List]    │
└─────────────────────────────────────┴───────────────────────┘
```

### Frontend → Backend Polling

```javascript
// Status polling (every 1 second)
setInterval(() => fetch('/status').then(updateUI), 1000);

// Alerts polling (every 2 seconds)
setInterval(() => fetch('/alerts?limit=20').then(updateFeed), 2000);

// Video stream (continuous MJPEG)
<img src="/video_feed">  // Browser handles multipart stream natively
```

---

## 13. Deployment Modes

### Mode 1: Web Dashboard (Primary)

```bash
python app.py
# → Flask server on http://localhost:5000
# → Open browser to access dashboard
# → Start pipeline via UI (select source)
```

### Mode 2: Headless / CLI

```bash
python run_headless.py 0                              # USB webcam
python run_headless.py path/to/video.mp4              # Video file
python run_headless.py rtsp://user:pass@ip:554/stream # RTSP camera
python run_headless.py http://192.168.1.5:8080/video  # IP Webcam app
python run_headless.py http://127.0.0.1:4747/video    # DroidCam
```

### Quick Setup (Windows)

```bash
setup.bat
# Creates venv → installs dependencies → downloads YOLO model → launches app
```

---

## 14. External Integrations

### Supported Video Sources

```
┌────────────────────┬──────────────────────────────────────────┐
│ Source Type         │ Connection String                        │
├────────────────────┼──────────────────────────────────────────┤
│ USB Webcam         │ 0, 1, 2 (device index)                  │
│ RTSP IP Camera     │ rtsp://user:pass@192.168.x.x:554/stream │
│ HTTP IP Webcam     │ http://192.168.x.x:8080/video           │
│ DroidCam (USB)     │ http://127.0.0.1:4747/video             │
│ Local Video File   │ /path/to/video.mp4                      │
└────────────────────┴──────────────────────────────────────────┘
```

### ML Model Support

```
┌─────────────────┬────────────────────────────────────────────┐
│ Model           │ Notes                                      │
├─────────────────┼────────────────────────────────────────────┤
│ YOLOv11n        │ Default, fast, lightweight (~5.6 MB)       │
│ YOLOv26n        │ Newer architecture, swappable at runtime   │
│ Custom .pt      │ Any Ultralytics-compatible model           │
└─────────────────┴────────────────────────────────────────────┘
```

### Face Recognition Backends

```
┌──────────────────┬─────────────────────────────────────────────┐
│ Backend          │ Install Command                             │
├──────────────────┼─────────────────────────────────────────────┤
│ DeepFace (rec.)  │ pip install deepface tf-keras               │
│ face_recognition │ pip install face-recognition dlib           │
│ None (disabled)  │ Feature silently skipped                    │
└──────────────────┴─────────────────────────────────────────────┘
```

---

## 15. Tech Stack Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        TECH STACK                               │
├────────────────┬────────────────────────────────────────────────┤
│ LAYER          │ TECHNOLOGY                                     │
├────────────────┼────────────────────────────────────────────────┤
│ ML Detection   │ Ultralytics YOLOv11/v26                       │
│ Tracking       │ SORT + Kalman Filter (filterpy, scipy)        │
│ Face Recog.    │ DeepFace / face_recognition (optional)        │
│ Video I/O      │ OpenCV (opencv-python)                        │
│ Web Server     │ Flask + Flask-CORS                            │
│ Database       │ SQLite3 (built-in Python)                     │
│ TTS            │ pyttsx3 (offline)                             │
│ Frontend       │ Vanilla HTML/CSS/JS + Chart.js                │
│ Streaming      │ MJPEG over HTTP (multipart/x-mixed-replace)  │
│ Language       │ Python 3.8+                                   │
│ Platform       │ Windows / Linux / macOS                       │
└────────────────┴────────────────────────────────────────────────┘
```

### Dependency Graph

```
app.py (Flask)
  ├── core/pipeline.py
  │     ├── ultralytics (YOLO)
  │     ├── opencv-python (VideoCapture, imencode)
  │     ├── core/sort.py
  │     │     ├── filterpy (KalmanFilter)
  │     │     ├── scipy (linear_sum_assignment)
  │     │     └── numpy
  │     ├── engine/behavior.py
  │     ├── engine/game.py
  │     │     └── core/config.py
  │     ├── engine/face_engine.py
  │     │     ├── deepface (optional)
  │     │     ├── face_recognition (optional)
  │     │     └── core/database.py (sqlite3)
  │     ├── engine/vision.py
  │     │     └── numpy, opencv
  │     └── engine/voice.py
  │           └── pyttsx3
  ├── core/database.py
  ├── core/config.py
  └── utils/translations.py
```

---

## Privacy & Design Principles

| Principle                | Implementation                                          |
|--------------------------|---------------------------------------------------------|
| **No frame storage**     | Frames are never saved to disk                          |
| **Embedding-only faces** | Only numerical embeddings stored, not images            |
| **Minimal PII**          | Alerts contain track IDs + timestamps, not personal data|
| **Offline-first**        | No cloud dependencies, all processing is local          |
| **Graceful degradation** | Optional features (face, voice) silently disabled if unavailable |
| **Modular architecture** | Each subsystem is independent and replaceable           |
| **Thread-safe**          | Shared state protected with locks, queues               |
| **Runtime-configurable** | Model, mode, effects all switchable without restart     |

---

*Generated for WatchDog AI Crowd Surveillance System*
