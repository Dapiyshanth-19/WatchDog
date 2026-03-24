# -*- coding: utf-8 -*-
"""
WatchDog – Core video processing pipeline (v5 – Full AI Suite).

Frame flow:
    capture → resize → YOLO detect → SORT track → behaviour analyse
    → anomaly detect → threat detect → face recognition (async)
    → game logic → predictive analytics → draw boxes → vision FX → stream
"""

import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

from core.config import (
    CAMERA_SOURCE, MODEL_NAME, MODEL_PATH, CONFIDENCE_THRESHOLD,
    FRAME_WIDTH, FRAME_HEIGHT,
    SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESH,
    CROWD_THRESHOLD, MAX_DETECTION_AREA_RATIO,
)
from core.sort import Sort
from engine.behavior import BehaviorAnalyzer
from engine.vision import VisionEngine
from engine.anomaly import AnomalyDetector
from engine.threat import ThreatDetector
from engine.predictor import CrowdPredictor
from engine import voice
from utils.translations import DETECTION_MODES, class_name, COCO_CLASS_MAP
from core import database as db
from engine import game
from engine import face_engine


# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "running":       False,
    "fps":           0.0,
    "count":         0,
    "obj_count":     0,
    "tracks":        [],
    "alerts":        [],
    "frame_jpg":     None,
    "error":         None,
    "game":          {},
    "anomalies":     [],
    "threats":       [],
    "crowd_stats":   {},
    "prediction":    {},
    "risk":          {},
}

# ── Shared instances (accessible from app.py for API) ────────────────────────
anomaly_detector: AnomalyDetector | None = None
threat_detector:  ThreatDetector  | None = None
crowd_predictor:  CrowdPredictor  | None = None

_cfg = {
    "detection_mode": "people",
    "source":         CAMERA_SOURCE,
    "vision_mode":    "normal",
    "trails_on":      False,
    "network_on":     False,
    "predict_on":     False,
    "heatmap_on":     False,
    "voice_on":       True,
    "voice_lang":     "en",
    "model_path":     MODEL_PATH,
}

_lock       = threading.Lock()
_stop_event = threading.Event()
_thread: threading.Thread | None = None


# ── Colours ────────────────────────────────────────────────────────────────────
_PALETTE = [
    (0,255,100),(255,140,0),(0,140,255),(255,0,140),
    (140,255,0),(0,255,200),(140,0,255),(255,255,0),
    (0,200,255),(255,80,80),(80,255,80),(80,80,255),
]
def _color(tid): return _PALETTE[int(tid) % len(_PALETTE)]

_ALERT_COLOR  = (0, 60, 255)
_OBJECT_COLOR = (255, 200, 0)
_ELIM_COLOR   = (0, 0, 230)
_WIN_COLOR    = (0, 215, 255)    # gold for winners
_TARGET_COLOR = (50, 180, 255)   # highlighted target


def _build_live_narration(track_data: list[dict], lang: str = "en") -> str:
    """Build scene narration text for continuous voice updates."""
    people = [t for t in track_data if t.get("class_id") == 0]

    if lang == "ta":
        if not people:
            return "கேமராவில் இப்போது யாரும் இல்லை."

        standing = []
        sitting = []
        others = []
        for p in people:
            act = (p.get("activity_ta") or "").strip()
            tid = p.get("id")
            if "அமர" in act:
                sitting.append(tid)
            elif "நிற்க" in act:
                standing.append(tid)
            else:
                others.append((tid, act or "நிற்கிறார்"))

        parts = [f"கேமராவில் {len(people)} நபர்கள் உள்ளனர்."]
        if standing:
            parts.append("நிற்கும் நபர்கள்: " + ", ".join(f"{i}" for i in standing) + ".")
        if sitting:
            parts.append("அமர்ந்திருக்கும் நபர்கள்: " + ", ".join(f"{i}" for i in sitting) + ".")
        for tid, act in others[:4]:
            parts.append(f"நபர் {tid} {act}.")
        return " ".join(parts)

    if not people:
        return "No person is visible in the camera."

    standing = []
    sitting = []
    others = []
    for p in people:
        act = (p.get("activity_en") or "").strip().lower()
        tid = p.get("id")
        if "sit" in act:
            sitting.append(tid)
        elif "stand" in act:
            standing.append(tid)
        else:
            others.append((tid, act or "standing"))

    parts = [f"{len(people)} people are visible in the camera."]
    if standing:
        parts.append("Standing persons: " + ", ".join(str(i) for i in standing) + ".")
    if sitting:
        parts.append("Sitting persons: " + ", ".join(str(i) for i in sitting) + ".")
    for tid, act in others[:4]:
        parts.append(f"Person {tid} is {act}.")
    return " ".join(parts)


# ── Box / overlay drawing ──────────────────────────────────────────────────────
def _draw_boxes(frame, track_data, player_snapshot: dict | None = None,
                target_name: str = ""):
    """
    Draw bounding boxes.

    player_snapshot : int_key dict from game.get_status()["players"]
    target_name     : _cfg target player name (gets golden box)
    """
    ps = player_snapshot or {}

    for td in track_data:
        x1, y1, x2, y2 = int(td["x1"]), int(td["y1"]), int(td["x2"]), int(td["y2"])
        tid       = td["id"]
        cls_id    = td.get("class_id", 0)
        is_alert  = td.get("is_alert", False)
        is_person = cls_id == 0

        pinfo         = ps.get(tid, {})
        status        = pinfo.get("status", "alive")
        player_name   = pinfo.get("name", "")
        is_target     = pinfo.get("is_target", False)
        is_eliminated = status == "eliminated"
        is_winner     = status == "winner"

        # ── Box colour ────────────────────────────────────────────────────
        if is_eliminated:
            color = _ELIM_COLOR
        elif is_winner:
            color = _WIN_COLOR
        elif is_target:
            color = _TARGET_COLOR
        elif is_alert:
            color = _ALERT_COLOR
        elif is_person:
            color = _color(tid)
        else:
            color = _OBJECT_COLOR

        # Thick border for target
        thickness = 3 if is_target else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # ── Label ─────────────────────────────────────────────────────────
        # Keep labels clean: no numeric IDs in UI.
        if is_person:
            if player_name and not player_name.startswith("Unknown"):
                label = player_name
            else:
                label = "Person"
        else:
            label = class_name(cls_id, "en")

        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lsz[1] - 6), (x1 + lsz[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Sub-label (status / activity) ─────────────────────────────────
        h_frame = frame.shape[0]
        # If y2+16 would be off-screen, draw inside bbox instead
        sub_y = y2 + 16 if y2 + 20 < h_frame else y2 - 24
        if is_eliminated:
            cv2.putText(frame, "ELIMINATED", (x1, sub_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _ELIM_COLOR, 2, cv2.LINE_AA)
        elif is_winner:
            cv2.putText(frame, "WINNER!", (x1, sub_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WIN_COLOR, 2, cv2.LINE_AA)
        elif is_person:
            act = td.get("activity_en", "")
            if act:
                cv2.putText(frame, act, (x1, sub_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_finish_line(frame, gstatus: dict):
    """Draw the finish line / win zone on the frame."""
    win_zone_y = gstatus.get("win_zone_y", 0)
    if win_zone_y <= 0:
        return
    h, w = frame.shape[:2]
    x1 = gstatus.get("win_zone_x1") or 0
    x2 = gstatus.get("win_zone_x2") or w

    # Dashed green line
    dash_len = 20
    for sx in range(x1, x2, dash_len * 2):
        ex = min(sx + dash_len, x2)
        cv2.line(frame, (sx, win_zone_y), (ex, win_zone_y), (0, 240, 80), 3, cv2.LINE_AA)

    # "FINISH" label with background
    label = "FINISH"
    lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    tx = x1 + 8
    ty = win_zone_y - 6
    cv2.rectangle(frame, (tx - 4, ty - lsz[1] - 4), (tx + lsz[0] + 4, ty + 4),
                  (0, 60, 0), -1)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 80), 2, cv2.LINE_AA)


def _draw_game_overlay(frame, gstatus: dict):
    """Phase banner + timers at the bottom of the frame."""
    if not gstatus.get("game_active"):
        return
    h, w   = frame.shape[:2]
    phase  = gstatus.get("current_phase", "")
    r_ph   = gstatus.get("remaining_phase", 0)
    r_gm   = gstatus.get("remaining_game",  0)
    total  = gstatus.get("total_duration",  0)

    if phase == "moving":
        label, color = "GREEN LIGHT – MOVE!", (0, 220, 50)
    elif phase == "frozen":
        label, color = "RED LIGHT – FREEZE!", (40, 40, 230)
    elif phase == "ended":
        label, color = "GAME OVER", (180, 180, 180)
    else:
        return

    # Semi-transparent bar at bottom
    bar_h = 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Phase label
    tsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
    tx  = (w - tsz[0]) // 2
    cv2.putText(frame, label, (tx, h - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)

    # Phase countdown (left side)
    if r_ph > 0:
        ph_str = f"{r_ph:.1f}s"
        cv2.putText(frame, ph_str, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Game countdown (right side)
    if total > 0 and r_gm > 0:
        gm_str = f"Game: {r_gm:.0f}s"
        gtsz = cv2.getTextSize(gm_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, gm_str, (w - gtsz[0] - 8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_hud(frame, person_count, obj_count, fps, events, mode):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cc = (0, 0, 255) if person_count >= CROWD_THRESHOLD else (0, 255, 100)
    cv2.putText(frame, f"People: {person_count}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cc, 2, cv2.LINE_AA)
    if obj_count:
        cv2.putText(frame, f"Objects: {obj_count}", (180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, mode.upper(), (w // 2 - 30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 108, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2, cv2.LINE_AA)

    y = 62
    for ev in events[:3]:
        cv2.putText(frame, f"! {ev['details_en'][:52]}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 80, 255), 2, cv2.LINE_AA)
        y += 19


# ── Camera open helper ─────────────────────────────────────────────────────────
def _open_capture(source):
    raw = str(source).strip()

    if raw.isdigit():
        src = int(raw)
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return src, cap
        _set_error(f"Webcam {src} not found.")
        return src, None

    if raw.startswith("http://") or raw.startswith("https://"):
        from urllib.parse import urlparse
        parsed       = urlparse(raw.rstrip("/"))
        base_no_path = f"{parsed.scheme}://{parsed.netloc}"
        candidates   = []
        if parsed.path and parsed.path != "/":
            candidates.append(raw)
        candidates += [
            base_no_path + "/video",
            base_no_path + "/mjpeg",
            base_no_path + "/shot.jpg",
            base_no_path + "/videostream.cgi",
            base_no_path + "/stream",
        ]
        for url in candidates:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return url, cap
                cap.release()
        _set_error(f"Could not connect to {base_no_path}.")
        return source, None

    cap = cv2.VideoCapture(raw)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return raw, cap
        cap.release()
        _set_error(f"Opened {raw} but could not read frames.")
        return raw, None

    _set_error(f"Cannot open: {raw}")
    return raw, None


def _set_error(msg: str):
    with _lock:
        state["error"]   = msg
        state["running"] = False


# ── Pipeline thread ────────────────────────────────────────────────────────────
def _run(initial_source, model, model_path):
    with _lock:
        state["error"]   = None
        state["running"] = True

    try:
        global anomaly_detector, threat_detector, crowd_predictor

        tracker  = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS,
                        iou_threshold=SORT_IOU_THRESH)
        analyzer = BehaviorAnalyzer()
        vision   = VisionEngine(frame_w=FRAME_WIDTH, frame_h=FRAME_HEIGHT)

        # New AI modules
        anomaly_detector = AnomalyDetector()
        threat_detector  = ThreatDetector()
        crowd_predictor  = CrowdPredictor()

        voice.start()
        voice.set_enabled(_cfg.get("voice_on", True))
        face_engine.clear_results()

        cap_source, cap = _open_capture(initial_source)
        if cap is None:
            return

        fps_avg      = 0.0
        t_prev       = time.time()
        t_count_log  = time.time()
        t_narration  = 0.0
        COUNT_LOG_INT = 5.0
        NARRATION_INTERVAL = 4.0

        while not _stop_event.is_set():

            # ── Dynamic model reload ───────────────────────────────────────
            new_path = _cfg.get("model_path", MODEL_PATH)
            if new_path != model_path:
                try:
                    model      = YOLO(new_path)
                    model_path = new_path
                except Exception as exc:
                    _cfg["model_path"] = model_path
                    with _lock:
                        state["error"] = f"Model reload failed: {exc}"

            ret, frame = cap.read()
            if not ret:
                if isinstance(cap_source, int) or str(cap_source).startswith(("http", "rtsp")):
                    with _lock:
                        state["error"] = "Camera disconnected or stream ended."
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            mode        = _cfg["detection_mode"]
            classes     = DETECTION_MODES.get(mode, [0])
            vision_mode = _cfg["vision_mode"]
            v_lang      = _cfg.get("voice_lang", "en")

            # ── Detection ─────────────────────────────────────────────────
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                            classes=classes)
            boxes = results[0].boxes

            person_dets = np.empty((0, 5))
            obj_dets    = []

            frame_area = FRAME_WIDTH * FRAME_HEIGHT
            max_det_area = frame_area * MAX_DETECTION_AREA_RATIO

            if boxes is not None and len(boxes):
                xyxy    = boxes.xyxy.cpu().numpy()
                confs   = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cf, cid = confs[i], cls_ids[i]
                    # Skip oversized detections (person too close to camera)
                    det_area = (x2 - x1) * (y2 - y1)
                    if cid == 0 and det_area > max_det_area:
                        continue
                    if cid == 0:
                        person_dets = np.vstack([person_dets, [x1, y1, x2, y2, cf]])
                    else:
                        obj_dets.append((x1, y1, x2, y2, cf, cid))

            # ── Tracking ──────────────────────────────────────────────────
            tracks = tracker.update(person_dets)

            # ── Behaviour analysis ────────────────────────────────────────
            now    = time.time()
            events = analyzer.update(tracks, now)
            alert_ids   = {ev["track_id"] for ev in events}
            crowd_alarm = any(ev["type"] == "CrowdLimit" for ev in events)

            for ev in events:
                db.log_alert(1, ev["type"], {
                    "details":    ev["details_en"],
                    "details_ta": ev["details_ta"],
                    "track_id":   ev["track_id"],
                })
                speak_text = ev["details_ta"] if v_lang == "ta" else ev["details_en"]
                voice.speak(speak_text, v_lang)

            if now - t_count_log >= COUNT_LOG_INT:
                db.log_count(1, len(tracks))
                crowd_predictor.record(len(tracks), now)
                t_count_log = now

            # ── Anomaly detection ────────────────────────────────────────
            anomaly_events = anomaly_detector.update(tracks, now)
            crowd_stats = anomaly_detector.get_crowd_stats(tracks)
            for aev in anomaly_events:
                db.log_alert(1, aev["type"], {
                    "details":    aev["details_en"],
                    "details_ta": aev["details_ta"],
                    "track_id":   aev["track_id"],
                })
                speak_text = aev["details_ta"] if v_lang == "ta" else aev["details_en"]
                voice.speak(speak_text, v_lang)
                alert_ids.add(aev["track_id"])

            # ── Threat detection ─────────────────────────────────────────
            threat_events = threat_detector.update(frame, tracks, now)
            for tev in threat_events:
                db.log_alert(1, tev["type"], {
                    "details":    tev["details_en"],
                    "details_ta": tev["details_ta"],
                    "track_id":   tev["track_id"],
                })
                speak_text = tev["details_ta"] if v_lang == "ta" else tev["details_en"]
                voice.speak(speak_text, v_lang)

            # ── Face recognition (async, non-blocking) ────────────────────
            face_engine.submit_frame(frame, tracks)
            face_names = face_engine.get_results()
            for tid, name in face_names.items():
                game.assign_name(tid, name)

            # ── Game logic ────────────────────────────────────────────────
            game_events = game.update(tracks, FRAME_WIDTH, FRAME_HEIGHT)
            for gev in game_events:
                db.log_alert(1, gev["type"], {
                    "details":    gev["details_en"],
                    "details_ta": gev["details_ta"],
                    "track_id":   gev["track_id"],
                })
                speak_text = gev["details_ta"] if v_lang == "ta" else gev["details_en"]
                voice.speak(speak_text, v_lang)

            all_events = events + game_events + anomaly_events + threat_events

            # Snapshot game state
            gstatus         = game.get_status()
            player_snapshot = {int(k): v for k, v in gstatus["players"].items()}
            target_name     = gstatus.get("target_name", "")

            # ── Build track data ─────────────────────────────────────────
            track_data = []
            for trk in tracks:
                x1, y1, x2, y2, tid = trk
                tid   = int(tid)
                pinfo = player_snapshot.get(tid, {})
                track_data.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "id": tid, "class_id": 0,
                    "activity_en": analyzer.get_activity_label(tid, "en"),
                    "activity_ta": analyzer.get_activity_label(tid, "ta"),
                    "is_alert":    tid in alert_ids,
                    "name":        pinfo.get("name", ""),
                    "player_number": pinfo.get("player_number", 0),
                    "status":      pinfo.get("status", "alive"),
                    "is_target":   pinfo.get("is_target", False),
                })

            obj_data = []
            for (x1, y1, x2, y2, cf, cid) in obj_dets:
                obj_data.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "id": -1, "class_id": int(cid),
                    "activity_en": class_name(cid, "en"),
                    "activity_ta": class_name(cid, "ta"),
                    "is_alert": False, "name": "", "player_number": 0,
                    "status": "alive", "is_target": False,
                })

            # ── Continuous scene narration ────────────────────────────────
            if _cfg.get("voice_on", False):
                if (now - t_narration) >= NARRATION_INTERVAL:
                    voice.speak_live(_build_live_narration(track_data, v_lang), v_lang)
                    t_narration = now
            else:
                t_narration = 0.0

            # ── FPS ───────────────────────────────────────────────────────
            t_now   = time.time()
            fps_avg = 0.9 * fps_avg + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev  = t_now

            # ── Annotate ──────────────────────────────────────────────────
            annotated = frame.copy()
            _draw_finish_line(annotated, gstatus)
            threat_detector.draw_zones(annotated)
            threat_detector.draw_fire_overlay(annotated)
            _draw_boxes(annotated, track_data + obj_data, player_snapshot, target_name)
            _draw_game_overlay(annotated, gstatus)

            vision.mode       = vision_mode
            vision.trails_on  = _cfg.get("trails_on",  False)
            vision.network_on = _cfg.get("network_on", False)
            vision.predict_on = _cfg.get("predict_on", False)
            vision.heatmap_on = _cfg.get("heatmap_on", False)
            annotated = vision.apply(annotated, tracks, crowd_alarm=crowd_alarm)

            _draw_hud(annotated, len(tracks), len(obj_data), fps_avg, all_events, vision_mode)

            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])

            with _lock:
                state["fps"]         = round(fps_avg, 1)
                state["count"]       = len(tracks)
                state["obj_count"]   = len(obj_data)
                state["tracks"]      = track_data
                state["objects"]     = obj_data
                state["alerts"]      = all_events
                state["frame_jpg"]   = jpg.tobytes()
                state["game"]        = gstatus
                state["anomalies"]   = anomaly_events
                state["threats"]     = threat_events
                state["crowd_stats"] = crowd_stats
                state["prediction"]  = crowd_predictor.get_trend()
                state["risk"]        = crowd_predictor.get_risk_assessment()

    except BaseException as e:
        with _lock:
            state["error"] = f"Fatal Pipeline Error: {e}"
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        face_engine.clear_results()
        with _lock:
            state["running"]   = False
            state["frame_jpg"] = None


# ── Public API ─────────────────────────────────────────────────────────────────
def start(source=None):
    global _thread
    if source is None:
        source = CAMERA_SOURCE
    if _thread and _thread.is_alive():
        return False, "Already running"

    model_path = _cfg.get("model_path", MODEL_PATH)
    try:
        model = YOLO(model_path)
    except Exception as exc:
        return False, f"Failed to load model '{model_path}': {exc}"

    _cfg["source"] = source
    _stop_event.clear()
    _thread = threading.Thread(target=_run, args=(source, model, model_path), daemon=True)
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
        return True, f"Detection mode → {mode}"
    return False, f"Unknown mode: {mode}"


def set_vision(updates: dict):
    for k, v in updates.items():
        if k in {"network_on", "trails_on", "predict_on"}:
            # Keep visual link/trail/prediction overlays disabled.
            _cfg["network_on"] = False
            _cfg["trails_on"] = False
            _cfg["predict_on"] = False
            continue
        if k in _cfg:
            _cfg[k] = v
    if "voice_on" in updates:
        voice.set_enabled(updates["voice_on"])
    return True, "Vision config updated"


def set_model(path: str):
    """Switch YOLO model at runtime (takes effect on next frame)."""
    _cfg["model_path"] = path
    return True, f"Model path queued → {path}"


def get_state() -> dict:
    with _lock:
        return dict(state)
