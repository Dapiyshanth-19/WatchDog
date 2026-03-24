# -*- coding: utf-8 -*-
"""
WatchDog – Flask server (v5 – Full Game + Face Edition).
"""

import os
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

from core import database as db
from core import pipeline
from core.multi_camera import manager as multi_cam
from engine import game
from engine import face_engine
from core.config import SERVER_HOST, SERVER_PORT, CAMERA_SOURCE, FACES_DIR

app = Flask(__name__)
CORS(app)
db.init_db()

os.makedirs(FACES_DIR, exist_ok=True)


# ── Video stream ───────────────────────────────────────────────────────────────
def _placeholder_jpg() -> bytes:
    img = np.full((480, 640, 3), 18, np.uint8)
    cv2.putText(img, "WatchDog", (205, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (45, 45, 45), 2)
    cv2.putText(img, "PHANTOM VISION", (170, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 120), 2)
    cv2.putText(img, "Press Start to begin", (178, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


_PLACEHOLDER = None


def _gen_frames():
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        _PLACEHOLDER = _placeholder_jpg()
    while True:
        s   = pipeline.get_state()
        jpg = s.get("frame_jpg") or _PLACEHOLDER
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Core REST API ──────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    s = pipeline.get_state()
    return jsonify({
        "running":     s["running"],
        "fps":         s["fps"],
        "count":       s["count"],
        "obj_count":   s["obj_count"],
        "tracks":      s["tracks"],
        "objects":     s.get("objects", []),
        "alerts":      s["alerts"],
        "error":       s.get("error"),
        "mode":        pipeline._cfg.get("detection_mode", "people"),
        "vision":      pipeline._cfg.get("vision_mode", "normal"),
        "game":        s.get("game", {}),
        "anomalies":   s.get("anomalies", []),
        "threats":     s.get("threats", []),
        "crowd_stats": s.get("crowd_stats", {}),
        "prediction":  s.get("prediction", {}),
        "risk":        s.get("risk", {}),
    })


@app.route("/alerts")
def alerts():
    limit = int(request.args.get("limit", 30))
    return jsonify(db.get_alerts(limit))


@app.route("/alert", methods=["POST"])
def post_alert():
    data = request.get_json(force=True) or {}
    db.log_alert(
        camera_id=data.get("camera", 1),
        event_type=data.get("type", "Manual"),
        details={"details": data.get("details", ""), "details_ta": data.get("details_ta", "")},
    )
    return jsonify({"ok": True})


@app.route("/cameras")
def cameras():
    return jsonify(db.get_cameras())


@app.route("/start", methods=["POST"])
def start():
    data   = request.get_json(force=True) or {}
    source = data.get("source", CAMERA_SOURCE)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    ok, msg = pipeline.start(source)
    return jsonify({"started": ok, "message": msg})

@app.route("/test_sync")
def test_sync():
    print("DEBUG: test_sync called", flush=True)
    from ultralytics import YOLO
    model_path = pipeline._cfg.get("model_path", "models/yolo26n.pt")
    print(f"DEBUG: pre-loading model {model_path}", flush=True)
    model = YOLO(model_path)
    print("DEBUG: model loaded, starting pipeline._run sync", flush=True)
    try:
        pipeline._run(0, model)
    except Exception as e:
        print(f"DEBUG ERROR inside test_sync: {e}", flush=True)
    return "Done"


@app.route("/stop", methods=["POST"])
def stop():
    ok, msg = pipeline.stop()
    return jsonify({"stopped": ok, "message": msg})


@app.route("/config", methods=["POST"])
def config():
    data = request.get_json(force=True) or {}
    mode = data.get("detection_mode")
    if mode:
        ok, msg = pipeline.set_detection_mode(mode)
        return jsonify({"ok": ok, "message": msg})
    return jsonify({"ok": False, "message": "Nothing to configure"})


@app.route("/vision", methods=["POST"])
def vision():
    data = request.get_json(force=True) or {}
    ok, msg = pipeline.set_vision(data)
    return jsonify({"ok": ok, "message": msg})


@app.route("/counts/history")
def counts_history():
    with db._conn() as c:
        rows = c.execute(
            "SELECT timestamp, count FROM counts ORDER BY id DESC LIMIT 60"
        ).fetchall()
    return jsonify([{"t": r["timestamp"], "v": r["count"]} for r in reversed(rows)])


# ── Model switching ────────────────────────────────────────────────────────────
@app.route("/model", methods=["POST"])
def switch_model():
    data = request.get_json(force=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"ok": False, "message": "Missing 'path' field."})
    ok, msg = pipeline.set_model(path)
    return jsonify({"ok": ok, "message": msg})


# ── Game endpoints ─────────────────────────────────────────────────────────────
@app.route("/game/configure", methods=["POST"])
def game_configure():
    """
    Configure all game parameters before starting.

    JSON body:
    {
        "total_duration":    120,   // seconds (0 = unlimited)
        "movement_duration": 10,    // green-light phase length (seconds)
        "freeze_duration":   5,     // red-light phase length   (seconds)
        "target_name":       "Alice",  // registered face name (optional)
        "win_zone_y":        380,   // finish-line Y pixel (0 = disabled)
        "win_zone_x1":       0,     // left bound  (0 = full width)
        "win_zone_x2":       0      // right bound (0 = full width)
    }
    """
    data = request.get_json(force=True) or {}
    game.configure(
        total_duration    = float(data.get("total_duration",    0)),
        movement_duration = float(data.get("movement_duration", 10)),
        freeze_duration   = float(data.get("freeze_duration",   5)),
        target_name       = str(data.get("target_name",         "")),
        win_zone_y        = int(data.get("win_zone_y",          0)),
        win_zone_x1       = int(data.get("win_zone_x1",         0)),
        win_zone_x2       = int(data.get("win_zone_x2",         0)),
    )
    return jsonify({"ok": True, "config": game.get_config()})


@app.route("/game/start", methods=["POST"])
def game_start():
    """Start (or restart) the game with the currently configured settings."""
    game.start_game()
    return jsonify({"ok": True, "message": "Game started.", "status": game.get_status()})


@app.route("/game/freeze", methods=["POST"])
def game_freeze():
    game.set_freeze(True)
    return jsonify({"ok": True, "message": "Freeze activated.", "status": game.get_status()})


@app.route("/game/unfreeze", methods=["POST"])
def game_unfreeze():
    game.set_freeze(False)
    return jsonify({"ok": True, "message": "Freeze deactivated.", "status": game.get_status()})


@app.route("/game/reset", methods=["POST"])
def game_reset():
    game.reset()
    return jsonify({"ok": True, "message": "Game reset.", "status": game.get_status()})


@app.route("/game/status")
def game_status_route():
    return jsonify(game.get_status())


@app.route("/game/config")
def game_config_route():
    return jsonify(game.get_config())


@app.route("/game/winners")
def game_winners():
    return jsonify(game.winners)


@app.route("/game/set_zone", methods=["POST"])
def game_set_zone():
    """
    Set the finish-line Y position.
    JSON: { "win_zone_y": 380, "win_zone_x1": 0, "win_zone_x2": 0 }
    """
    data = request.get_json(force=True) or {}
    cfg  = game.get_config()
    game.configure(
        total_duration    = cfg["total_duration"],
        movement_duration = cfg["movement_duration"],
        freeze_duration   = cfg["freeze_duration"],
        target_name       = cfg["target_name"],
        win_zone_y        = int(data.get("win_zone_y",  cfg["win_zone_y"])),
        win_zone_x1       = int(data.get("win_zone_x1", cfg["win_zone_x1"])),
        win_zone_x2       = int(data.get("win_zone_x2", cfg["win_zone_x2"])),
    )
    return jsonify({"ok": True, "win_zone_y": game.get_config()["win_zone_y"]})


# ── Face registration endpoints ────────────────────────────────────────────────
@app.route("/face/register", methods=["POST"])
def face_register():
    """
    Register a face.  Multipart form-data:
        name  : person's display name
        image : image file(s) (jpg / png) - supports multiple
    """
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Missing 'name' field."}), 400

    files = request.files.getlist("image")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"ok": False, "message": "Missing 'image' file."}), 400

    registered_count = 0
    
    for i, file in enumerate(files):
        if file.filename == "":
            continue
        # Read file bytes
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue

        safe_name  = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        # Unique timestamp for each file to avoid overwrite
        image_path = os.path.join(FACES_DIR, f"{safe_name}_{int(time.time())}_{i}.jpg")
        cv2.imwrite(image_path, img)

        ok, msg = face_engine.register(name, img, image_path)
        if ok:
            registered_count += 1

    if registered_count == 0:
        return jsonify({"ok": False, "message": "Could not register face from provided images."}), 422

    return jsonify({"ok": True, "message": f"Registered '{name}' successfully with {registered_count} photos."}), 200


@app.route("/face/users")
def face_users():
    users = db.get_users()
    return jsonify([{"id": u["id"], "name": u["name"], "image_path": u["image_path"]}
                    for u in users])


@app.route("/face/delete", methods=["POST"])
def face_delete():
    data = request.get_json(force=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Missing 'name'."}), 400
    deleted = db.delete_user(name)
    face_engine.clear_results()
    return jsonify({"ok": deleted,
                    "message": f"Deleted '{name}'." if deleted else "User not found."})


# ── AI Analytics endpoints ─────────────────────────────────────────────────────
@app.route("/analytics/crowd")
def analytics_crowd():
    """Real-time crowd statistics from anomaly detector."""
    s = pipeline.get_state()
    return jsonify({
        "crowd_stats": s.get("crowd_stats", {}),
        "anomalies":   s.get("anomalies", []),
    })


@app.route("/analytics/prediction")
def analytics_prediction():
    """Crowd trend prediction and forecasting."""
    s = pipeline.get_state()
    pred = s.get("prediction", {})
    risk = s.get("risk", {})
    hourly = []
    if pipeline.crowd_predictor:
        hourly = pipeline.crowd_predictor.get_hourly_forecast()
    return jsonify({
        "prediction":      pred,
        "risk":            risk,
        "hourly_forecast": hourly,
    })


@app.route("/analytics/threats")
def analytics_threats():
    """Current threat events and zone configuration."""
    s = pipeline.get_state()
    zones = []
    if pipeline.threat_detector:
        zones = pipeline.threat_detector.get_zones()
    return jsonify({
        "threats": s.get("threats", []),
        "zones":   zones,
    })


@app.route("/analytics/full")
def analytics_full():
    """All analytics in one call — for dashboard polling."""
    s = pipeline.get_state()
    hourly = []
    zones = []
    if pipeline.crowd_predictor:
        hourly = pipeline.crowd_predictor.get_hourly_forecast()
    if pipeline.threat_detector:
        zones = pipeline.threat_detector.get_zones()
    return jsonify({
        "crowd_stats":     s.get("crowd_stats", {}),
        "anomalies":       s.get("anomalies", []),
        "threats":         s.get("threats", []),
        "prediction":      s.get("prediction", {}),
        "risk":            s.get("risk", {}),
        "hourly_forecast": hourly,
        "zones":           zones,
    })


# ── Threat zone management ────────────────────────────────────────────────────
@app.route("/zones", methods=["GET"])
def get_zones():
    if pipeline.threat_detector:
        return jsonify(pipeline.threat_detector.get_zones())
    return jsonify([])


@app.route("/zones", methods=["POST"])
def add_zone():
    """Add a restricted zone. JSON: {name, x1, y1, x2, y2, type?}"""
    data = request.get_json(force=True) or {}
    if not pipeline.threat_detector:
        return jsonify({"ok": False, "message": "Pipeline not running."}), 400
    name = data.get("name", "Zone")
    pipeline.threat_detector.add_zone(
        name=name,
        x1=int(data.get("x1", 0)),
        y1=int(data.get("y1", 0)),
        x2=int(data.get("x2", 200)),
        y2=int(data.get("y2", 200)),
        zone_type=data.get("type", "restricted"),
    )
    return jsonify({"ok": True, "zones": pipeline.threat_detector.get_zones()})


@app.route("/zones/clear", methods=["POST"])
def clear_zones():
    if pipeline.threat_detector:
        pipeline.threat_detector.clear_zones()
    return jsonify({"ok": True})


# ── Multi-camera endpoints ────────────────────────────────────────────────────
@app.route("/multicam/add", methods=["POST"])
def multicam_add():
    """Add a camera. JSON: {name, source}"""
    data = request.get_json(force=True) or {}
    name = data.get("name", "Camera")
    source = data.get("source", "0")
    cam_id = multi_cam.add_camera(name, source)
    return jsonify({"ok": True, "cam_id": cam_id})


@app.route("/multicam/remove", methods=["POST"])
def multicam_remove():
    data = request.get_json(force=True) or {}
    ok = multi_cam.remove_camera(int(data.get("cam_id", 0)))
    return jsonify({"ok": ok})


@app.route("/multicam/start", methods=["POST"])
def multicam_start():
    data = request.get_json(force=True) or {}
    cam_id = data.get("cam_id")
    if cam_id is not None:
        ok = multi_cam.start_camera(int(cam_id))
    else:
        multi_cam.start_all()
        ok = True
    return jsonify({"ok": ok})


@app.route("/multicam/stop", methods=["POST"])
def multicam_stop():
    data = request.get_json(force=True) or {}
    cam_id = data.get("cam_id")
    if cam_id is not None:
        ok = multi_cam.stop_camera(int(cam_id))
    else:
        multi_cam.stop_all()
        ok = True
    return jsonify({"ok": ok})


@app.route("/multicam/status")
def multicam_status():
    return jsonify(multi_cam.get_aggregate_stats())


@app.route("/multicam/feed/<int:cam_id>")
def multicam_feed(cam_id):
    """MJPEG stream for a specific camera."""
    def gen():
        while True:
            jpg = multi_cam.get_camera_frame(cam_id)
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/multicam/set_model", methods=["POST"])
def multicam_set_model():
    """Set shared YOLO model for multi-camera. JSON: {model_path}"""
    data = request.get_json(force=True) or {}
    path = data.get("model_path", "").strip()
    if not path:
        return jsonify({"ok": False, "message": "Missing model_path."})
    from ultralytics import YOLO
    try:
        model = YOLO(path)
        multi_cam.set_model(model)
        return jsonify({"ok": True, "message": f"Model set: {path}"})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)})


# ── Dashboard ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
