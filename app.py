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

import database as db
import pipeline
import game
import face_engine
from config import SERVER_HOST, SERVER_PORT, CAMERA_SOURCE, FACES_DIR

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
        "running":   s["running"],
        "fps":       s["fps"],
        "count":     s["count"],
        "obj_count": s["obj_count"],
        "tracks":    s["tracks"],
        "alerts":    s["alerts"],
        "error":     s.get("error"),
        "mode":      pipeline._cfg.get("detection_mode", "people"),
        "vision":    pipeline._cfg.get("vision_mode", "normal"),
        "game":      s.get("game", {}),
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
        image : image file (jpg / png)
    """
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Missing 'name' field."}), 400

    file = request.files.get("image")
    if file is None:
        return jsonify({"ok": False, "message": "Missing 'image' file."}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "message": "Could not decode image."}), 400

    safe_name  = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    image_path = os.path.join(FACES_DIR, f"{safe_name}.jpg")
    cv2.imwrite(image_path, img)

    ok, msg = face_engine.register(name, img, image_path)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 422)


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


# ── Dashboard ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
