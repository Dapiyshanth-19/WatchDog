"""
WatchDog – Flask server (v2).
Run:  python app.py
Open: http://localhost:5000
"""

import time
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

import database as db
import pipeline
from config import SERVER_HOST, SERVER_PORT, CAMERA_SOURCE

app = Flask(__name__)
CORS(app)
db.init_db()


# ── Video stream ──────────────────────────────────────────────────────────────
def _placeholder_jpg() -> bytes:
    img = np.full((480, 640, 3), 25, np.uint8)
    cv2.putText(img, "WatchDog", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60,60,60), 2)
    cv2.putText(img, "Camera Offline", (195, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 1)
    cv2.putText(img, "Press Start to begin", (175, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60,60,60), 1)
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


# ── REST API ──────────────────────────────────────────────────────────────────
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
    # Convert numeric strings to int
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


@app.route("/counts/history")
def counts_history():
    """Return last 60 count log entries for the chart."""
    with db._conn() as c:
        rows = c.execute(
            "SELECT timestamp, count FROM counts ORDER BY id DESC LIMIT 60"
        ).fetchall()
    data = [{"t": r["timestamp"], "v": r["count"]} for r in reversed(rows)]
    return jsonify(data)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
