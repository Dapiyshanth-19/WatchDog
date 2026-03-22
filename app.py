"""
WatchDog – Flask web server & REST API.
Run:  python app.py
Then open http://localhost:5000 in your browser.
"""

import time
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

import database as db
import pipeline
from config import SERVER_HOST, SERVER_PORT, CAMERA_SOURCE

app = Flask(__name__)
CORS(app)

db.init_db()


# ── MJPEG video stream ────────────────────────────────────────────────────────
def _gen_frames():
    """Generator that yields MJPEG frames."""
    placeholder = _make_placeholder()
    while True:
        s = pipeline.get_state()
        jpg = s.get("frame_jpg")
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        else:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
        time.sleep(0.03)   # ~33 FPS ceiling for the HTTP stream


def _make_placeholder() -> bytes:
    """Return a small grey JPEG shown when the camera is off."""
    import cv2, numpy as np
    img = np.full((240, 320, 3), 40, np.uint8)
    cv2.putText(img, "Camera Offline", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    s = pipeline.get_state()
    return jsonify({
        "running": s["running"],
        "fps":     s["fps"],
        "count":   s["count"],
        "error":   s.get("error"),
    })


@app.route("/alerts")
def alerts():
    limit = int(request.args.get("limit", 20))
    return jsonify(db.get_alerts(limit))


@app.route("/alert", methods=["POST"])
def post_alert():
    data = request.get_json(force=True) or {}
    db.log_alert(
        camera_id=data.get("camera", 1),
        event_type=data.get("type", "Manual"),
        details=data.get("details"),
    )
    return jsonify({"ok": True})


@app.route("/cameras")
def cameras():
    return jsonify(db.get_cameras())


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(force=True) or {}
    source = data.get("source", CAMERA_SOURCE)
    ok, msg = pipeline.start(source)
    return jsonify({"started": ok, "message": msg})


@app.route("/stop", methods=["POST"])
def stop():
    ok, msg = pipeline.stop()
    return jsonify({"stopped": ok, "message": msg})


# ── Dashboard (served from templates/) ───────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
