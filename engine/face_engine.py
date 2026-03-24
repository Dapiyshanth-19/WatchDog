# -*- coding: utf-8 -*-
"""
WatchDog – Face recognition add-on.

Runs ALONGSIDE YOLO/SORT – does NOT replace them.
Identifies tracked people by comparing face crops against stored embeddings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP – install ONE of these backends:

  Option A (recommended):
      pip install deepface tf-keras

  Option B:
      pip install face-recognition dlib
      (requires CMake + Visual C++ Build Tools on Windows)

If neither is installed, face recognition is silently disabled –
the rest of WatchDog continues to work normally.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Threading model
───────────────
A single daemon thread (face-engine) processes frames from a queue.
The main pipeline calls submit_frame() every N frames (non-blocking).
Results are written to a shared dict and read back by the pipeline on
the next frame via get_results().  The pipeline never waits.
"""

import json
import os
import queue
import threading

import cv2
import numpy as np

from core.config import FACE_RECOGNITION_INTERVAL, FACE_SIMILARITY_THRESHOLD
from core import database as db


# ── Backend auto-detection ─────────────────────────────────────────────────────
_BACKEND: str | None = None
_fr_lib = None

try:
    from deepface import DeepFace        # noqa: F401 – imported for use below
    _BACKEND = "deepface"
except ImportError:
    pass

if _BACKEND is None:
    try:
        import face_recognition as _fr_lib  # noqa: F811
        _BACKEND = "face_recognition"
    except ImportError:
        pass


# ── Shared state ───────────────────────────────────────────────────────────────
_results: dict[int, str] = {}          # track_id → recognised name
_results_lock = threading.Lock()

_task_queue: queue.Queue = queue.Queue(maxsize=2)
_frame_counter = 0


# ── Embedding helpers ──────────────────────────────────────────────────────────
def _extract_embedding(crop: np.ndarray) -> np.ndarray | None:
    """Return face embedding array from a BGR image crop, or None on failure."""
    if _BACKEND == "deepface":
        try:
            res = DeepFace.represent(
                crop,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="ssd",
            )
            if res:
                return np.array(res[0]["embedding"])
        except Exception:
            return None

    elif _BACKEND == "face_recognition" and _fr_lib is not None:
        try:
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = _fr_lib.face_encodings(rgb)
            if encs:
                return np.array(encs[0])
        except Exception:
            return None

    return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _match_score(emb: np.ndarray, known: np.ndarray) -> float:
    """
    Return similarity in [0, 1] – higher = better match.
    • DeepFace / Facenet  → cosine similarity
    • face_recognition    → 1 − normalised euclidean distance
    """
    if len(emb) != len(known):
        return 0.0
    if _BACKEND == "deepface":
        return _cosine_sim(emb, known)
    # face_recognition uses euclidean; tolerance ≈ 0.6
    dist = float(np.linalg.norm(emb - known))
    return max(0.0, 1.0 - dist / 1.2)


# ── Worker thread ──────────────────────────────────────────────────────────────
def _do_recognise(frame: np.ndarray, tracks: np.ndarray) -> dict[int, str]:
    """Blocking recognition – runs inside the worker thread only."""
    users = db.get_users()
    if not users:
        return {}

    known_embeddings: list[np.ndarray] = []
    known_names:      list[str]        = []
    for u in users:
        try:
            emb_data = json.loads(u["embedding"])
            # Handle legacy 1D array
            if emb_data and isinstance(emb_data[0], float):
                embs = [np.array(emb_data)]
            else:
                embs = [np.array(e) for e in emb_data]
            
            for emb in embs:
                known_embeddings.append(emb)
                known_names.append(u["name"])
        except Exception:
            continue

    if not known_embeddings:
        return {}

    h, w   = frame.shape[:2]
    output: dict[int, str] = {}

    for trk in tracks:
        x1, y1, x2, y2, tid = trk
        tid = int(tid)
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop = frame[cy1:cy2, cx1:cx2]

        # Skip crops that are too small to contain a recognisable face
        if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 25:
            continue

        emb = _extract_embedding(crop)
        if emb is None:
            continue

        best_score = 0.0
        best_name  = "Unknown"
        for known_emb, name in zip(known_embeddings, known_names):
            score = _match_score(emb, known_emb)
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score >= FACE_SIMILARITY_THRESHOLD:
            output[tid] = best_name

    return output


def _worker():
    while True:
        try:
            frame, tracks = _task_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            recognised = _do_recognise(frame, tracks)
            with _results_lock:
                _results.update(recognised)
        except Exception:
            pass
        finally:
            _task_queue.task_done()


# Start daemon worker at import time – safe even if no backend is installed
_worker_thread = threading.Thread(target=_worker, daemon=True, name="face-engine")
_worker_thread.start()


# ── Public API ─────────────────────────────────────────────────────────────────
def is_available() -> bool:
    """True if a face recognition backend is installed."""
    return _BACKEND is not None


def register(name: str, image: np.ndarray, image_path: str = "") -> tuple[bool, str]:
    """
    Detect a face in image and save its embedding to the database.

    Parameters
    ----------
    name       : display name for this person
    image      : BGR numpy image (from cv2.imread or decoded upload)
    image_path : path where the image is already saved (optional, for reference)

    Returns
    -------
    (success: bool, message: str)
    """
    if not is_available():
        return False, (
            "No face recognition backend installed. "
            "Run:  pip install deepface tf-keras  "
            "  or  pip install face-recognition dlib"
        )

    emb = _extract_embedding(image)
    if emb is None:
        return False, "No face detected in the provided image. Use a clear, front-facing photo."

    db.save_user(name, emb.tolist(), image_path)
    return True, f"Registered '{name}' successfully."


def submit_frame(frame: np.ndarray, tracks: np.ndarray):
    """
    Submit a frame for asynchronous face recognition (non-blocking).

    • Only submits every FACE_RECOGNITION_INTERVAL frames.
    • Silently drops submission if the queue is full (previous frame
      still processing) to avoid any pipeline slowdown.
    """
    if not is_available():
        return

    global _frame_counter
    _frame_counter += 1
    if _frame_counter % FACE_RECOGNITION_INTERVAL != 0:
        return
    if len(tracks) == 0:
        return

    try:
        _task_queue.put_nowait((frame.copy(), tracks.copy()))
    except queue.Full:
        pass  # previous frame still in progress – skip this one


def get_results() -> dict[int, str]:
    """Return the latest recognition results (non-blocking, thread-safe)."""
    with _results_lock:
        return dict(_results)


def clear_results():
    """Flush cached results – call on pipeline stop."""
    with _results_lock:
        _results.clear()
    global _frame_counter
    _frame_counter = 0
