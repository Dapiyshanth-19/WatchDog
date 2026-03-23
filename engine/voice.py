# -*- coding: utf-8 -*-
"""
WatchDog – Offline AI voice alert engine.
Uses pyttsx3 (no internet needed).  Tamil voice requires a Tamil TTS voice
installed on the OS (e.g. Windows Tamil language pack).
Falls back to English if Tamil voice not found.
"""

import threading
import queue
import time

_q:      queue.Queue       = queue.Queue()
_thread: threading.Thread | None = None
_engine                    = None
_ready                     = False
_enabled                   = False   # toggled by user


def _worker():
    global _engine, _ready
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 145)
        _engine.setProperty("volume", 0.95)
        _ready = True
    except Exception as exc:
        print(f"[voice] pyttsx3 not available: {exc}")
        return

    while True:
        item = _q.get()
        if item is None:
            break
        text, lang = item
        if not _enabled:
            continue
        try:
            # Try to select a Tamil voice when lang=="ta"
            if lang == "ta":
                voices = _engine.getProperty("voices")
                tamil_voice = next(
                    (v for v in voices if "tamil" in v.name.lower() or
                                          "ta-IN" in (v.languages[0] if v.languages else "")),
                    None,
                )
                if tamil_voice:
                    _engine.setProperty("voice", tamil_voice.id)
                else:
                    # No Tamil voice — speak English fallback
                    pass
            else:
                voices = _engine.getProperty("voices")
                en_voice = next(
                    (v for v in voices if "english" in v.name.lower() or
                                          "en" in (v.languages[0] if v.languages else "")),
                    None,
                )
                if en_voice:
                    _engine.setProperty("voice", en_voice.id)

            _engine.say(text)
            _engine.runAndWait()
        except Exception as exc:
            print(f"[voice] speak error: {exc}")


def start():
    global _thread
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_worker, daemon=True)
    _thread.start()


def speak(text: str, lang: str = "en"):
    """Queue a phrase to be spoken (non-blocking)."""
    if not _ready or not _enabled:
        return
    try:
        _q.put_nowait((text, lang))
    except queue.Full:
        pass


def set_enabled(on: bool):
    global _enabled
    _enabled = on


def is_ready() -> bool:
    return _ready
