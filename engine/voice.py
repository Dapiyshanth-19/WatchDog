# -*- coding: utf-8 -*-
"""
WatchDog – Offline AI voice alert engine.
Uses pyttsx3 (no internet needed).  Tamil voice requires a Tamil TTS voice
installed on the OS (e.g. Windows Tamil language pack).
Falls back to English if Tamil voice not found.
"""

import threading
import queue

_q:      queue.Queue       = queue.Queue(maxsize=128)
_thread: threading.Thread | None = None
_engine                    = None
_ready                     = False
_enabled                   = False   # toggled by user
_rate                      = 145
_volume                    = 0.95
_q_lock                    = threading.Lock()


def _set_voice_for_lang(lang: str):
    if _engine is None:
        return
    if lang == "ta":
        voices = _engine.getProperty("voices")
        tamil_voice = next(
            (v for v in voices if "tamil" in v.name.lower() or
                                "ta" in str(v.languages).lower()),
            None,
        )
        if tamil_voice:
            _engine.setProperty("voice", tamil_voice.id)
    else:
        voices = _engine.getProperty("voices")
        en_voice = next(
            (v for v in voices if "english" in v.name.lower() or
                                "en" in str(v.languages).lower()),
            None,
        )
        if en_voice:
            _engine.setProperty("voice", en_voice.id)


def _drop_oldest_items(keep_last: int = 40):
    """Trim queue to avoid long-delayed audio on long runs."""
    with _q_lock:
        while _q.qsize() > keep_last:
            try:
                _q.get_nowait()
            except queue.Empty:
                break


def _enqueue(text: str, lang: str, prefer_latest: bool = False):
    if prefer_latest:
        _drop_oldest_items(keep_last=20)
    try:
        _q.put_nowait((text, lang))
    except queue.Full:
        # Drop one old message and retry so new live speech can continue.
        try:
            _q.get_nowait()
        except queue.Empty:
            pass
        try:
            _q.put_nowait((text, lang))
        except queue.Full:
            pass


def _worker():
    global _engine, _ready
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", _rate)
        _engine.setProperty("volume", _volume)
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
            _engine.setProperty("rate", _rate)
            _engine.setProperty("volume", _volume)
            _set_voice_for_lang(lang)
            _engine.say(text)
            _engine.runAndWait()
        except Exception as exc:
            print(f"[voice] speak error: {exc}")
            # Try to recover engine for long-running sessions.
            try:
                import pyttsx3
                _engine = pyttsx3.init()
                _engine.setProperty("rate", _rate)
                _engine.setProperty("volume", _volume)
            except Exception as rexc:
                print(f"[voice] recovery failed: {rexc}")


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
    text = (text or "").strip()
    if not text:
        return
    _enqueue(text, lang, prefer_latest=False)


def speak_live(text: str, lang: str = "en"):
    """
    Queue live narration with queue trimming so continuous narration does not lag.
    Keeps latest speech responsive even when many alerts are produced.
    """
    if not _ready or not _enabled:
        return
    text = (text or "").strip()
    if not text:
        return

    _enqueue(text, lang, prefer_latest=True)


def set_enabled(on: bool):
    global _enabled
    _enabled = on
    if not on:
        # Flush pending queue when user turns voice off.
        with _q_lock:
            while True:
                try:
                    _q.get_nowait()
                except queue.Empty:
                    break


def set_rate(rate: int):
    global _rate
    _rate = max(70, min(int(rate), 300))


def set_volume(volume: float):
    global _volume
    _volume = max(0.0, min(float(volume), 1.0))


def is_ready() -> bool:
    return _ready
