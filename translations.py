"""
WatchDog – UI and alert strings in English and Tamil.
"""

STRINGS = {
    # ── Activity labels ───────────────────────────────────────────────────────
    "activity_running": {
        "en": "Running fast",
        "ta": "வேகமாக ஓடுகிறார்",
    },
    "activity_loitering": {
        "en": "Standing still too long",
        "ta": "நீண்ட நேரம் நின்று கொண்டிருக்கிறார்",
    },
    "activity_walking": {
        "en": "Moving normally",
        "ta": "சாதாரணமாக நடக்கிறார்",
    },
    "activity_standing": {
        "en": "Standing",
        "ta": "நிற்கிறார்",
    },
    "activity_group": {
        "en": "Part of a group",
        "ta": "கூட்டத்தில் உள்ளார்",
    },

    # ── Alert messages ─────────────────────────────────────────────────────────
    "alert_running": {
        "en": "Person {id} is running!",
        "ta": "நபர் {id} ஓடுகிறார்!",
    },
    "alert_loitering": {
        "en": "Person {id} has been standing still for {sec}s",
        "ta": "நபர் {id} {sec} வினாடிகளாக நின்று கொண்டிருக்கிறார்",
    },
    "alert_crowd": {
        "en": "Crowd alert! {n} people detected (limit: {lim})",
        "ta": "கூட்ட எச்சரிக்கை! {n} நபர்கள் கண்டறியப்பட்டனர் (வரம்பு: {lim})",
    },
    "alert_object": {
        "en": "{label} detected",
        "ta": "{label} கண்டறியப்பட்டது",
    },

    # ── Object class names (English + Tamil) ───────────────────────────────────
    "class_person":     {"en": "Person",     "ta": "நபர்"},
    "class_bicycle":    {"en": "Bicycle",    "ta": "மிதிவண்டி"},
    "class_car":        {"en": "Car",        "ta": "கார்"},
    "class_motorcycle": {"en": "Motorcycle", "ta": "மோட்டார்சைக்கிள்"},
    "class_bus":        {"en": "Bus",        "ta": "பேருந்து"},
    "class_truck":      {"en": "Truck",      "ta": "லாரி"},
    "class_backpack":   {"en": "Backpack",   "ta": "பையை"},
    "class_handbag":    {"en": "Handbag",    "ta": "கைப்பை"},
    "class_suitcase":   {"en": "Suitcase",   "ta": "சூட்கேஸ்"},
    "class_bottle":     {"en": "Bottle",     "ta": "பாட்டில்"},
    "class_phone":      {"en": "Phone",      "ta": "தொலைபேசி"},
    "class_laptop":     {"en": "Laptop",     "ta": "லேப்டாப்"},
    "class_bag":        {"en": "Bag",        "ta": "பை"},
    "class_unknown":    {"en": "Object",     "ta": "பொருள்"},
}

# COCO class id → translation key
COCO_CLASS_MAP = {
    0:  "class_person",
    1:  "class_bicycle",
    2:  "class_car",
    3:  "class_motorcycle",
    5:  "class_bus",
    7:  "class_truck",
    24: "class_backpack",
    26: "class_handbag",
    28: "class_suitcase",
    39: "class_bottle",
    67: "class_phone",
    63: "class_laptop",
}

# Detection mode → COCO class ids (None = all classes)
DETECTION_MODES = {
    "people":   [0],
    "objects":  [24, 26, 28, 39, 67, 63],
    "vehicles": [1, 2, 3, 5, 7],
    "both":     [0, 24, 26, 28, 39, 67, 63],
    "all":      None,
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Return translated string, filling in any {placeholders}."""
    entry = STRINGS.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    try:
        return text.format(**kwargs)
    except KeyError:
        return text


def class_name(coco_id: int, lang: str = "en") -> str:
    key = COCO_CLASS_MAP.get(coco_id, "class_unknown")
    return t(key, lang)
