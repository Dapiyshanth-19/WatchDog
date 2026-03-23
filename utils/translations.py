"""
WatchDog – UI and alert strings in English, Tamil, Hindi, Spanish, French, Arabic.
"""

# Supported languages
LANGUAGES = {
    "en": "English",
    "ta": "தமிழ்",
    "hi": "हिन्दी",
    "es": "Español",
    "fr": "Français",
    "ar": "العربية",
}

STRINGS = {
    # ── Activity labels ───────────────────────────────────────────────────────
    "activity_running": {
        "en": "Running fast",
        "ta": "வேகமாக ஓடுகிறார்",
        "hi": "तेज़ दौड़ रहा है",
        "es": "Corriendo rápido",
        "fr": "Court vite",
        "ar": "يركض بسرعة",
    },
    "activity_loitering": {
        "en": "Standing still too long",
        "ta": "நீண்ட நேரம் நின்று கொண்டிருக்கிறார்",
        "hi": "बहुत देर से खड़ा है",
        "es": "Parado demasiado tiempo",
        "fr": "Immobile trop longtemps",
        "ar": "واقف لفترة طويلة",
    },
    "activity_walking": {
        "en": "Moving normally",
        "ta": "சாதாரணமாக நடக்கிறார்",
        "hi": "सामान्य रूप से चल रहा है",
        "es": "Moviéndose normalmente",
        "fr": "Se déplace normalement",
        "ar": "يتحرك بشكل طبيعي",
    },
    "activity_standing": {
        "en": "Standing",
        "ta": "நிற்கிறார்",
        "hi": "खड़ा है",
        "es": "De pie",
        "fr": "Debout",
        "ar": "واقف",
    },
    "activity_sitting": {
        "en": "Sitting",
        "ta": "அமர்ந்திருக்கிறார்",
        "hi": "बैठा है",
        "es": "Sentado",
        "fr": "Assis",
        "ar": "جالس",
    },
    "activity_bending": {
        "en": "Bending down",
        "ta": "குனிந்திருக்கிறார்",
        "hi": "झुका हुआ है",
        "es": "Agachado",
        "fr": "Penché",
        "ar": "منحني",
    },
    "activity_group": {
        "en": "Part of a group",
        "ta": "கூட்டத்தில் உள்ளார்",
        "hi": "समूह का हिस्सा",
        "es": "Parte de un grupo",
        "fr": "Fait partie d'un groupe",
        "ar": "جزء من مجموعة",
    },

    # ── Alert messages ─────────────────────────────────────────────────────────
    "alert_running": {
        "en": "Person {id} is running!",
        "ta": "நபர் {id} ஓடுகிறார்!",
        "hi": "व्यक्ति {id} दौड़ रहा है!",
        "es": "¡Persona {id} está corriendo!",
        "fr": "Personne {id} court !",
        "ar": "!الشخص {id} يركض",
    },
    "alert_loitering": {
        "en": "Person {id} has been standing still for {sec}s",
        "ta": "நபர் {id} {sec} வினாடிகளாக நின்று கொண்டிருக்கிறார்",
        "hi": "व्यक्ति {id} {sec} सेकंड से खड़ा है",
        "es": "Persona {id} parada por {sec}s",
        "fr": "Personne {id} immobile depuis {sec}s",
        "ar": "الشخص {id} واقف منذ {sec} ثانية",
    },
    "alert_crowd": {
        "en": "Crowd alert! {n} people detected (limit: {lim})",
        "ta": "கூட்ட எச்சரிக்கை! {n} நபர்கள் கண்டறியப்பட்டனர் (வரம்பு: {lim})",
        "hi": "भीड़ चेतावनी! {n} लोग पाए गए (सीमा: {lim})",
        "es": "¡Alerta de multitud! {n} personas detectadas (límite: {lim})",
        "fr": "Alerte foule ! {n} personnes détectées (limite : {lim})",
        "ar": "تنبيه حشد! تم اكتشاف {n} أشخاص (الحد: {lim})",
    },
    "alert_object": {
        "en": "{label} detected",
        "ta": "{label} கண்டறியப்பட்டது",
        "hi": "{label} पाया गया",
        "es": "{label} detectado",
        "fr": "{label} détecté",
        "ar": "تم اكتشاف {label}",
    },

    # ── Threat alerts ─────────────────────────────────────────────────────────
    "alert_fall": {
        "en": "FALL DETECTED! Person may need help",
        "ta": "விழுந்தது கண்டறியப்பட்டது! நபருக்கு உதவி தேவைப்படலாம்",
        "hi": "गिरना पाया गया! व्यक्ति को मदद की जरूरत हो सकती है",
        "es": "¡CAÍDA DETECTADA! La persona puede necesitar ayuda",
        "fr": "CHUTE DÉTECTÉE ! La personne peut avoir besoin d'aide",
        "ar": "تم اكتشاف سقوط! قد يحتاج الشخص للمساعدة",
    },
    "alert_fire": {
        "en": "FIRE/FLAME DETECTED! Immediate attention required",
        "ta": "தீ/நெருப்பு கண்டறியப்பட்டது! உடனடி கவனம் தேவை",
        "hi": "आग/लौ पाई गई! तुरंत ध्यान दें",
        "es": "¡FUEGO/LLAMA DETECTADO! Atención inmediata requerida",
        "fr": "FEU/FLAMME DÉTECTÉ ! Attention immédiate requise",
        "ar": "تم اكتشاف حريق! يتطلب اهتماماً فورياً",
    },
    "alert_intrusion": {
        "en": "INTRUSION! Unauthorized entry detected",
        "ta": "ஊடுருவல்! அங்கீகரிக்கப்படாத நுழைவு கண்டறியப்பட்டது",
        "hi": "घुसपैठ! अनधिकृत प्रवेश पाया गया",
        "es": "¡INTRUSIÓN! Entrada no autorizada detectada",
        "fr": "INTRUSION ! Entrée non autorisée détectée",
        "ar": "اقتحام! تم اكتشاف دخول غير مصرح به",
    },
    "alert_stampede": {
        "en": "STAMPEDE ALERT! Dangerous crowd movement",
        "ta": "நெருக்கடி எச்சரிக்கை! ஆபத்தான கூட்ட நகர்வு",
        "hi": "भगदड़ चेतावनी! खतरनाक भीड़ आंदोलन",
        "es": "¡ALERTA DE ESTAMPIDA! Movimiento peligroso de multitud",
        "fr": "ALERTE BOUSCULADE ! Mouvement de foule dangereux",
        "ar": "تنبيه تدافع! حركة حشد خطيرة",
    },
    "alert_crush": {
        "en": "CRUSH RISK! Dangerously dense crowd",
        "ta": "நெரிசல் ஆபத்து! ஆபத்தான அடர்த்தியான கூட்டம்",
        "hi": "कुचलने का खतरा! खतरनाक रूप से घनी भीड़",
        "es": "¡RIESGO DE APLASTAMIENTO! Multitud peligrosamente densa",
        "fr": "RISQUE D'ÉCRASEMENT ! Foule dangereusement dense",
        "ar": "خطر السحق! حشد كثيف بشكل خطير",
    },

    # ── Anomaly alerts ────────────────────────────────────────────────────────
    "alert_dispersal": {
        "en": "CROWD DISPERSAL! People scattering rapidly",
        "ta": "கூட்டம் சிதறுகிறது! நபர்கள் விரைவாக சிதறுகிறார்கள்",
        "hi": "भीड़ बिखर रही है! लोग तेजी से भाग रहे हैं",
        "es": "¡DISPERSIÓN DE MULTITUD! Personas dispersándose rápidamente",
        "fr": "DISPERSION DE FOULE ! Personnes se dispersant rapidement",
        "ar": "تفرق الحشد! الناس يتفرقون بسرعة",
    },
    "alert_convergence": {
        "en": "CROWD CONVERGENCE! People rushing to one point",
        "ta": "கூட்ட குவிப்பு! நபர்கள் ஒரு இடத்தில் குவிகிறார்கள்",
        "hi": "भीड़ एकत्रीकरण! लोग एक जगह दौड़ रहे हैं",
        "es": "¡CONVERGENCIA DE MULTITUD! Personas corriendo a un punto",
        "fr": "CONVERGENCE DE FOULE ! Personnes se précipitant vers un point",
        "ar": "تجمع حشد! الناس يتدفقون إلى نقطة واحدة",
    },

    # ── Object class names (all 80 COCO classes) ──────────────────────────────
    "class_person":       {"en": "Person",       "ta": "நபர்"},
    "class_bicycle":      {"en": "Bicycle",      "ta": "மிதிவண்டி"},
    "class_car":          {"en": "Car",          "ta": "கார்"},
    "class_motorcycle":   {"en": "Motorcycle",   "ta": "மோட்டார்சைக்கிள்"},
    "class_airplane":     {"en": "Airplane",     "ta": "விமானம்"},
    "class_bus":          {"en": "Bus",          "ta": "பேருந்து"},
    "class_train":        {"en": "Train",        "ta": "ரயில்"},
    "class_truck":        {"en": "Truck",        "ta": "லாரி"},
    "class_boat":         {"en": "Boat",         "ta": "படகு"},
    "class_traffic_light":{"en": "Traffic Light", "ta": "சிக்னல் விளக்கு"},
    "class_fire_hydrant": {"en": "Fire Hydrant", "ta": "தீ குழாய்"},
    "class_stop_sign":    {"en": "Stop Sign",    "ta": "நிறுத்த அடையாளம்"},
    "class_parking_meter":{"en": "Parking Meter", "ta": "பார்க்கிங் மீட்டர்"},
    "class_bench":        {"en": "Bench",        "ta": "பெஞ்ச்"},
    "class_bird":         {"en": "Bird",         "ta": "பறவை"},
    "class_cat":          {"en": "Cat",          "ta": "பூனை"},
    "class_dog":          {"en": "Dog",          "ta": "நாய்"},
    "class_horse":        {"en": "Horse",        "ta": "குதிரை"},
    "class_sheep":        {"en": "Sheep",        "ta": "ஆடு"},
    "class_cow":          {"en": "Cow",          "ta": "மாடு"},
    "class_elephant":     {"en": "Elephant",     "ta": "யானை"},
    "class_bear":         {"en": "Bear",         "ta": "கரடி"},
    "class_zebra":        {"en": "Zebra",        "ta": "வரிக்குதிரை"},
    "class_giraffe":      {"en": "Giraffe",      "ta": "ஒட்டகச்சிவிங்கி"},
    "class_backpack":     {"en": "Backpack",     "ta": "முதுகுப்பை"},
    "class_umbrella":     {"en": "Umbrella",     "ta": "குடை"},
    "class_handbag":      {"en": "Handbag",      "ta": "கைப்பை"},
    "class_tie":          {"en": "Tie",          "ta": "டை"},
    "class_suitcase":     {"en": "Suitcase",     "ta": "சூட்கேஸ்"},
    "class_frisbee":      {"en": "Frisbee",      "ta": "ஃபிரிஸ்பி"},
    "class_skis":         {"en": "Skis",         "ta": "ஸ்கீ"},
    "class_snowboard":    {"en": "Snowboard",    "ta": "ஸ்னோபோர்டு"},
    "class_sports_ball":  {"en": "Ball",         "ta": "பந்து"},
    "class_kite":         {"en": "Kite",         "ta": "பட்டம்"},
    "class_baseball_bat": {"en": "Baseball Bat", "ta": "பேஸ்பால் மட்டை"},
    "class_baseball_glove":{"en":"Baseball Glove","ta": "பேஸ்பால் கையுறை"},
    "class_skateboard":   {"en": "Skateboard",   "ta": "ஸ்கேட்போர்டு"},
    "class_surfboard":    {"en": "Surfboard",    "ta": "சர்ஃப்போர்டு"},
    "class_tennis_racket":{"en": "Tennis Racket", "ta": "டென்னிஸ் ராக்கெட்"},
    "class_bottle":       {"en": "Bottle",       "ta": "பாட்டில்"},
    "class_wine_glass":   {"en": "Wine Glass",   "ta": "ஒயின் கிளாஸ்"},
    "class_cup":          {"en": "Cup",          "ta": "கோப்பை"},
    "class_fork":         {"en": "Fork",         "ta": "முள்கரண்டி"},
    "class_knife":        {"en": "Knife",        "ta": "கத்தி"},
    "class_spoon":        {"en": "Spoon",        "ta": "கரண்டி"},
    "class_bowl":         {"en": "Bowl",         "ta": "கிண்ணம்"},
    "class_banana":       {"en": "Banana",       "ta": "வாழைப்பழம்"},
    "class_apple":        {"en": "Apple",        "ta": "ஆப்பிள்"},
    "class_sandwich":     {"en": "Sandwich",     "ta": "சாண்ட்விச்"},
    "class_orange":       {"en": "Orange",       "ta": "ஆரஞ்சு"},
    "class_broccoli":     {"en": "Broccoli",     "ta": "ப்ரோக்கோலி"},
    "class_carrot":       {"en": "Carrot",       "ta": "கேரட்"},
    "class_hot_dog":      {"en": "Hot Dog",      "ta": "ஹாட் டாக்"},
    "class_pizza":        {"en": "Pizza",        "ta": "பீட்சா"},
    "class_donut":        {"en": "Donut",        "ta": "டோனட்"},
    "class_cake":         {"en": "Cake",         "ta": "கேக்"},
    "class_chair":        {"en": "Chair",        "ta": "நாற்காலி"},
    "class_couch":        {"en": "Couch",        "ta": "சோபா"},
    "class_potted_plant": {"en": "Plant",        "ta": "செடி"},
    "class_bed":          {"en": "Bed",          "ta": "படுக்கை"},
    "class_dining_table": {"en": "Table",        "ta": "மேசை"},
    "class_toilet":       {"en": "Toilet",       "ta": "கழிப்பறை"},
    "class_tv":           {"en": "TV/Monitor",   "ta": "டிவி"},
    "class_laptop":       {"en": "Laptop",       "ta": "லேப்டாப்"},
    "class_mouse":        {"en": "Mouse",        "ta": "மவுஸ்"},
    "class_remote":       {"en": "Remote",       "ta": "ரிமோட்"},
    "class_keyboard":     {"en": "Keyboard",     "ta": "கீபோர்டு"},
    "class_phone":        {"en": "Phone",        "ta": "தொலைபேசி"},
    "class_microwave":    {"en": "Microwave",    "ta": "மைக்ரோவேவ்"},
    "class_oven":         {"en": "Oven",         "ta": "அடுப்பு"},
    "class_toaster":      {"en": "Toaster",      "ta": "டோஸ்டர்"},
    "class_sink":         {"en": "Sink",         "ta": "சிங்க்"},
    "class_refrigerator": {"en": "Refrigerator", "ta": "குளிர்சாதனம்"},
    "class_book":         {"en": "Book",         "ta": "புத்தகம்"},
    "class_clock":        {"en": "Clock",        "ta": "கடிகாரம்"},
    "class_vase":         {"en": "Vase",         "ta": "குடுவை"},
    "class_scissors":     {"en": "Scissors",     "ta": "கத்தரிக்கோல்"},
    "class_teddy_bear":   {"en": "Teddy Bear",   "ta": "டெடி பியர்"},
    "class_hair_drier":   {"en": "Hair Dryer",   "ta": "ஹேர் டிரையர்"},
    "class_toothbrush":   {"en": "Toothbrush",   "ta": "பல் தூரிகை"},
    "class_unknown":      {"en": "Object",       "ta": "பொருள்"},
}

# ── COCO class id -> translation key (all 80 classes) ─────────────────────────
COCO_CLASS_MAP = {
    0:  "class_person",        1:  "class_bicycle",       2:  "class_car",
    3:  "class_motorcycle",    4:  "class_airplane",      5:  "class_bus",
    6:  "class_train",         7:  "class_truck",         8:  "class_boat",
    9:  "class_traffic_light", 10: "class_fire_hydrant",  11: "class_stop_sign",
    12: "class_parking_meter", 13: "class_bench",         14: "class_bird",
    15: "class_cat",           16: "class_dog",           17: "class_horse",
    18: "class_sheep",         19: "class_cow",           20: "class_elephant",
    21: "class_bear",          22: "class_zebra",         23: "class_giraffe",
    24: "class_backpack",      25: "class_umbrella",      26: "class_handbag",
    27: "class_tie",           28: "class_suitcase",      29: "class_frisbee",
    30: "class_skis",          31: "class_snowboard",     32: "class_sports_ball",
    33: "class_kite",          34: "class_baseball_bat",  35: "class_baseball_glove",
    36: "class_skateboard",    37: "class_surfboard",     38: "class_tennis_racket",
    39: "class_bottle",        40: "class_wine_glass",    41: "class_cup",
    42: "class_fork",          43: "class_knife",         44: "class_spoon",
    45: "class_bowl",          46: "class_banana",        47: "class_apple",
    48: "class_sandwich",      49: "class_orange",        50: "class_broccoli",
    51: "class_carrot",        52: "class_hot_dog",       53: "class_pizza",
    54: "class_donut",         55: "class_cake",          56: "class_chair",
    57: "class_couch",         58: "class_potted_plant",  59: "class_bed",
    60: "class_dining_table",  61: "class_toilet",        62: "class_tv",
    63: "class_laptop",        64: "class_mouse",         65: "class_remote",
    66: "class_keyboard",      67: "class_phone",         68: "class_microwave",
    69: "class_oven",          70: "class_toaster",       71: "class_sink",
    72: "class_refrigerator",  73: "class_book",          74: "class_clock",
    75: "class_vase",          76: "class_scissors",      77: "class_teddy_bear",
    78: "class_hair_drier",    79: "class_toothbrush",
}

# Detection mode -> COCO class ids (None = all classes)
DETECTION_MODES = {
    "people":   [0],
    "objects":  None,     # detect ALL object classes
    "vehicles": [1, 2, 3, 4, 5, 6, 7, 8],
    "both":     None,     # people + all objects
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
