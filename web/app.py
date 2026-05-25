"""
Sign Language Translator — Reliable detection edition
"""
import os, json, base64, pickle, sys, time
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from gtts import gTTS
from string import ascii_lowercase
import mediapipe as mp
from collections import deque, Counter

BASE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(BASE)
AUDIO_DIR  = os.path.join(ROOT, "audio")
USERS_FILE = os.path.join(BASE, "users.json")
TMP_DIR    = os.path.join(BASE, "static", "tmp")
LM_MODEL   = os.path.join(ROOT, "landmark_model.keras")
ENCODER    = os.path.join(BASE, "label_encoder.pkl")
SCALER     = os.path.join(BASE, "scaler.pkl")
RF_MODEL   = os.path.join(BASE, "rf_model.pkl")
os.makedirs(TMP_DIR, exist_ok=True)

sys.path.insert(0, ROOT)

# ── MediaPipe ──────────────────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request

LANDMARKER = os.path.join(BASE, "hand_landmarker.task")
if not os.path.exists(LANDMARKER):
    print("Downloading hand landmarker...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        LANDMARKER)

_hand_opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,   # lower = detects more
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3)
hand_detector = mp_vision.HandLandmarker.create_from_options(_hand_opts)

def detect_hands(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res    = hand_detector.detect(mp_img)
    return res.hand_landmarks[0] if res.hand_landmarks else None

LETTERS = {ch: str(i) for i, ch in enumerate(ascii_lowercase, start=1)}

# ── Load models ────────────────────────────────────────────────────────────────
try:
    from feature_extractor import extract_features
    _fe_ok = True
except ImportError:
    _fe_ok = False

# MLP-only mode (works without rf_model.pkl)
USE_MLP = _fe_ok and os.path.exists(LM_MODEL) and os.path.exists(ENCODER) and os.path.exists(SCALER)
USE_RF  = USE_MLP and os.path.exists(RF_MODEL)

if USE_MLP:
    print("Loading MLP model...")
    mlp_model = tf.keras.models.load_model(LM_MODEL)
    with open(ENCODER, "rb") as f: label_encoder = pickle.load(f)
    with open(SCALER,  "rb") as f: scaler        = pickle.load(f)
    CATEGORIES = list(label_encoder.classes_)
    print(f"MLP ready — {len(CATEGORIES)} classes")
else:
    mlp_model = label_encoder = scaler = None
    CATEGORIES = []

if USE_RF:
    print("Loading RF model...")
    with open(RF_MODEL, "rb") as f: rf_model = pickle.load(f)
    print("RF ready")
else:
    rf_model = None
    if USE_MLP:
        print("RF model not found — using MLP only (still works well)")

def predict_landmarks(lm):
    """Predict letter from landmarks. Uses ensemble if RF available, else MLP only."""
    if not USE_MLP:
        return "?", 0.0
    vec    = extract_features(lm).reshape(1, -1)
    vec_sc = scaler.transform(vec)
    mlp_p  = mlp_model.predict(vec_sc, verbose=0)[0]

    if USE_RF:
        rf_p = rf_model.predict_proba(vec_sc)[0]
        probs = 0.6 * mlp_p + 0.4 * rf_p
    else:
        probs = mlp_p

    idx  = int(np.argmax(probs))
    conf = float(probs[idx])
    label = label_encoder.inverse_transform([idx])[0]
    return label, conf

def top3_predictions(lm):
    """Return top-3 (label, confidence) tuples."""
    if not USE_MLP:
        return []
    vec    = extract_features(lm).reshape(1, -1)
    vec_sc = scaler.transform(vec)
    mlp_p  = mlp_model.predict(vec_sc, verbose=0)[0]
    if USE_RF:
        rf_p  = rf_model.predict_proba(vec_sc)[0]
        probs = 0.6 * mlp_p + 0.4 * rf_p
    else:
        probs = mlp_p
    top_idx = np.argsort(probs)[::-1][:3]
    return [(label_encoder.inverse_transform([i])[0].upper(), round(float(probs[i])*100))
            for i in top_idx]

# ── Per-session state ──────────────────────────────────────────────────────────
class ClientState:
    def __init__(self):
        self.buf           = deque(maxlen=6)
        self.last_locked   = None
        self.lock_time     = 0.0
        self.no_hand_count = 0
        self.same_letter_count = 0  # how many times same letter confirmed in a row

_states: dict = {}

def get_state(uid: str) -> ClientState:
    if uid not in _states:
        _states[uid] = ClientState()
    return _states[uid]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f: return json.load(f)
    return {}

def save_users(u):
    with open(USERS_FILE, "w") as f: json.dump(u, f, indent=2)

def find_sign_image(idx):
    for ext in ("png", "jpeg", "jpg"):
        p = os.path.join(AUDIO_DIR, f"{idx}.{ext}")
        if os.path.exists(p): return p
    return None

def find_space_image():
    for ext in ("png", "jpeg", "jpg"):
        p = os.path.join(AUDIO_DIR, f"space.{ext}")
        if os.path.exists(p): return p
    return None

def img_to_b64(path):
    if not path or not os.path.exists(path): return None
    with open(path, "rb") as f:
        ext  = path.rsplit(".", 1)[-1]
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" in session: return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    d = request.json
    u, p, colour = d.get("username","").strip(), d.get("password",""), d.get("colour","").strip().lower()
    if not u or not p or not colour: return jsonify(ok=False, msg="Fill in all fields.")
    if len(p) < 4: return jsonify(ok=False, msg="Password min 4 characters.")
    users = load_users()
    if u in users: return jsonify(ok=False, msg="Username already exists.")
    users[u] = {"password": generate_password_hash(p), "colour": colour}
    save_users(users)
    return jsonify(ok=True, msg="Account created!")

@app.route("/login", methods=["POST"])
def login():
    d = request.json
    u, p = d.get("username","").strip(), d.get("password","")
    users = load_users()
    if u not in users: return jsonify(ok=False, msg="User not found.")
    ud = users[u]
    pw = ud["password"] if isinstance(ud, dict) else ud
    if not check_password_hash(pw, p): return jsonify(ok=False, msg="Incorrect password.")
    session["pending_user"] = u
    return jsonify(ok=True)

@app.route("/verify_colour", methods=["POST"])
def verify_colour():
    u = session.get("pending_user")
    if not u: return jsonify(ok=False, msg="Session expired."), 401
    colour = request.json.get("colour","").strip().lower()
    users  = load_users(); ud = users.get(u, {})
    stored = ud.get("colour","") if isinstance(ud, dict) else ""
    if not stored or colour == stored:
        session["user"] = u; session.pop("pending_user", None)
        return jsonify(ok=True)
    return jsonify(ok=False, msg="Wrong colour — try again")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("index"))
    return render_template("dashboard.html", username=session["user"])

# ── Predict ────────────────────────────────────────────────────────────────────
# Confidence threshold — lower = more responsive, higher = more accurate
CONF_THRESHOLD = 0.60   # was 0.72, too strict

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session: return jsonify(ok=False), 401
    uid   = session["user"]
    state = get_state(uid)

    img_data = request.json.get("image", "")
    try:
        _, encoded = img_data.split(",", 1)
        arr   = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: raise ValueError
    except Exception:
        return jsonify(hand=False, letter="—", conf=0, stable=False,
                       status="Bad frame", top3=[])

    # Enhance contrast slightly for better detection in varied lighting
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

    lm_raw = detect_hands(frame)

    if lm_raw is None:
        state.no_hand_count += 1
        # Only reset lock after hand gone for a while — prevents flicker
        if state.no_hand_count >= 8:
            state.last_locked = None
            state.buf.clear()
            state.same_letter_count = 0
        return jsonify(hand=False, letter="—", conf=0, stable=False,
                       status="No Hand Detected", top3=[])

    state.no_hand_count = 0

    # Reject tiny bounding boxes (false detections)
    xs = [l.x for l in lm_raw]; ys = [l.y for l in lm_raw]
    h, w = frame.shape[:2]
    bw = (max(xs) - min(xs)) * w
    bh = (max(ys) - min(ys)) * h
    if bw < 25 or bh < 25:
        return jsonify(hand=False, letter="—", conf=0, stable=False,
                       status="Hand too far", top3=[])

    label, conf = predict_landmarks(lm_raw)
    t3 = top3_predictions(lm_raw)

    if "unknown" in str(label).lower() or conf < 0.35:
        return jsonify(hand=True, letter="—", conf=round(conf*100), stable=False,
                       status="Detecting...", top3=t3)

    # Buffer: clear on gesture change for fast switching
    if state.buf:
        counts  = Counter(l for l, _ in state.buf)
        cur_top = counts.most_common(1)[0][0]
        if label != cur_top:
            state.buf.clear()
            state.same_letter_count = 0

    state.buf.append((label, conf))
    counts   = Counter(l for l, _ in state.buf)
    top, cnt = counts.most_common(1)[0]
    avg_conf = float(np.mean([c for l, c in state.buf if l == top]))

    # Stable = same letter seen 3+ times AND confidence above threshold
    stable = cnt >= 3 and avg_conf >= CONF_THRESHOLD

    new_letter = None
    now = time.time()

    if stable and top != "?":
        if top != state.last_locked:
            # New letter — lock it in after 0.8s hold (was 1.0s)
            if now - state.lock_time > 0.8:
                new_letter        = top.upper()
                state.last_locked = top
                state.lock_time   = now
                state.same_letter_count = 1
                state.buf.clear()
        else:
            # Same letter held — allow re-adding after 2s gap
            state.same_letter_count += 1
            if state.same_letter_count >= 15 and now - state.lock_time > 2.0:
                new_letter = top.upper()
                state.lock_time = now
                state.same_letter_count = 0
                state.buf.clear()

    return jsonify(
        hand=True,
        letter=top.upper() if stable else "—",
        conf=round(avg_conf * 100),
        stable=stable,
        new_letter=new_letter,
        status="Stable ✓" if stable else "Detecting...",
        top3=t3
    )

# ── Reset state (called when user clears sentence) ─────────────────────────────
@app.route("/reset_state", methods=["POST"])
def reset_state():
    if "user" not in session: return jsonify(ok=False), 401
    uid = session["user"]
    if uid in _states:
        del _states[uid]
    return jsonify(ok=True)

# ── Speak ──────────────────────────────────────────────────────────────────────
@app.route("/speak", methods=["POST"])
def speak():
    if "user" not in session: return jsonify(ok=False), 401
    text = request.json.get("text", "").strip()
    if not text: return jsonify(ok=False, msg="No text")
    fname = f"speech_{session['user']}.mp3"
    path  = os.path.join(TMP_DIR, fname)
    gTTS(text=text.lower(), lang="en").save(path)
    return jsonify(ok=True, url=f"/static/tmp/{fname}")

# ── Signs ──────────────────────────────────────────────────────────────────────
@app.route("/signs", methods=["POST"])
def signs():
    if "user" not in session: return jsonify(ok=False), 401
    text = request.json.get("text", "").strip().lower()
    if not text: return jsonify(ok=False, msg="No text")
    result = []
    for ch in text:
        if ch == " ":
            result.append({"char": "SPC", "img": img_to_b64(find_space_image())})
        elif ch in LETTERS:
            idx = int(LETTERS[ch]) - 1
            result.append({"char": ch.upper(), "img": img_to_b64(find_sign_image(idx))})
    return jsonify(ok=True, signs=result)

if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0")
