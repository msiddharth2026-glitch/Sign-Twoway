"""
Sign Language Translator — Fixed & Reliable
"""
import os, json, base64, pickle, sys, time
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from gtts import gTTS
from string import ascii_lowercase
from collections import deque, Counter

BASE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(BASE)
DATASET_DIR = os.path.join(ROOT, "dataset", "original_images")
USERS_FILE  = os.path.join(BASE, "users.json")
TMP_DIR     = os.path.join(BASE, "static", "tmp")
SIGNS_DIR   = os.path.join(BASE, "static", "signs")
LM_MODEL    = os.path.join(ROOT, "landmark_model.keras")
# Also check web folder (for Railway deployment)
if not os.path.exists(LM_MODEL):
    LM_MODEL = os.path.join(BASE, "landmark_model.keras")
ENCODER     = os.path.join(BASE, "label_encoder.pkl")
SCALER      = os.path.join(BASE, "scaler.pkl")
RF_MODEL    = os.path.join(BASE, "rf_model.pkl")
os.makedirs(TMP_DIR, exist_ok=True)

sys.path.insert(0, ROOT)

LETTERS = {ch: str(i) for i, ch in enumerate(ascii_lowercase, start=1)}
DIGITS  = set('0123456789')

# ── Load models ────────────────────────────────────────────────────────────────
try:
    from feature_extractor import extract_features
    _fe_ok = True
except ImportError:
    _fe_ok = False
    print("WARNING: feature_extractor not found")

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
    print("WARNING: MLP model not found")

if USE_RF:
    print("Loading RF model...")
    with open(RF_MODEL, "rb") as f: rf_model = pickle.load(f)
    print("RF ready")
else:
    rf_model = None

# ── Landmark object (converts JSON dict to attribute-based object) ─────────────
class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, d):
        self.x = float(d["x"])
        self.y = float(d["y"])
        self.z = float(d.get("z", 0.0))

def predict_landmarks(lm_list):
    """Single inference — returns (label, conf, top3)."""
    if not USE_MLP:
        return "?", 0.0, []
    try:
        vec    = extract_features(lm_list).reshape(1, -1)
        vec_sc = scaler.transform(vec)
        mlp_p  = mlp_model.predict(vec_sc, verbose=0)[0]
        if USE_RF:
            rf_p  = rf_model.predict_proba(vec_sc)[0]
            probs = 0.6 * mlp_p + 0.4 * rf_p
        else:
            probs = mlp_p
        top_idx = np.argsort(probs)[::-1]
        label   = str(label_encoder.inverse_transform([top_idx[0]])[0])
        conf    = float(probs[top_idx[0]])
        top3    = [(str(label_encoder.inverse_transform([i])[0]).upper(),
                    round(float(probs[i]) * 100)) for i in top_idx[:3]]
        return label, conf, top3
    except Exception as e:
        print(f"Prediction error: {e}")
        return "?", 0.0, []

# ── Per-session state ──────────────────────────────────────────────────────────
class ClientState:
    def __init__(self):
        self.buf           = deque(maxlen=8)  # slightly smaller for faster response
        self.last_locked   = None
        self.lock_time     = 0.0
        self.no_hand_count = 0

_states: dict = {}

def get_state(uid):
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

def img_to_b64(path):
    if not path or not os.path.exists(path): return None
    with open(path, "rb") as f:
        ext  = path.rsplit(".", 1)[-1].lower()
        mime = "jpeg" if ext in ("jpg","jpeg") else ("gif" if ext=="gif" else "png")
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()

def get_sign_image(ch):
    ch_up = ch.upper()
    for ext in ("jpg", "gif", "png"):
        p = os.path.join(SIGNS_DIR, f"{ch_up}.{ext}")
        if os.path.exists(p): return img_to_b64(p)
    letter_dir = os.path.join(DATASET_DIR, ch_up)
    if os.path.isdir(letter_dir):
        for f in sorted(os.listdir(letter_dir)):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                return img_to_b64(os.path.join(letter_dir, f))
    return None

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "sl-translator-secret-2024")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_PERMANENT"] = False

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" in session: return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    d = request.json or {}
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
    d = request.json or {}
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
    colour = (request.json or {}).get("colour","").strip().lower()
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
CONF_THRESHOLD = 0.65  # balanced — accurate but responsive

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session: return jsonify(ok=False), 401
    uid   = session["user"]
    state = get_state(uid)
    data  = request.json or {}

    # Accept multi-hand (hands) or single-hand (landmarks)
    hands_data = data.get("hands")
    landmarks  = data.get("landmarks")

    if hands_data and len(hands_data) > 0:
        raw_hands = hands_data
    elif landmarks and len(landmarks) == 21:
        raw_hands = [landmarks]
    else:
        raw_hands = []

    if not raw_hands:
        state.no_hand_count += 1
        if state.no_hand_count >= 6:
            state.last_locked = None
            state.buf.clear()
        return jsonify(hand=False, letter="—", conf=0, stable=False,
                       status="No Hand Detected", top3=[])

    state.no_hand_count = 0

    # Convert all hands to LM objects
    try:
        all_hands_lm = []
        for hand in raw_hands[:2]:  # max 2 hands
            lms = [LM(l) for l in hand]
            if len(lms) == 21:
                all_hands_lm.append(lms)
        if not all_hands_lm:
            raise ValueError("No valid hands")
    except Exception:
        return jsonify(hand=False, letter="—", conf=0, stable=False,
                       status="Bad landmarks", top3=[])

    # Validate primary hand bounding box
    primary = all_hands_lm[0]
    xs = [l.x for l in primary]; ys = [l.y for l in primary]
    bw = max(xs) - min(xs); bh = max(ys) - min(ys)
    if bw < 0.04 or bh < 0.04 or bw > 0.92 or bh > 0.92:
        return jsonify(hand=False, letter="—", conf=0, stable=False,
                       status="No Hand Detected", top3=[])

    # Use primary hand for prediction (model trained on single-hand features)
    # Two-hand signs: use the hand with larger bounding box (dominant hand)
    if len(all_hands_lm) == 2:
        def bbox_area(lms):
            xs = [l.x for l in lms]; ys = [l.y for l in lms]
            return (max(xs)-min(xs)) * (max(ys)-min(ys))
        lm_raw = max(all_hands_lm, key=bbox_area)
    else:
        lm_raw = primary

    label, conf, t3 = predict_landmarks(lm_raw)

    if "unknown" in str(label).lower() or conf < 0.30:
        return jsonify(hand=True, letter="—", conf=round(conf*100),
                       stable=False, status="Detecting...", top3=t3)

    # Temporal buffer — clear on gesture change
    if state.buf:
        cur_top = Counter(l for l,_ in state.buf).most_common(1)[0][0]
        if label != cur_top:
            state.buf.clear()

    state.buf.append((label, conf))
    counts   = Counter(l for l,_ in state.buf)
    top, cnt = counts.most_common(1)[0]
    avg_conf = float(np.mean([c for l,c in state.buf if l==top]))
    stable   = cnt >= 4 and avg_conf >= CONF_THRESHOLD  # 4 frames instead of 5

    new_letter = None
    now = time.time()
    if stable and top != "?":
        if top != state.last_locked:
            if now - state.lock_time > 0.5:  # faster lock — was 0.7s
                new_letter      = top.upper()
                state.last_locked = top
                state.lock_time   = now
                state.buf.clear()
        elif now - state.lock_time > 2.5:
            # Allow same letter again after 2.5s gap
            new_letter      = top.upper()
            state.lock_time = now
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

@app.route("/reset_state", methods=["POST"])
def reset_state():
    if "user" not in session: return jsonify(ok=False), 401
    _states.pop(session["user"], None)
    return jsonify(ok=True)

@app.route("/speak", methods=["POST"])
def speak():
    if "user" not in session: return jsonify(ok=False), 401
    text = (request.json or {}).get("text","").strip()
    if not text: return jsonify(ok=False, msg="No text")
    try:
        # text is letters joined — convert to readable words
        # e.g. "HELLO WORLD" stays as is, already space-separated words
        spoken = text.lower()
        fname = f"speech_{session['user']}.mp3"
        path  = os.path.join(TMP_DIR, fname)
        gTTS(text=spoken, lang="en", slow=False).save(path)
        return jsonify(ok=True, url=f"/static/tmp/{fname}")
    except Exception as e:
        return jsonify(ok=False, msg=str(e))

@app.route("/signs", methods=["POST"])
def signs():
    if "user" not in session: return jsonify(ok=False), 401
    text = (request.json or {}).get("text","").strip().lower()
    if not text: return jsonify(ok=False, msg="No text")
    result = []
    for ch in text:
        if ch == " ":
            result.append({"char":"SPC","img":None})
        elif ch in LETTERS or ch in DIGITS:
            result.append({"char":ch.upper(),"img":get_sign_image(ch)})
    return jsonify(ok=True, signs=result)

if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0")
