import os
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
except Exception:
    from keras.models import load_model
    from keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'skincancer-ai-secret-key-change-in-production')

# ── Database & Login Manager ──────────────────────────────────────────────────
db = None
try:
    if not firebase_admin._apps:
        firebase_creds_raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY") or os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if firebase_creds_raw:
            import json, base64
            cleaned = firebase_creds_raw.strip()
            if not cleaned.startswith("{"):
                cleaned = base64.b64decode(cleaned).decode("utf-8")
            cred_dict = json.loads(cleaned)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        else:
            cert_path = os.path.join(BASE_DIR, "serviceAccountKey.json")
            if os.path.exists(cert_path):
                cred = credentials.Certificate(cert_path)
                firebase_admin.initialize_app(cred)
            else:
                print("[WARN] No Firebase credentials found. Provide FIREBASE_SERVICE_ACCOUNT_KEY or serviceAccountKey.json.")
    if firebase_admin._apps:
        db = firestore.client()
        print("[INFO] Connected to Firebase Firestore")
except Exception as e:
    print(f"[ERROR] Firebase initialization error: {e}")

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please sign in to access the diagnostics tool.'
login_manager.login_message_category = 'error'


class User(UserMixin):
    """User model for authentication."""
    def __init__(self, user_id, email, first_name, last_name, password_hash, created_at=None, avatar_url=None):
        self.id = user_id
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.password_hash = password_hash
        self.avatar_url = avatar_url
        self.created_at = created_at or datetime.utcnow()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def get_by_email(email):
        if db is None:
            return None
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        if query:
            doc = query[0]
            data = doc.to_dict()
            return User(
                user_id=doc.id,
                email=data.get('email'),
                first_name=data.get('first_name'),
                last_name=data.get('last_name'),
                password_hash=data.get('password_hash'),
                created_at=data.get('created_at'),
                avatar_url=data.get('avatar_url')
            )
        return None

    @staticmethod
    def get_by_id(user_id):
        if db is None:
            return None
        doc_ref = db.collection('users').document(str(user_id))
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return User(
                user_id=doc.id,
                email=data.get('email'),
                first_name=data.get('first_name'),
                last_name=data.get('last_name'),
                password_hash=data.get('password_hash'),
                created_at=data.get('created_at'),
                avatar_url=data.get('avatar_url')
            )
        return None

    def save(self):
        if db is None:
            print("[WARN] Firestore not connected; cannot save user.")
            return
        doc_ref = db.collection('users').document(str(self.id)) if self.id else db.collection('users').document()
        if not self.id:
            self.id = doc_ref.id
        doc_ref.set({
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'password_hash': self.password_hash,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at
        })


@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)


MODEL_CANDIDATES = [
    # EfficientNetB3 (v2.0 — highest accuracy, trained with train_unified_model.py)
    os.path.join(BASE_DIR, "skin-cancer-7-classes_EfficientNetB3_Unified.keras"),
    # MobileNetV2 fallbacks (v1.x — used if EfficientNetB3 not yet trained)
    os.path.join(BASE_DIR, "skin-cancer-7-classes_Unified_MobileNet.keras"),
    os.path.join(BASE_DIR, "skin-cancer-7-classes_MobileNet_ph2_model.keras"),
    os.path.join(BASE_DIR, "skin-cancer-7-classes_MobileNet_ph1_model.keras"),
    os.path.join(BASE_DIR, "MobileNet.h5"),
]
GATEKEEPER_MODEL_PATH = os.path.join(BASE_DIR, "gatekeeper_model.keras")
GATEKEEPER_SKIN_THRESHOLD = 0.50
SEX_ENCODER_PATH = os.path.join(BASE_DIR, "skin-cancer-7-classes_sex_encoder.pkl")
LOC_ENCODER_PATH = os.path.join(BASE_DIR, "skin-cancer-7-classes_loc_encoder.pkl")
AGE_SCALER_PATH = os.path.join(BASE_DIR, "skin-cancer-7-classes_age_scaler.pkl")

CLASSES = ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]

CLASSES_FULL = {
    "bkl": "Benign Keratosis",
    "nv": "Melanocytic Nevi",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular Lesion",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses / Bowen's Disease",
}

DANGEROUS_CLASSES = ["mel", "akiec"]


class CompatDepthwiseConv2D(BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


print("[INFO] Loading model and preprocessors...")
model = None
gatekeeper_model = None
sex_encoder = None
loc_encoder = None
age_scaler = None
model_path_used = None
model_expects_tabular = False

model_load_errors = []
for candidate in MODEL_CANDIDATES:
    if not os.path.exists(candidate):
        continue

    try:
        model = load_model(
            candidate,
            custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D},
            compile=False,
        )
        model_path_used = candidate
        break
    except Exception as e:
        model_load_errors.append(f"{os.path.basename(candidate)}: {e}")

if model is None:
    print("[ERROR] Could not load any model candidate.")
    if model_load_errors:
        print("[ERROR] Model load attempts:")
        for err in model_load_errors:
            print(f"  - {err}")
else:
    model_expects_tabular = len(model.inputs) > 1
    print(f"[INFO] Model loaded from: {model_path_used}")
    print(f"[INFO] Model input count: {len(model.inputs)}")

    try:
        with open(SEX_ENCODER_PATH, "rb") as f:
            sex_encoder = pickle.load(f)
        with open(LOC_ENCODER_PATH, "rb") as f:
            loc_encoder = pickle.load(f)
        with open(AGE_SCALER_PATH, "rb") as f:
            age_scaler = pickle.load(f)
        print("[INFO] Preprocessors loaded successfully!")
        print(f"[INFO] Sex classes: {sex_encoder.classes_}")
        print(f"[INFO] Localization classes: {loc_encoder.classes_}")
    except Exception as e:
        print(f"[WARN] Model loaded but preprocessors unavailable: {e}")


def _load_gatekeeper(path):
    import zipfile, json, re, tempfile, shutil, h5py
    import tf_keras as _tf_keras

    def _cls_to_h5key(name):
        return re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()

    def _assign_recursive(layer_list, h5file, prefix):
        counters = {}
        for layer in layer_list:
            cls_key = _cls_to_h5key(type(layer).__name__)
            sub_layers = getattr(layer, "layers", None)
            is_container = (
                isinstance(layer, (_tf_keras.Sequential, _tf_keras.Model))
                and sub_layers
            )
            if is_container:
                if not any(getattr(l, "variables", None) for l in sub_layers):
                    continue
                cnt = counters.get(cls_key, 0)
                sub_prefix = f"{prefix}/{cls_key}" + (f"_{cnt}" if cnt > 0 else "")
                counters[cls_key] = cnt + 1
                _assign_recursive(sub_layers, h5file, sub_prefix + "/layers")
            elif layer.variables:
                cnt = counters.get(cls_key, 0)
                layer_prefix = f"{prefix}/{cls_key}" + (f"_{cnt}" if cnt > 0 else "")
                counters[cls_key] = cnt + 1
                for i, v in enumerate(layer.variables):
                    h5_key = f"{layer_prefix}/vars/{i}"
                    if h5_key in h5file:
                        v.assign(h5file[h5_key][()])

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, "r") as zf:
            cfg_raw = zf.read("config.json").decode("utf-8")
            zf.extract("model.weights.h5", tmpdir)

        cfg = json.loads(cfg_raw)
        model = _tf_keras.Model.from_config(cfg["config"])

        h5_path = os.path.join(tmpdir, "model.weights.h5")
        with h5py.File(h5_path, "r") as h5f:
            _assign_recursive(model.layers, h5f, "layers")

        return model
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if os.path.exists(GATEKEEPER_MODEL_PATH):
    try:
        gatekeeper_model = _load_gatekeeper(GATEKEEPER_MODEL_PATH)
        print(f"[INFO] Gatekeeper model loaded from: {GATEKEEPER_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load gatekeeper model: {e}")
else:
    print("[WARN] Gatekeeper model not found. Continuing without gatekeeper filter.")


def encode_tabular(age, sex, localization):
    if sex_encoder is None or loc_encoder is None or age_scaler is None:
        raise RuntimeError(
            "Tabular preprocessors are not loaded; cannot run multi-input inference."
        )

    num_sex_classes = len(sex_encoder.classes_)
    num_loc_classes = len(loc_encoder.classes_)

    age_scaled = age_scaler.transform(np.array([[float(age)]]))[0]

    sex_ohe = np.zeros(num_sex_classes)
    if sex in sex_encoder.classes_:
        sex_ohe[sex_encoder.transform([sex])[0]] = 1.0
    else:
        sex_ohe[sex_encoder.transform(["unknown"])[0]] = 1.0

    loc_ohe = np.zeros(num_loc_classes)
    if localization in loc_encoder.classes_:
        loc_ohe[loc_encoder.transform([localization])[0]] = 1.0
    else:
        loc_ohe[loc_encoder.transform(["unknown"])[0]] = 1.0

    return np.concatenate([age_scaled, sex_ohe, loc_ohe])


def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_skin_probability(img_array):
    if gatekeeper_model is None:
        return None

    pred = gatekeeper_model.predict(img_array, verbose=0)
    skin_prob = float(np.squeeze(pred))
    return float(np.clip(skin_prob, 0.0, 1.0))


# ── Authentication Routes ─────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not all([first_name, last_name, email, password]):
            flash("All fields are required.", "error")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('register'))

        if User.get_by_email(email):
            flash("An account with this email already exists.", "error")
            return redirect(url_for('register'))

        user = User(user_id=None, first_name=first_name, last_name=last_name, email=email, password_hash="")
        user.set_password(password)
        user.save()

        flash("Account created successfully! Please sign in.", "success")
        return redirect(url_for('login'))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.get_by_email(email)

        if user is None or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return redirect(url_for('login'))

        login_user(user, remember=True)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('get_started'))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for('login'))


# ── Application Routes ────────────────────────────────────────────────────────

@app.route("/")
@login_required
def home():
    sex_options = (
        list(sex_encoder.classes_) if sex_encoder else ["male", "female", "unknown"]
    )
    loc_options = list(loc_encoder.classes_) if loc_encoder else []
    return render_template(
        "index.html", sex_options=sex_options, loc_options=loc_options, user=current_user
    )


@app.route("/get-started")
@login_required
def get_started():
    """Awareness page about the 7 skin cancer types."""
    return render_template("get-started.html", user=current_user)


@app.route("/dashboard")
@login_required
def dashboard():
    """User's prediction history dashboard."""
    query = []
    if db is not None:
        predictions_ref = db.collection('predictions')
        query = predictions_ref.where('user_id', '==', str(current_user.id)).get()
    
    class PredictionWrapper:
        def __init__(self, data):
            self.patient_age = data.get('patient_age')
            self.patient_sex = data.get('patient_sex')
            self.localization = data.get('localization')
            self.predicted_class = data.get('predicted_class')
            self.predicted_class_full = data.get('predicted_class_full')
            self.confidence = data.get('confidence')
            self.is_dangerous = data.get('is_dangerous')
            self.gatekeeper_passed = data.get('gatekeeper_passed')
            self.created_at = data.get('created_at')

    predictions = [PredictionWrapper(doc.to_dict()) for doc in query]
    # Sort predictions in Python to avoid needing a Firestore composite index
    predictions.sort(key=lambda p: p.created_at, reverse=True)
    total = len(predictions)
    dangerous_count = sum(1 for p in predictions if p.is_dangerous)
    safe_count = total - dangerous_count
    return render_template(
        "dashboard.html",
        user=current_user,
        predictions=predictions,
        total=total,
        dangerous_count=dangerous_count,
        safe_count=safe_count,
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    age = request.form.get("age", "").strip()
    sex = request.form.get("sex", "unknown").strip()
    localization = request.form.get("localization", "unknown").strip()

    if not age:
        return jsonify({"error": "Age is required"}), 400

    try:
        age = float(age)
        if age < 0 or age > 120:
            return jsonify({"error": "Please enter a valid age (0-120)"}), 400
    except ValueError:
        return jsonify({"error": "Age must be a number"}), 400

    try:
        img_array = preprocess_image(file)

        skin_prob = predict_skin_probability(img_array)
        if skin_prob is not None and skin_prob < GATEKEEPER_SKIN_THRESHOLD:
            return jsonify(
                {
                    "predicted_class": "not_skin",
                    "predicted_class_full": "Not a skin lesion image",
                    "confidence": round((1.0 - skin_prob) * 100, 2),
                    "is_dangerous": False,
                    "all_probabilities": {k: 0.0 for k in CLASSES},
                    "class_names_full": CLASSES_FULL,
                    "gatekeeper": {
                        "enabled": True,
                        "passed": False,
                        "skin_probability": round(skin_prob * 100, 2),
                        "threshold": round(GATEKEEPER_SKIN_THRESHOLD * 100, 2),
                    },
                }
            )

        if model_expects_tabular:
            tab_array = encode_tabular(age, sex, localization)
            tab_array = np.expand_dims(tab_array, axis=0)
            predictions = model.predict([img_array, tab_array], verbose=0)[0]
        else:
            predictions = model.predict(img_array, verbose=0)[0]

        top_index = int(np.argmax(predictions))
        predicted_class = CLASSES[top_index]
        confidence = float(predictions[top_index] * 100)
        is_dangerous = predicted_class in DANGEROUS_CLASSES

        all_probs = {
            CLASSES[i]: float(predictions[i] * 100) for i in range(len(CLASSES))
        }

        # ── Save prediction to database ───────────────────────────────────────
        try:
            record_data = {
                'user_id': str(current_user.id),
                'patient_age': age,
                'patient_sex': sex,
                'localization': localization,
                'predicted_class': predicted_class,
                'predicted_class_full': CLASSES_FULL.get(predicted_class, predicted_class),
                'confidence': round(confidence, 2),
                'is_dangerous': is_dangerous,
                'gatekeeper_passed': True if skin_prob is not None else None,
                'created_at': datetime.utcnow()
            }
            db.collection('predictions').add(record_data)
        except Exception as db_err:
            print(f"[WARN] Could not save prediction to DB: {db_err}")

        return jsonify(
            {
                "predicted_class": predicted_class,
                "predicted_class_full": CLASSES_FULL.get(predicted_class, predicted_class),
                "confidence": round(confidence, 2),
                "is_dangerous": is_dangerous,
                "all_probabilities": all_probs,
                "class_names_full": CLASSES_FULL,
                "gatekeeper": {
                    "enabled": gatekeeper_model is not None,
                    "passed": True if skin_prob is not None else None,
                    "skin_probability": (
                        round(skin_prob * 100, 2) if skin_prob is not None else None
                    ),
                    "threshold": round(GATEKEEPER_SKIN_THRESHOLD * 100, 2),
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_file": (
                os.path.basename(model_path_used) if model_path_used else None
            ),
            "gatekeeper_loaded": gatekeeper_model is not None,
            "preprocessors_loaded": all(
                x is not None for x in [sex_encoder, loc_encoder, age_scaler]
            ),
            "model_expects_tabular": model_expects_tabular,
        }
    )


@app.route("/api/classes")
@login_required
def api_classes():
    sex_options = (
        list(sex_encoder.classes_) if sex_encoder else ["male", "female", "unknown"]
    )
    loc_options = (
        list(loc_encoder.classes_)
        if loc_encoder
        else [
            "abdomen", "acral", "back", "chest", "ear", "face", "foot",
            "genital", "hand", "lower extremity", "neck", "scalp",
            "trunk", "upper extremity", "unknown",
        ]
    )
    return jsonify(
        {
            "classes": CLASSES,
            "classes_full": CLASSES_FULL,
            "dangerous_classes": DANGEROUS_CLASSES,
            "sex_options": sex_options,
            "localization_options": loc_options,
        }
    )

@app.route("/explore")
@login_required
def explore():
    return render_template("explore.html", user=current_user)

@app.route("/articles")
@login_required
def articles():
    return render_template("articles.html", user=current_user)

@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        
        if first_name and last_name:
            current_user.first_name = first_name
            current_user.last_name = last_name

        if "profile_picture" in request.files:
            file = request.files["profile_picture"]
            if file and file.filename != "":
                try:
                    # In serverless environments like Vercel, the filesystem is read-only.
                    # Storing avatar as base64 Data URL avoids filesystem errors and persists across serverless instances.
                    import base64
                    file_bytes = file.read()
                    mime_type = file.mimetype or "image/jpeg"
                    b64_str = base64.b64encode(file_bytes).decode("utf-8")
                    current_user.avatar_url = f"data:{mime_type};base64,{b64_str}"
                except Exception as upload_err:
                    print(f"[WARN] Profile picture upload error: {upload_err}")

        current_user.save()
        flash("Profile updated successfully.", "success")
        return redirect(url_for("account"))

    return render_template("account.html", user=current_user)

@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    password = request.form.get("password", "")
    if not current_user.check_password(password):
        flash("Incorrect password. Account deletion failed.", "error")
        return redirect(url_for("account"))
    
    user_id = str(current_user.id)
    
    if db is not None:
        # Delete predictions
        predictions_ref = db.collection('predictions')
        query = predictions_ref.where('user_id', '==', user_id).get()
        for doc in query:
            db.collection('predictions').document(doc.id).delete()
        
        # Delete user
        db.collection('users').document(user_id).delete()
    
    logout_user()
    flash("Your account has been successfully deleted.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
