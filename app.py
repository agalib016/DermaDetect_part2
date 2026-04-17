import os
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from PIL import Image
import tensorflow
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
except Exception:
    from keras.models import load_model
    from keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'skincancer-ai-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Database & Login Manager ──────────────────────────────────────────────────
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please sign in to access the diagnostics tool.'
login_manager.login_message_category = 'error'


class User(UserMixin, db.Model):
    """User model for authentication."""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Create tables
with app.app_context():
    db.create_all()
    print("[INFO] Database initialized (users.db)")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
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
    """
    Load gatekeeper_model.keras which was saved with an early Keras 3 that stored:
      - config.json  using old module path keras.src.engine.functional
      - model.weights.h5  using Keras-3 class-type positional keys (e.g. conv2d/vars/0)

    Strategy:
      1. Build the tf_keras model using from_config (handles TFOpLambda & old config).
      2. Walk the layer tree in creation order; resolve each layer's H5 key by
         class-name counter (BatchNormalization → batch_normalization, Conv2D → conv2d …).
      3. Assign weights values directly – bypasses the naming-format mismatch.
    """
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
                    continue  # e.g. augmentation Sequential (no weights)
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
        # Extract config + weights from the .keras ZIP
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
    """User registration."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Validation
        if not all([first_name, last_name, email, password]):
            flash("All fields are required.", "error")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
            return redirect(url_for('register'))

        # Create user
        user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please sign in.", "success")
        return redirect(url_for('login'))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return redirect(url_for('login'))

        login_user(user, remember=True)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('home'))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """User logout."""
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

        return jsonify(
            {
                "predicted_class": predicted_class,
                "predicted_class_full": CLASSES_FULL.get(
                    predicted_class, predicted_class
                ),
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
    """Health check endpoint for monitoring and CI."""
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
    """Return available lesion classes and metadata."""
    sex_options = (
        list(sex_encoder.classes_) if sex_encoder else ["male", "female", "unknown"]
    )
    loc_options = (
        list(loc_encoder.classes_)
        if loc_encoder
        else [
            "abdomen",
            "acral",
            "back",
            "chest",
            "ear",
            "face",
            "foot",
            "genital",
            "hand",
            "lower extremity",
            "neck",
            "scalp",
            "trunk",
            "upper extremity",
            "unknown",
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
