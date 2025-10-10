# app/app.py
import os, json, numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")
CLASS_JSON = os.path.join(BASE_DIR, "model", "class_indices.json")

UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL = load_model(MODEL_PATH)
with open(CLASS_JSON, "r") as f:
    class_indices = json.load(f)
inv_map = {v: k for k, v in class_indices.items()}  # invert mapping

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    img = image.load_img(save_path, target_size=(224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = float(MODEL.predict(x)[0][0])  # sigmoid prob
    class_idx = int(pred >= 0.5)
    class_name = inv_map[class_idx]
    confidence = pred if class_idx == 1 else 1 - pred
    confidence_pct = round(confidence * 100, 2)

    return render_template("result.html", label=class_name.capitalize(),
                           confidence=confidence_pct, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
