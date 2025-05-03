from flask import Flask, request, jsonify
import joblib
import numpy as np
import gdown
import os

app = Flask(__name__)

model_path = "model55.joblib"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=11fMj58bXC3z8rpNC9SpMPJHX1AsM1gm1"
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)

@app.route("/")
def home():
    return "Model API is working ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})
