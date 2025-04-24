
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("model4.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
