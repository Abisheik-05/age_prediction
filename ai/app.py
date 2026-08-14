from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

@app.route("/predict-age", methods=["POST"])
def predict_age():
    try:
        image_path = request.json["image_path"]

        result = DeepFace.analyze(
            img_path=image_path,
            actions=["age"],
            detector_backend="retinaface",
            enforce_detection=False
        )

        age = result[0]["age"]

        return jsonify({
            "success": True,
            "age": round(age)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(port=5001, debug=True)