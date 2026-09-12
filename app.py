from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# =========================
# LOAD TRAINED MODEL
# =========================

model = tf.keras.models.load_model("brain_tumour_classifier.keras")

print("Model loaded successfully!")


# =========================
# CLASS NAMES
# =========================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION API
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image was uploaded
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # =========================
        # READ IMAGE
        # =========================

        image = Image.open(file).convert("RGB")

        # =========================
        # RESIZE
        # =========================

        image = image.resize((128, 128))

        # =========================
        # CONVERT TO NUMPY
        # =========================

        image_array = np.array(image)

        # =========================
        # NORMALIZATION
        # =========================

        image_array = image_array / 255.0

        # =========================
        # ADD BATCH DIMENSION
        # =========================

        image_array = np.expand_dims(image_array, axis=0)

        # =========================
        # PREDICTION
        # =========================

        predictions = model.predict(image_array, verbose=0)

        probabilities = predictions[0]

        predicted_index = np.argmax(probabilities)

        predicted_class = class_names[predicted_index]

        confidence = float(probabilities[predicted_index]) * 100

        # =========================
        # RETURN RESULT
        # =========================

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": {
                "glioma": round(float(probabilities[0]) * 100, 2),
                "meningioma": round(float(probabilities[1]) * 100, 2),
                "notumor": round(float(probabilities[2]) * 100, 2),
                "pituitary": round(float(probabilities[3]) * 100, 2)
            }
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":
    app.run(debug=True)