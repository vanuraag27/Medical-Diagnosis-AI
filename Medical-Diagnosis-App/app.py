import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS  # Enables cross-origin requests (Optional)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS (Optional for frontend integration)

# Load the trained model
MODEL_DIR = r"C:\Users\HP\Desktop\Medical-Diagnosis-App\model"
MODEL_PATH = os.path.join(MODEL_DIR, "medical_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)  # ✅ Ensure model is loaded here

# Print model architecture (Make sure this is AFTER model is loaded)
print("✅ Model loaded successfully!")
print(model.summary())  # ✅ Now it should work


# Define image preprocessing function
def preprocess_image(image):
    try:
        # Convert image to grayscale (1 channel)
        img = image.convert("L")  

        # Resize image to match model input (128, 128)
        img = img.resize((128, 128))

        # Convert to NumPy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # ✅ Ensure correct shape: (batch_size, height, width, channels)
        img_array = img_array.reshape(1, 128, 128, 1)  # Model expects (None, 128, 128, 1)

        return img_array
    except Exception as e:
        return str(e)
# Define API Endpoints
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Medical Diagnosis API is running!"})

LABELS = {0: "Healthy", 1: "Disease Detected"}
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image uploads, preprocesses images, and returns model predictions.
    """
    try:
        # Check if file is provided
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image = Image.open(file)

        # Process the image
        processed_image = preprocess_image(image)
        if isinstance(processed_image, str):
            return jsonify({"error": f"Image preprocessing failed: {processed_image}"}), 400

        # Make prediction
        prediction = model.predict(processed_image)

        # Get class and confidence
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence_score = float(np.max(prediction))

        # ✅ Return a human-readable label
        predicted_label = LABELS.get(predicted_class, "Unknown")

        return jsonify({
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence_score": confidence_score
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
