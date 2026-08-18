"""
Age Prediction Service using ResNet50
Loads trained model once at startup and provides age prediction via Flask.

Model: age_resnet50_best.keras
- ResNet50 backbone (frozen)
- Regression output: Dense(1, linear)
- Input: 224x224 RGB images
- Output: Age (1-100)
- Validation MAE: 8.0992
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask import Flask, request, jsonify
import json

# =====================================================================
# LOGGING SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('age_prediction_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ai" / "models" / "age_resnet50_best.keras"
IMAGE_SIZE = (224, 224)
VALID_AGE_RANGE = (1, 100)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FORMATS = {'image/jpeg', 'image/png'}

# =====================================================================
# FLASK APP SETUP
# =====================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model (loaded once at startup)
model = None

# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model():
    """
    Load the trained ResNet50 model.
    Called once at application startup.
    """
    global model
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("Model loaded successfully!")
        logger.info(f"Model architecture: {model.summary()}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# =====================================================================
# IMAGE PREPROCESSING (EXACT MATCH TO TRAINING)
# =====================================================================
def preprocess_image(image_path):
    """
    Preprocess image exactly as done during training:
    1. Load image
    2. Resize to 224x224
    3. Convert to numpy array (0-255 range)
    4. Apply ResNet50 preprocess_input (ImageNet normalization)
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        np.array: Preprocessed image batch (1, 224, 224, 3)
    
    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    try:
        # Load image with target size
        img = image.load_img(str(image_path), target_size=IMAGE_SIZE)
        
        # Convert to numpy array (0-255 range)
        img_array = image.img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply ResNet50 preprocessing (ImageNet normalization)
        # This matches: train_datagen.preprocessing_function=preprocess_input
        img_array = preprocess_input(img_array)
        
        logger.debug(f"Image preprocessed successfully: shape={img_array.shape}, dtype={img_array.dtype}")
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

# =====================================================================
# AGE PREDICTION
# =====================================================================
def predict_age(image_path):
    """
    Predict age from image.
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        dict: {
            'predicted_age': int (1-100),
            'raw_output': float (unclamped model output),
            'confidence': float (always 1.0 for regression)
        }
    
    Raises:
        RuntimeError: If model is not loaded
        ValueError: If image preprocessing fails
    """
    if model is None:
        logger.error("Model not loaded. Call load_model() first.")
        raise RuntimeError("Model not loaded")
    
    try:
        # Preprocess image (matches training preprocessing exactly)
        img_array = preprocess_image(image_path)
        
        # Run inference
        logger.debug(f"Running inference on: {image_path}")
        raw_output = model.predict(img_array, verbose=0)[0][0]
        raw_output = float(raw_output)
        
        # Clamp to valid age range [1, 100]
        predicted_age = max(VALID_AGE_RANGE[0], min(VALID_AGE_RANGE[1], raw_output))
        predicted_age = int(round(predicted_age))
        
        result = {
            'predicted_age': predicted_age,
            'raw_output': raw_output,
            'confidence': 1.0  # Regression model always has confidence 1.0
        }
        
        logger.info(f"Prediction complete: raw_output={raw_output:.2f}, clamped_age={predicted_age}")
        return result
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

# =====================================================================
# FLASK ROUTES
# =====================================================================

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "OK", "service": "age_prediction_service"}), 200

@app.route("/predict-age", methods=["POST"])
def predict_age_endpoint():
    """
    Predict age from uploaded image or image path.
    
    POST /predict-age
    
    Request (file upload):
        Form data: image (file)
    
    OR
    
    Request (file path):
        JSON: {"image_path": "/path/to/image.jpg"}
    
    Response:
        {
            "predicted_age": 24,
            "raw_output": 24.35,
            "confidence": 1.0,
            "image": "filename.jpg"
        }
    
    Error Response:
        {
            "error": "Error message",
            "status": 400
        }
    """
    try:
        image_path = None
        filename = "unknown"
        
        # Try to get image from file upload
        if 'image' in request.files:
            file = request.files['image']
            
            if file.filename == '':
                logger.warning("No file selected")
                return jsonify({"error": "No file selected"}), 400
            
            # Check file type
            if file.content_type not in ALLOWED_FORMATS:
                logger.warning(f"Invalid file type: {file.content_type}")
                return jsonify({"error": "Only JPG and PNG images are allowed"}), 400
            
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File too large: {file_size} bytes")
                return jsonify({"error": f"File size must be less than {MAX_FILE_SIZE / (1024*1024):.0f}MB"}), 400
            
            # Save temporary file
            temp_dir = Path(__file__).resolve().parent / "temp_uploads"
            temp_dir.mkdir(exist_ok=True)
            
            filename = file.filename
            image_path = temp_dir / filename
            file.save(str(image_path))
            logger.info(f"File uploaded and saved: {image_path}")
        
        # Try to get image from JSON body (image_path)
        elif request.is_json:
            data = request.get_json()
            if 'image_path' in data:
                image_path = Path(data['image_path'])
                filename = image_path.name
                logger.info(f"Using image path from request: {image_path}")
            else:
                logger.warning("No image or image_path provided in request")
                return jsonify({"error": "No image or image_path provided"}), 400
        
        else:
            logger.warning("Invalid request format")
            return jsonify({"error": "Request must contain image file or image_path in JSON"}), 400
        
        # Verify image exists
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return jsonify({"error": f"Image not found: {image_path}"}), 404
        
        # Predict age
        logger.info(f"Starting age prediction for: {image_path}")
        result = predict_age(str(image_path))
        
        # Prepare response
        response = {
            "predicted_age": result['predicted_age'],
            "raw_output": round(result['raw_output'], 2),
            "confidence": result['confidence'],
            "image": filename
        }
        
        logger.info(f"Prediction result: {json.dumps(response)}")
        
        return jsonify(response), 200
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# =====================================================================
# ERROR HANDLERS
# =====================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    logger.warning("File upload exceeded size limit")
    return jsonify({"error": f"File size must be less than {MAX_FILE_SIZE / (1024*1024):.0f}MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

# =====================================================================
# STARTUP & SHUTDOWN
# =====================================================================

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Age Prediction Service Starting")
    logger.info("="*70)
    logger.info(f"Model path: {MODEL_PATH}")
    
    # Load model before starting server (Flask 2.0+ compatible)
    load_model()
    
    logger.info("Service ready to handle requests")
    logger.info("Starting Flask server on http://127.0.0.1:5001")
    logger.info("="*70)
    
    # Start Flask server
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
