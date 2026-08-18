import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "dataset" / "age_prediction" / "test"
MODEL_PATH = MODEL_DIR / "age_resnet50_best.keras"
IMAGE_SIZE = (224, 224)

# -------------------------
# Load model
# -------------------------
print("Loading model...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model loaded successfully!\n")

# -------------------------
# Get age folders and sample images
# -------------------------
age_folders = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
print(f"Found {len(age_folders)} age folders\n")

# Test with 3 images per age folder (random sample)
results = []
for age_folder in age_folders:
    actual_age = int(age_folder.name)
    images_in_folder = list(age_folder.glob("*.jpg")) + list(age_folder.glob("*.png"))
    
    if not images_in_folder:
        continue
    
    # Pick up to 3 random images from this folder
    sample_images = random.sample(images_in_folder, min(3, len(images_in_folder)))
    
    for img_path in sample_images:
        # Load and preprocess image
        img = image.load_img(str(img_path), target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        predicted_age = model.predict(img_array, verbose=0)[0][0]
        predicted_age = float(predicted_age)
        
        # Calculate error
        error = abs(actual_age - predicted_age)
        
        results.append({
            'actual': actual_age,
            'predicted': predicted_age,
            'error': error,
            'image': img_path.name
        })

# -------------------------
# Print results
# -------------------------
print("="*70)
print(f"{'Actual Age':<12} {'Predicted Age':<15} {'Error':<10} {'Image':<30}")
print("="*70)

total_error = 0
for result in results:
    print(f"{result['actual']:<12} {result['predicted']:<15.2f} {result['error']:<10.2f} {result['image']:<30}")
    total_error += result['error']

print("="*70)
avg_error = total_error / len(results) if results else 0
print(f"\nTotal predictions: {len(results)}")
print(f"Mean Absolute Error: {avg_error:.4f}")
print(f"Min Error: {min(r['error'] for r in results):.4f}")
print(f"Max Error: {max(r['error'] for r in results):.4f}")
