# ResNet50 Age Prediction API Integration Guide

## Overview

This guide explains how to integrate the trained ResNet50 age prediction model into the backend and replace the DeepFace-based age prediction flow.

### Model Specifications
- **Model**: age_resnet50_best.keras
- **Architecture**: ResNet50 backbone (frozen) + regression head
- **Input**: 224×224 RGB images
- **Output**: Age (1–100)
- **Validation MAE**: 8.0992 years
- **Framework**: TensorFlow 2.x with Keras API

---

## Project Structure

```
age_prediction/
├── ai/
│   ├── models/
│   │   └── age_resnet50_best.keras        ← Trained model (loaded by service)
│   ├── dataset/
│   │   └── age_prediction/
│   │       ├── train/
│   │       └── test/
│   ├── train_resnet50.py                  ← Training script
│   └── test_inference.py                  ← Inference validation
│
├── backend/
│   ├── age_prediction_service.py          ← NEW: Flask service (Python)
│   ├── test_api.py                        ← NEW: API test script (Python)
│   ├── server_new.js                      ← NEW: Updated Node.js backend
│   ├── server.js                          ← OLD: Original DeepFace backend
│   ├── package.json                       ← Node dependencies
│   ├── uploads/                           ← Uploaded images
│   └── temp_uploads/                      ← Temporary image uploads (created by service)
│
├── frontend/
│   └── (unchanged - works with both backends)
│
└── README.md
```

---

## Architecture

### Current System (DeepFace)
```
Frontend (React)
    ↓ POST /api/predict [image]
Node.js Backend (Express) :5000
    ↓ POST http://127.0.0.1:5001/predict-age [image]
Flask Backend (DeepFace) :5001
    ↓
External DeepFace Analysis
    ↓ age
Node.js Backend
    ↓ {age, confidence}
Frontend
```

### New System (ResNet50)
```
Frontend (React)
    ↓ POST /api/predict [image]
Node.js Backend (Express) :5000
    ↓ POST http://127.0.0.1:5001/predict-age [image]
Flask Backend (ResNet50 Service) :5001
    ↓
Loaded Keras Model: age_resnet50_best.keras
    ↓ predicted_age
Flask Backend
    ↓ {predicted_age, confidence}
Node.js Backend
    ↓ {age, confidence}
Frontend
```

---

## Installation & Setup

### Step 1: Install Python Dependencies

The age prediction service requires TensorFlow and Flask.

```bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install tensorflow flask
# OR if using the existing ai-env virtual environment:
..\ai-env\Scripts\pip install flask

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import flask; print(flask.__version__)"
```

### Step 2: Verify Model Path

Ensure the trained model exists at:
```
ai/models/age_resnet50_best.keras
```

If not, copy it from the training output directory.

### Step 3: Update Node.js Backend

**Option A: Replace entire server.js** (Recommended)
```bash
# Backup original
cp backend/server.js backend/server_deepface_backup.js

# Use new version
cp backend/server_new.js backend/server.js
```

**Option B: Merge changes** (If you've customized server.js)
```javascript
// In server.js, replace the /api/predict endpoint with the new implementation
// See server_new.js for the new endpoint code
```

### Step 4: Install Additional Node Dependencies

The updated server.js requires one additional package:

```bash
cd backend
npm install form-data
```

Update `package.json` to include:
```json
{
  "dependencies": {
    "axios": "^1.19.0",
    "cors": "^2.8.6",
    "express": "^5.2.1",
    "form-data": "^4.0.0",
    "multer": "^2.2.0"
  }
}
```

---

## Running the System

### Terminal 1: Start Python Age Prediction Service

```bash
cd backend
python age_prediction_service.py
```

Expected output:
```
2026-08-18 12:34:56,789 - __main__ - INFO - ======================================================================
2026-08-18 12:34:56,789 - __main__ - INFO - Age Prediction Service Starting
2026-08-18 12:34:56,789 - __main__ - INFO - ======================================================================
2026-08-18 12:34:56,789 - __main__ - INFO - Loading model from: .../ai/models/age_resnet50_best.keras
2026-08-18 12:35:02,123 - __main__ - INFO - Model loaded successfully!
2026-08-18 12:35:02,123 - __main__ - INFO - Service ready to handle requests
 * Running on http://127.0.0.1:5001
```

The service loads the model **once** at startup and keeps it in memory for fast inference.

### Terminal 2: Start Node.js Backend

```bash
cd backend
node server.js
```

Expected output:
```
======================================================================
AGE PREDICTION BACKEND STARTED
======================================================================
Server running on port 5000

API Endpoints:
  GET  /                        - Health check
  GET  /api/health/service      - Check Python service status
  POST /api/predict             - Predict age from image

IMPORTANT: Start the Python age prediction service:
  python backend/age_prediction_service.py
```

### Terminal 3: Start Frontend (React)

```bash
cd frontend
npm install  # if not already done
npm run dev
```

The frontend remains **unchanged** and works with both the old DeepFace and new ResNet50 backends.

---

## Testing

### Quick Test: Check Service Status

```bash
# Check if Flask service is running
curl http://127.0.0.1:5001/

# Check if Node backend is running
curl http://127.0.0.1:5000/

# Check if Node backend can reach Flask service
curl http://127.0.0.1:5000/api/health/service
```

### Comprehensive Test Suite

Run the automated API test suite:

```bash
cd backend
python test_api.py
```

This tests:
1. ✓ Health check (service is running)
2. ✓ Prediction with file upload
3. ✓ Prediction with image path (JSON)
4. ✓ Error handling (missing file, invalid path)
5. ✓ Batch predictions (10 images with error metrics)

Expected output:
```
======================================================================
AGE PREDICTION API TEST SUITE
======================================================================
Service URL: http://127.0.0.1:5001
Predict Endpoint: http://127.0.0.1:5001/predict-age
Test Image Dir: .../ai/dataset/age_prediction/test

✓ Health check passed - service is running

✓ File upload prediction test passed
✓ JSON path prediction test passed
✓ Error handling tests passed
✓ Batch prediction test passed

======================================================================
✓ ALL TESTS PASSED
======================================================================
```

### Manual Test: Using curl

```bash
# Predict age from uploaded image
curl -X POST \
  -F "image=@path/to/image.jpg" \
  http://127.0.0.1:5000/api/predict

# Expected response:
# {
#   "age": 24,
#   "confidence": 100,
#   "image": "1692360000000-image.jpg",
#   "raw_output": 24.35,
#   "model": "ResNet50 (regression)"
# }
```

### Manual Test: Using Python

```python
import requests

# Upload image and get prediction
with open("test_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post("http://127.0.0.1:5000/api/predict", files=files)
    print(response.json())
```

---

## API Endpoints

### POST /api/predict
**Predict age from uploaded image**

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: form field "image" (JPG or PNG file)

**Response (Success - 200):**
```json
{
  "age": 24,
  "confidence": 100,
  "image": "1692360000000-image.jpg",
  "raw_output": 24.35,
  "model": "ResNet50 (regression)"
}
```

**Response (Error - 4xx/5xx):**
```json
{
  "message": "Age prediction failed",
  "error": "Error description"
}
```

**Possible Error Codes:**
- `400 Bad Request` - No image provided or invalid file type
- `413 Payload Too Large` - File exceeds 5MB
- `500 Internal Server Error` - Model inference failed or service unavailable

### GET /
**Backend health check**

**Response:**
```json
{
  "status": "OK"
}
```

### GET /api/health/service
**Check Python Flask service status**

**Response (Success - 200):**
```json
{
  "status": "OK",
  "service": "age_prediction_service",
  "url": "http://127.0.0.1:5001"
}
```

**Response (Error - 503):**
```json
{
  "status": "ERROR",
  "service": "age_prediction_service",
  "url": "http://127.0.0.1:5001",
  "error": "Connection refused",
  "hint": "Make sure to start the Python Flask service: python backend/age_prediction_service.py"
}
```

---

## Preprocessing Pipeline

The service preprocesses images **exactly** as done during training:

### Training Preprocessing (train_resnet50.py)
```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ResNet50 ImageNet normalization
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Images loaded with target_size=(224, 224)
```

### Inference Preprocessing (age_prediction_service.py)
```python
# Load image
img = image.load_img(image_path, target_size=(224, 224))

# Convert to array (0-255 range)
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Apply ResNet50 preprocessing (ImageNet normalization)
img_array = preprocess_input(img_array)

# Run inference
predicted_age = model.predict(img_array)[0][0]

# Clamp to valid range [1, 100]
predicted_age = max(1, min(100, predicted_age))
```

**Key Points:**
- Image size: 224×224
- Pixel range before preprocessing: 0-255
- Preprocessing: ResNet50 ImageNet normalization (same as training)
- Output: Unclamped float, then clamped to [1, 100] and rounded to int
- No additional normalization after preprocess_input

---

## Model Output Handling

### Raw Model Output
- Type: Float32 scalar
- Range: Can be negative or >100 (unconstrained)
- Example: -2.38, 24.35, 103.47

### Clamping Strategy
```python
# Clamp to valid age range
predicted_age = max(VALID_AGE_RANGE[0], min(VALID_AGE_RANGE[1], raw_output))
predicted_age = int(round(predicted_age))

# Examples:
# -2.38  → 1
# 24.35  → 24
# 103.47 → 100
```

This ensures predictions are always in the range [1, 100].

---

## Performance Characteristics

### Inference Time
- First prediction: ~3-5 seconds (model initialization)
- Subsequent predictions: ~0.5-1 second (GPU accelerated via DirectML)

### Memory Usage
- Model size: ~100 MB
- Runtime memory: ~500 MB - 1 GB (with batch processing)

### Accuracy
- Validation MAE: 8.10 years (on full test set)
- Best for ages 25-50 (typically 3-10 years error)
- Higher errors at extremes (very young/very old)

---

## Logging

### Python Service Logs
Located at: `backend/age_prediction_service.log`

Sample log entries:
```
2026-08-18 12:35:02,789 - __main__ - INFO - Loading model from: .../ai/models/age_resnet50_best.keras
2026-08-18 12:35:08,123 - __main__ - INFO - Model loaded successfully!
2026-08-18 12:35:15,456 - __main__ - INFO - Starting age prediction for: /path/to/image.jpg
2026-08-18 12:35:16,789 - __main__ - INFO - Prediction complete: raw_output=24.35, clamped_age=24
2026-08-18 12:35:16,812 - __main__ - INFO - Prediction result: {"predicted_age": 24, "raw_output": 24.35, "confidence": 1.0, "image": "test.jpg"}
```

### Node.js Backend Logs
Printed to console:

```
================================
Image Prediction Request Received
================================
Image Details:
  Filename: image.jpg
  Path: .../backend/uploads/1692360000000-image.jpg
  Size: 12345 bytes
  MIME Type: image/jpeg

Calling age prediction service...
Prediction Result:
  Predicted Age: 24
  Raw Output: 24.35
  Confidence: 100

================================
Response sent successfully
================================
```

---

## Migration from DeepFace

### What Changes
1. **Backend service**: Replaces Flask DeepFace with Flask ResNet50
2. **API endpoint remains the same**: POST /api/predict
3. **Frontend remains unchanged**: No code changes needed

### What Stays the Same
- **Frontend UI**: No changes
- **API request format**: Same multipart form data
- **API response format**: Same JSON structure
- **Response keys**: "age", "confidence", "image"

### Smooth Migration Checklist
- [ ] Backup original `server.js`
- [ ] Install Python dependencies (TensorFlow, Flask)
- [ ] Install Node.js dependency (form-data)
- [ ] Verify model path: `ai/models/age_resnet50_best.keras`
- [ ] Start Python Flask service: `python backend/age_prediction_service.py`
- [ ] Start Node.js backend: `node backend/server.js`
- [ ] Run test suite: `python backend/test_api.py`
- [ ] Test frontend: Upload image and verify prediction

### Rollback to DeepFace
If needed, you can quickly rollback:

```bash
# Restore original server.js
cp backend/server_deepface_backup.js backend/server.js

# Restart Node backend
node backend/server.js

# Start old DeepFace service (if still installed)
python backend/deepface_service.py  # or similar
```

---

## Troubleshooting

### Issue: "Cannot connect to age prediction service"
**Cause**: Python Flask service not running

**Solution**:
```bash
cd backend
python age_prediction_service.py
```

### Issue: "Model not found at ..."
**Cause**: Model file missing or path incorrect

**Solution**:
1. Verify file exists: `ai/models/age_resnet50_best.keras`
2. Check path in `age_prediction_service.py`: MODEL_PATH
3. Re-train if necessary: `python ai/train_resnet50.py`

### Issue: "Only JPG and PNG images are allowed"
**Cause**: Uploading unsupported format

**Solution**: Use JPG or PNG only

### Issue: Prediction time very slow (>10 seconds)
**Cause**: Model inference slow, possibly CPU-only

**Solution**:
1. Check if GPU is being used (DirectML output in logs)
2. Restart service to reinitialize GPU
3. Check GPU memory availability

### Issue: OutOfMemory error
**Cause**: Not enough GPU or CPU memory

**Solution**:
1. Close other applications
2. Reduce batch size (in training only)
3. Upgrade hardware

---

## Production Deployment

### Recommended Setup for Production

```
Load Balancer (nginx)
    ↓
Node.js Backend (Express) - Multiple instances
    ↓
Python Flask Service - Single instance (model loaded once)
    ↓
GPU Server (RTX 4050 or better)
```

### Environment Variables

Add to `backend/.env`:
```
FLASK_SERVICE_URL=http://127.0.0.1:5001
FLASK_SERVICE_TIMEOUT=60
NODE_ENV=production
```

Load with:
```javascript
require('dotenv').config();
const FLASK_SERVICE_URL = process.env.FLASK_SERVICE_URL || "http://127.0.0.1:5001";
```

### Process Manager

Use PM2 to manage processes:

```bash
# Install PM2
npm install -g pm2

# Start processes
pm2 start "python backend/age_prediction_service.py" --name "age-service"
pm2 start "node backend/server.js" --name "age-backend"

# Monitor
pm2 monit

# Logs
pm2 logs
```

---

## Files Summary

| File | Purpose | Language | Location |
|------|---------|----------|----------|
| `age_prediction_service.py` | Main Flask service (loads model, runs inference) | Python | `backend/` |
| `test_api.py` | Automated test suite | Python | `backend/` |
| `server_new.js` | Updated Node.js backend (replaces server.js) | JavaScript | `backend/` |
| `age_resnet50_best.keras` | Trained model (binary) | Model | `ai/models/` |

---

## References

- TensorFlow Keras: https://www.tensorflow.org/api_docs/python/tf/keras
- Flask: https://flask.palletsprojects.com/
- Express.js: https://expressjs.com/
- ResNet50 Architecture: https://arxiv.org/abs/1512.03385

---

## Questions & Support

For issues or questions:
1. Check logs: `backend/age_prediction_service.log` and console output
2. Run test suite: `python backend/test_api.py`
3. Verify service health: `curl http://127.0.0.1:5001/`
4. Check model path and file size
