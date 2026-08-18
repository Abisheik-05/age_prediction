"""
Test script for Age Prediction API
Tests the Flask backend at http://127.0.0.1:5001/predict-age
"""

import requests
import json
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================
SERVICE_URL = "http://127.0.0.1:5001"
PREDICT_ENDPOINT = f"{SERVICE_URL}/predict-age"

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
TEST_IMAGE_DIR = BACKEND_DIR.parent / "ai" / "dataset" / "age_prediction" / "test"

# =====================================================================
# TEST FUNCTIONS
# =====================================================================

def test_health_check():
    """Test if service is running."""
    logger.info("Testing health check...")
    try:
        response = requests.get(f"{SERVICE_URL}/", timeout=5)
        logger.info(f"Health check response: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

def test_prediction_with_file():
    """Test age prediction with file upload."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Prediction with File Upload")
    logger.info("="*70)
    
    # Find a test image
    test_image = None
    for age_folder in TEST_IMAGE_DIR.iterdir():
        if age_folder.is_dir():
            images = list(age_folder.glob("*.jpg")) + list(age_folder.glob("*.png"))
            if images:
                test_image = images[0]
                actual_age = int(age_folder.name)
                break
    
    if not test_image:
        logger.error("No test images found in dataset")
        return False
    
    logger.info(f"Using test image: {test_image}")
    logger.info(f"Actual age (folder): {actual_age}")
    
    try:
        # Upload image
        with open(str(test_image), 'rb') as f:
            files = {'image': f}
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Prediction result:")
            logger.info(f"  Predicted Age: {result['predicted_age']}")
            logger.info(f"  Raw Output: {result['raw_output']}")
            logger.info(f"  Confidence: {result['confidence']}")
            logger.info(f"  Image: {result['image']}")
            
            error = abs(result['predicted_age'] - actual_age)
            logger.info(f"  Error: {error} years")
            
            return True
        else:
            logger.error(f"Prediction failed: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def test_prediction_with_path():
    """Test age prediction with image path in JSON."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Prediction with Image Path (JSON)")
    logger.info("="*70)
    
    # Find a test image
    test_image = None
    for age_folder in TEST_IMAGE_DIR.iterdir():
        if age_folder.is_dir():
            images = list(age_folder.glob("*.jpg")) + list(age_folder.glob("*.png"))
            if images:
                test_image = images[1] if len(images) > 1 else images[0]
                actual_age = int(age_folder.name)
                break
    
    if not test_image:
        logger.error("No test images found in dataset")
        return False
    
    logger.info(f"Using test image: {test_image}")
    logger.info(f"Actual age (folder): {actual_age}")
    
    try:
        # Send image path in JSON
        payload = {"image_path": str(test_image)}
        response = requests.post(
            PREDICT_ENDPOINT,
            json=payload,
            timeout=30
        )
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Prediction result:")
            logger.info(f"  Predicted Age: {result['predicted_age']}")
            logger.info(f"  Raw Output: {result['raw_output']}")
            logger.info(f"  Confidence: {result['confidence']}")
            
            error = abs(result['predicted_age'] - actual_age)
            logger.info(f"  Error: {error} years")
            
            return True
        else:
            logger.error(f"Prediction failed: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Error Handling")
    logger.info("="*70)
    
    tests_passed = 0
    
    # Test 1: No file provided
    logger.info("\nTest 1: No file provided")
    try:
        response = requests.post(PREDICT_ENDPOINT, timeout=5)
        if response.status_code != 200:
            logger.info(f"✓ Correctly returned error: {response.status_code}")
            logger.info(f"  Response: {response.json()}")
            tests_passed += 1
        else:
            logger.error("✗ Should have returned an error")
    except Exception as e:
        logger.error(f"✗ Test failed: {str(e)}")
    
    # Test 2: Invalid file path
    logger.info("\nTest 2: Invalid file path")
    try:
        payload = {"image_path": "/nonexistent/path/image.jpg"}
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=5)
        if response.status_code != 200:
            logger.info(f"✓ Correctly returned error: {response.status_code}")
            logger.info(f"  Response: {response.json()}")
            tests_passed += 1
        else:
            logger.error("✗ Should have returned an error")
    except Exception as e:
        logger.error(f"✗ Test failed: {str(e)}")
    
    return tests_passed == 2

def run_batch_test(num_predictions=10):
    """Test multiple predictions to verify consistency."""
    logger.info("\n" + "="*70)
    logger.info(f"TEST: Batch Predictions ({num_predictions} images)")
    logger.info("="*70)
    
    predictions = []
    errors = []
    
    # Collect random images from different age folders
    image_list = []
    age_folders = sorted([d for d in TEST_IMAGE_DIR.iterdir() if d.is_dir()])[:num_predictions]
    
    for age_folder in age_folders:
        images = list(age_folder.glob("*.jpg")) + list(age_folder.glob("*.png"))
        if images:
            actual_age = int(age_folder.name)
            image_list.append((images[0], actual_age))
    
    logger.info(f"Testing {len(image_list)} images...")
    
    for test_image, actual_age in image_list:
        try:
            with open(str(test_image), 'rb') as f:
                files = {'image': f}
                response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                pred_age = result['predicted_age']
                error = abs(pred_age - actual_age)
                predictions.append(pred_age)
                errors.append(error)
                
                logger.info(f"Age {actual_age:3d}: Predicted {pred_age:3d}, Error: {error:5.2f} years")
            else:
                logger.error(f"Prediction failed for {test_image}: {response.text}")
        
        except Exception as e:
            logger.error(f"Error predicting {test_image}: {str(e)}")
    
    if errors:
        logger.info("\n" + "="*70)
        logger.info("BATCH TEST SUMMARY")
        logger.info("="*70)
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info(f"Mean Absolute Error: {sum(errors) / len(errors):.2f} years")
        logger.info(f"Min Error: {min(errors):.2f} years")
        logger.info(f"Max Error: {max(errors):.2f} years")
        logger.info(f"Std Dev: {np.std(errors):.2f} years" if len(errors) > 1 else "")
        return True
    
    return False

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import numpy as np
    
    logger.info("="*70)
    logger.info("AGE PREDICTION API TEST SUITE")
    logger.info("="*70)
    logger.info(f"Service URL: {SERVICE_URL}")
    logger.info(f"Predict Endpoint: {PREDICT_ENDPOINT}")
    logger.info(f"Test Image Dir: {TEST_IMAGE_DIR}")
    
    # Run tests
    all_passed = True
    
    # Health check
    if not test_health_check():
        logger.error("\n✗ Health check failed - service may not be running")
        logger.error("Start the service with: python age_prediction_service.py")
        sys.exit(1)
    else:
        logger.info("✓ Health check passed - service is running\n")
    
    # Individual prediction tests
    if test_prediction_with_file():
        logger.info("✓ File upload prediction test passed")
    else:
        logger.error("✗ File upload prediction test failed")
        all_passed = False
    
    if test_prediction_with_path():
        logger.info("✓ JSON path prediction test passed")
    else:
        logger.error("✗ JSON path prediction test failed")
        all_passed = False
    
    # Error handling
    if test_error_handling():
        logger.info("✓ Error handling tests passed")
    else:
        logger.error("✗ Error handling tests failed")
        all_passed = False
    
    # Batch test
    if run_batch_test(num_predictions=10):
        logger.info("✓ Batch prediction test passed")
    else:
        logger.error("✗ Batch prediction test failed")
        all_passed = False
    
    # Summary
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.error("✗ SOME TESTS FAILED")
    logger.info("="*70)
    
    sys.exit(0 if all_passed else 1)
