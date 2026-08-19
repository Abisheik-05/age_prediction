from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from ..config import settings
from ..face_detection import FaceDetector
from ..schemas import PredictionResponse

FACE_DETECTOR = FaceDetector()

router = APIRouter()
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _preprocess_image_bytes(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    """Convert uploaded bytes into a normalized tensor ready for inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    cropped_face = FACE_DETECTOR.crop_largest_face(image)
    if cropped_face is None:
        raise ValueError("No face detected")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform(cropped_face).unsqueeze(0).to(device)


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_age(request: Request, image: UploadFile) -> PredictionResponse:
    """Predict a person's age from an uploaded facial image."""
    if image.filename is None or image.filename == "":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image file provided.")

    file_extension = Path(image.filename).suffix.lower()
    if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Allowed formats: jpg, jpeg, png",
        )

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded image exceeds the 5 MB size limit.",
        )

    temp_path: Path | None = None
    try:
        temp_dir = settings.uploads_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=file_extension) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        device_name = str(request.app.state.device)
        try:
            image_tensor = _preprocess_image_bytes(image_bytes, torch.device(device_name))
        except ValueError as exc:
            if str(exc) == "No face detected":
                return JSONResponse(status_code=400, content={"error": "No face detected"})
            raise

        model: torch.nn.Module = request.app.state.model
        start_time = time.perf_counter()
        with torch.no_grad():
            prediction = model(image_tensor)
        inference_time = time.perf_counter() - start_time

        raw_age = float(prediction.item())
        rounded_age = int(round(raw_age))
        clamped_age = max(0, min(100, rounded_age))

        confidence = max(0, min(100, int(round(100.0 - abs(raw_age - clamped_age) * 10.0))))
        if confidence == 0 and clamped_age >= 0:
            confidence = 1

        return PredictionResponse(
            age=clamped_age,
            confidence=confidence,
            model="ResNet101",
            inference_time=round(inference_time, 2),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
