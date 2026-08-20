from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from pathlib import Path
from uuid import uuid4

from fastapi import File, Form, UploadFile

from database import users_collection, feedback_collection
from ..schemas import HistoryEntry, HistorySaveRequest

router = APIRouter()


@router.post("/history/save", response_model=HistoryEntry, status_code=status.HTTP_201_CREATED)
async def save_history_entry(payload: HistorySaveRequest) -> HistoryEntry:
    """Save a prediction record to MongoDB using the existing Motor collection."""
    doc = {
        "image_name": payload.image_name,
        "predicted_age": payload.predicted_age,
        "confidence": payload.confidence,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        result = await users_collection.insert_one(doc)
    except Exception as exc:  # pragma: no cover - connection errors handled at app startup
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save prediction history: {exc}",
        ) from exc

    saved = await users_collection.find_one({"_id": result.inserted_id})
    if saved is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction history was not persisted.",
        )

    return HistoryEntry(
        image_name=saved["image_name"],
        predicted_age=saved["predicted_age"],
        confidence=saved["confidence"],
        created_at=saved["created_at"],
    )


@router.get("/history", response_model=list[HistoryEntry])
async def list_history_entries() -> list[HistoryEntry]:
    """Return all prediction records sorted by newest first."""
    try:
        cursor = users_collection.find().sort("created_at", -1)
        entries = await cursor.to_list(length=None)
    except Exception as exc:  # pragma: no cover - connection errors handled at app startup
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch prediction history: {exc}",
        ) from exc

    return [
        HistoryEntry(
            image_name=item["image_name"],
            predicted_age=item["predicted_age"],
            confidence=item["confidence"],
            created_at=item["created_at"],
        )
        for item in entries
    ]
@router.get("/history/stats")
async def get_history_stats():
    try:
        entries = await users_collection.find().to_list(length=None)

        total_predictions = len(entries)

        if total_predictions == 0:
            return {
                "total_predictions": 0,
                "average_age": 0,
                "average_confidence": 0
            }

        average_age = round(
            sum(item["predicted_age"] for item in entries) / total_predictions,
            1
        )

        average_confidence = round(
            sum(item["confidence"] for item in entries) / total_predictions,
            1
        )

        return {
            "total_predictions": total_predictions,
            "average_age": average_age,
            "average_confidence": average_confidence
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {exc}",
        ) from exc

@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def save_feedback(
    image: UploadFile = File(...),
    predicted_age: int = Form(...),
    actual_age: int = Form(...),
):
    try:
        # Base dataset folder
        dataset_root = Path("NewData")

        # Create age folder if it doesn't exist
        age_folder = dataset_root / str(actual_age)
        age_folder.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        extension = Path(image.filename).suffix or ".jpg"
        filename = f"{uuid4()}{extension}"

        image_path = age_folder / filename

        # Save image
        contents = await image.read()

        with open(image_path, "wb") as f:
            f.write(contents)

        # Optional: save metadata in MongoDB
        feedback_doc = {
            "image_name": filename,
            "predicted_age": predicted_age,
            "actual_age": actual_age,
            "saved_path": str(image_path),
            "created_at": datetime.now(timezone.utc),
        }

        await feedback_collection.insert_one(feedback_doc)

        return {
            "message": "Feedback saved successfully",
            "filename": filename,
            "folder": str(age_folder),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save feedback: {exc}",
        ) from exc