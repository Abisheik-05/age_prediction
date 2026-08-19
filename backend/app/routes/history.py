from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from database import users_collection
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
