from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["online"] = Field(..., description="API service status")
    model: Literal["ResNet101"] = Field(..., description="Model architecture in use")
    device: Literal["cuda", "cpu"] = Field(..., description="Active device used by PyTorch")


class PredictionResponse(BaseModel):
    age: int = Field(..., ge=0, le=100, description="Predicted age in years")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score in percent")
    model: Literal["ResNet101"] = Field(default="ResNet101", description="Model architecture")
    inference_time: float = Field(..., ge=0.0, description="Inference time in seconds")


class HistorySaveRequest(BaseModel):
    image_name: str = Field(..., min_length=1, max_length=255, description="Uploaded image name")
    predicted_age: int = Field(..., ge=0, le=100, description="Predicted age")
    confidence: int = Field(..., ge=0, le=100, description="Prediction confidence percent")


class HistoryEntry(BaseModel):
    image_name: str = Field(..., description="Uploaded image name")
    predicted_age: int = Field(..., ge=0, le=100, description="Predicted age")
    confidence: int = Field(..., ge=0, le=100, description="Prediction confidence percent")
    created_at: datetime = Field(..., description="UTC creation timestamp")
