from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(request: Request) -> HealthResponse:
    device: str = getattr(request.app.state, "device", "cpu")
    return HealthResponse(
        status="online",
        model="ResNet101",
        device=device,
    )
