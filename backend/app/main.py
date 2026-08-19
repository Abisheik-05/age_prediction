from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from database import client
from .config import detect_device, settings
from .model_loader import load_model
from .routes.health import router as health_router
from .routes.history import router as history_router
from .routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once during application startup."""
    device_name = detect_device()
    device = torch.device(device_name)
    app.state.device = device_name

    try:
        await client.admin.command("ping")
        print("MongoDB connection verified")
    except Exception as exc:
        raise RuntimeError(f"MongoDB connection failed: {exc}") from exc

    print("Loading model...")
    model, _ = load_model(settings.model_path, device)
    app.state.model = model
    print("Model loaded successfully")
    print(f"Device: {device_name}")
    yield
    app.state.model = None


app = FastAPI(
    title=settings.app_title,
    version=settings.version,
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(history_router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Chrono AI backend is running"}
