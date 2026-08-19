import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101
from tqdm import tqdm

MODEL_PATH = Path(__file__).resolve().parent / "models" / "age_resnet101_finetuned_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_model():
    model = resnet101(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return model.to(DEVICE)


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = build_model()
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format in {model_path}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def list_images(folder: Path):
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return images


def main():
    if len(sys.argv) != 2:
        print("Usage: python ai/predict_folder.py path_to_folder")
        raise SystemExit(1)

    folder = Path(sys.argv[1])

    try:
        model = load_model(MODEL_PATH)
        image_paths = list_images(folder)

        if not image_paths:
            raise ValueError(f"No supported images found in {folder}")

        output_rows = []
        for image_path in tqdm(image_paths, desc="Predicting ages"):
            with torch.no_grad():
                image_tensor = preprocess_image(image_path)
                prediction = model(image_tensor)
                predicted_age = float(prediction.item())

            output_rows.append((image_path.name, f"{predicted_age:.2f}"))

        csv_path = folder / "predictions.csv"
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["image_name", "predicted_age"])
            writer.writerows(output_rows)

        print(f"Saved predictions to: {csv_path}")

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
