import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "submission_model.pth"
TEST_DIR = BASE_DIR / "dataset" / "test"
LOG_DIR = BASE_DIR / "logs"
REPORT_PATH = LOG_DIR / "evaluation_report.txt"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
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


def collect_test_samples():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset folder not found: {TEST_DIR}")

    samples = []
    for age_dir in sorted(TEST_DIR.iterdir()):
        if not age_dir.is_dir():
            continue

        try:
            true_age = int(age_dir.name)
        except ValueError:
            continue

        for image_path in sorted(age_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((image_path, true_age))

    return samples


def main():
    try:
        model = load_model(MODEL_PATH)
        samples = collect_test_samples()

        if not samples:
            raise ValueError(f"No valid images found in {TEST_DIR}")

        y_true = []
        y_pred = []
        errors = []
        with tqdm(total=len(samples), desc="Evaluating submission model") as pbar:
            for image_path, true_age in samples:
                with torch.no_grad():
                    image_tensor = preprocess_image(image_path)
                    prediction = model(image_tensor)
                    predicted_age = float(prediction.item())

                error = abs(float(true_age) - predicted_age)
                y_true.append(float(true_age))
                y_pred.append(predicted_age)
                errors.append(error)
                pbar.update(1)

        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        errors_array = np.array(errors, dtype=np.float64)
        mae = float(np.mean(errors_array))
        rmse = float(np.sqrt(np.mean(np.square(errors_array))))
        median_error = float(np.median(errors_array))

        within_1 = float(np.mean(errors_array <= 1.0) * 100.0)
        within_3 = float(np.mean(errors_array <= 3.0) * 100.0)
        within_5 = float(np.mean(errors_array <= 5.0) * 100.0)
        within_10 = float(np.mean(errors_array <= 10.0) * 100.0)

        print("=" * 80)
        print("SUBMISSION MODEL EVALUATION")
        print("=" * 80)
        print(f"Model: {MODEL_PATH}")
        print(f"Dataset: {TEST_DIR}")
        print(f"Total Images: {len(samples)}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Median Absolute Error: {median_error:.4f}")
        print(f"Within 1 year: {within_1:.2f}%")
        print(f"Within 3 years: {within_3:.2f}%")
        print(f"Within 5 years: {within_5:.2f}%")
        print(f"Within 10 years: {within_10:.2f}%")
        print("=" * 80)

        age_ranges = [
            (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)
        ]

        print("--- Error by age range ---")
        age_range_lines = ["--- Error by age range ---"]
        for lower, upper in age_ranges:
            if upper == 100:
                mask = (y_true >= lower) & (y_true <= upper)
            else:
                mask = (y_true >= lower) & (y_true < upper)

            if not np.any(mask):
                continue

            bucket_true = y_true[mask]
            bucket_pred = y_pred[mask]
            bucket_errors = np.abs(bucket_true - bucket_pred)
            bucket_mae = float(np.mean(bucket_errors))
            bucket_n = int(mask.sum())

            print(f"Age {lower:>3}-{upper:>2} : MAE={bucket_mae:.2f}  (n={bucket_n})")
            age_range_lines.append(f"Age {lower:>3}-{upper:>2} : MAE={bucket_mae:.2f}  (n={bucket_n})")

        print("=" * 80)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        report_lines = [
            "SUBMISSION MODEL EVALUATION",
            "=" * 80,
            f"Model: {MODEL_PATH}",
            f"Dataset: {TEST_DIR}",
            f"Total Images: {len(samples)}",
            f"MAE: {mae:.4f}",
            f"RMSE: {rmse:.4f}",
            f"Median Absolute Error: {median_error:.4f}",
            f"Within 1 year: {within_1:.2f}%",
            f"Within 3 years: {within_3:.2f}%",
            f"Within 5 years: {within_5:.2f}%",
            f"Within 10 years: {within_10:.2f}%",
            "=" * 80,
            *age_range_lines,
            "=" * 80,
        ]

        with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
            report_file.write("\n".join(report_lines) + "\n")

        print(f"Report saved to: {REPORT_PATH}")

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
