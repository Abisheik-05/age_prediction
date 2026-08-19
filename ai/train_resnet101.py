import csv
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet101_Weights, resnet101
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
IMAGE_SIZE = 224
SEED = 42
PATIENCE = 5
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5
CURRENT_BEST_VAL_MAE = 5.2812
BASE_BEST_CHECKPOINT = MODEL_DIR / "age_resnet101_best.pth"
FINE_TUNE_BEST_CHECKPOINT = MODEL_DIR / "age_resnet101_finetuned_best.pth"
FINE_TUNE_FINAL_CHECKPOINT = MODEL_DIR / "age_resnet101_finetuned_final.pth"
SUBMISSION_MODEL_PATH = MODEL_DIR / "submission_model.pth"
CSV_LOG_PATH = LOG_DIR / "training_log.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
NUM_WORKERS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

# -------------------------
# Utility helpers
# -------------------------
def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return 0.0, 0.0, "CPU"

    free_memory, total_memory = torch.cuda.mem_get_info()
    gpu_name = torch.cuda.get_device_name(0)
    return free_memory / (1024 ** 3), total_memory / (1024 ** 3), gpu_name


def load_checkpoint_state_dict(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if any(k.startswith("layer") or k.startswith("conv") or k.startswith("fc") for k in checkpoint.keys()):
            return checkpoint

    raise ValueError(f"Unsupported checkpoint format for {path}")


def predict_with_tta(model, image_tensor, device):
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        original_pred = model(image_tensor)
        flipped_pred = model(torch.flip(image_tensor, dims=[-1]))
        average_pred = (original_pred + flipped_pred) / 2.0

    return average_pred.squeeze(0)


# -------------------------
# Custom age dataset
# -------------------------
class AgeFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root_dir}")

        for age_dir in sorted(self.root_dir.iterdir()):
            if not age_dir.is_dir():
                continue

            try:
                age = int(age_dir.name)
            except ValueError:
                continue

            for image_path in sorted(age_dir.iterdir()):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((image_path, age))

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, age = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, float(age)


def main():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Expected dataset directories. Looking for {TRAIN_DIR} and {VAL_DIR}."
        )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AgeFolderDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = AgeFolderDataset(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
    )

    print("=" * 80)
    print("RESNET101 AGE REGRESSION FINE-TUNING")
    print("=" * 80)
    free_vram_gb, total_vram_gb, gpu_name = get_gpu_memory_info()
    if torch.cuda.is_available():
        print(f"GPU name: {gpu_name}")
        print(f"Available VRAM: {free_vram_gb:.2f} GB")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")
    else:
        print("GPU name: CPU")
        print("Available VRAM: N/A")
        print("Total VRAM: N/A")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Fine-tune epochs: {FINE_TUNE_EPOCHS}")
    print(f"Learning rate: {FINE_TUNE_LR}")
    print(f"Resume checkpoint: {BASE_BEST_CHECKPOINT}")
    print("=" * 80)

    try:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    except Exception:
        model = resnet101(weights="IMAGENET1K_V2")

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    model = model.to(DEVICE)

    checkpoint_state = load_checkpoint_state_dict(BASE_BEST_CHECKPOINT)
    model.load_state_dict(checkpoint_state, strict=True)
    print(f"Loaded fine-tuning checkpoint: {BASE_BEST_CHECKPOINT}")

    model.train()
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_mae = CURRENT_BEST_VAL_MAE
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.time()
    first_epoch_time = None

    csv_header_written = CSV_LOG_PATH.exists()
    with open(CSV_LOG_PATH, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not csv_header_written:
            writer.writerow(["epoch", "train_loss", "val_loss", "train_mae", "val_mae", "learning_rate"])

        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            train_mae_total = 0.0
            train_steps = 0

            progress = tqdm(train_loader, desc=f"Fine-tune epoch {epoch}/{FINE_TUNE_EPOCHS} [train]", leave=True)
            for images, ages in progress:
                images = images.to(DEVICE, non_blocking=PIN_MEMORY)
                ages = ages.to(DEVICE, non_blocking=PIN_MEMORY).view(-1, 1)

                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, ages)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    batch_mae = torch.mean(torch.abs(outputs - ages)).item()

                batch_size = images.size(0)
                train_loss_total += loss.item() * batch_size
                train_mae_total += batch_mae * batch_size
                train_steps += batch_size
                progress.set_postfix({"loss": loss.item(), "mae": batch_mae})

            train_loss = train_loss_total / train_steps
            train_mae = train_mae_total / train_steps

            model.eval()
            val_loss_total = 0.0
            val_mae_total = 0.0
            val_steps = 0

            with torch.no_grad():
                for images, ages in tqdm(val_loader, desc=f"Fine-tune epoch {epoch}/{FINE_TUNE_EPOCHS} [val]", leave=False):
                    images = images.to(DEVICE, non_blocking=PIN_MEMORY)
                    ages = ages.to(DEVICE, non_blocking=PIN_MEMORY).view(-1, 1)

                    with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                        outputs = model(images)
                        loss = criterion(outputs, ages)

                    batch_size = images.size(0)
                    val_loss_total += loss.item() * batch_size
                    val_mae_total += torch.mean(torch.abs(outputs - ages)).item() * batch_size
                    val_steps += batch_size

            val_loss = val_loss_total / val_steps
            val_mae = val_mae_total / val_steps

            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_elapsed = time.time() - epoch_start
            if first_epoch_time is None:
                first_epoch_time = epoch_elapsed

            print(
                f"Epoch {epoch}/{FINE_TUNE_EPOCHS} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_mae={train_mae:.4f} | val_mae={val_mae:.4f} | "
                f"lr={current_lr:.6f} | epoch_time={epoch_elapsed:.1f}s"
            )

            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(train_mae, 6),
                round(val_mae, 6),
                round(current_lr, 8),
            ])
            csv_file.flush()

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), FINE_TUNE_BEST_CHECKPOINT)
                shutil.copy2(FINE_TUNE_BEST_CHECKPOINT, SUBMISSION_MODEL_PATH)
                print(f"Saved improved checkpoint: {FINE_TUNE_BEST_CHECKPOINT}")
                print(f"Saved submission model copy: {SUBMISSION_MODEL_PATH}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best validation MAE: {best_val_mae:.4f}")
                break

    total_training_time = time.time() - training_start
    torch.save(model.state_dict(), FINE_TUNE_FINAL_CHECKPOINT)

    print("=" * 80)
    print("Fine-tuning complete.")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Total training time: {total_training_time / 60:.2f} minutes")
    print(f"Best checkpoint: {FINE_TUNE_BEST_CHECKPOINT}")
    print(f"Final checkpoint: {FINE_TUNE_FINAL_CHECKPOINT}")
    print(f"Submission model: {SUBMISSION_MODEL_PATH}")
    print(f"CSV log: {CSV_LOG_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
