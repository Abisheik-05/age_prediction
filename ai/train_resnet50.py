import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "age_prediction"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Data setup
# -------------------------
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Expected training data folders under {DATASET_DIR}. "
        "Create train/ and test/ directories with age-labelled subdirectories."
    )

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# -------------------------
# Custom data generator wrapper to map class indices to ages (001→1, 002→2, ..., 100→100)
# -------------------------
class AgeDataGenerator:
    """Wrapper that adds 1 to class labels to map folder indices to actual ages."""
    def __init__(self, base_generator):
        self.base_generator = base_generator
        self.samples = base_generator.samples
        self.num_classes = base_generator.num_classes
        self.class_indices = base_generator.class_indices
    
    def __iter__(self):
        return self
    
    def __next__(self):
        x, y = next(self.base_generator)
        # Add 1 to labels to map from 0-99 range to 1-100 age range
        y = y + 1
        return x, y
    
    def __len__(self):
        return len(self.base_generator)

train_gen_base = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True,
    seed=SEED,
)

val_gen_base = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False,
    seed=SEED,
)

# Wrap generators to map labels to ages (add 1)
train_generator = AgeDataGenerator(train_gen_base)
val_generator = AgeDataGenerator(val_gen_base)

print("Classes (mapped to ages):", {k: v+1 for k, v in train_gen_base.class_indices.items()})

# -------------------------
# Model creation
# -------------------------
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
outputs = base_model(inputs, training=False)
outputs = layers.GlobalAveragePooling2D()(outputs)
outputs = layers.Dense(256, activation="relu")(outputs)
outputs = layers.Dropout(0.3)(outputs)
outputs = layers.Dense(1, activation="linear")(outputs)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mae",
    metrics=["mae"],
)

model.summary()

# -------------------------
# Training
# -------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "age_resnet50_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ],
)

# Save final model
model.save(MODEL_DIR / "age_resnet50_final.keras")
print("Training complete.")
print(f"Saved best model to: {MODEL_DIR / 'age_resnet50_best.keras'}")
print(f"Saved final model to: {MODEL_DIR / 'age_resnet50_final.keras'}")
