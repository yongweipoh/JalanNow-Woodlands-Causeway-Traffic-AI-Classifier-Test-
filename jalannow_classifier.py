"""
Woodlands-JB Traffic Camera — Keras Multi-Output Classification & Prediction
=========================================================================
Trains a MobileNetV2-based multi-output model to classify:
  1) Crowd density at Woodlands Checkpoint (before/after customs)
  2) Crowd activity — walking on causeway vs. inside checkpoint area
  3) Weather conditions affecting pedestrian/vehicle traffic

Fetches the LATEST live image from data.gov.sg API and generates
a comprehensive prediction report as a .txt file.

Designed to work with images collected by the Woodlands-JB Traffic Camera
Image Extractor (API-Based) by YongWei.

API Source : https://api.data.gov.sg/v1/transport/traffic-images
             Camera 2701: Woodlands Causeway (Towards Johor)
             Camera 2702: Woodlands Checkpoint

Requirements:
    pip install tensorflow requests Pillow numpy matplotlib scikit-learn

Author: YongWei (Extended ML Pipeline)
"""

import os
import sys
import json
import time
import logging
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, GaussianNoise
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# --- Paths (match your extractor's directory structure) ---
BASE_DIR = r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow"
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")      # Labeled training images
MODEL_DIR = os.path.join(BASE_DIR, "models")                     # Saved model weights
PREDICTION_OUTPUT_DIR = os.path.join(BASE_DIR, "predictions")    # .txt prediction reports
LOG_DIR = os.path.join(BASE_DIR, "training_logs")                # TensorBoard logs

# --- API (same as extractor — free, no key required) ---
API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
TARGET_CAMERAS = {
    "2701": "Woodlands_Causeway_Towards_Johor",
    "2702": "Woodlands_Checkpoint",
}

# --- Model Hyperparameters ---
IMG_SIZE = (224, 224)           # MobileNetV2 standard input
BATCH_SIZE = 16
EPOCHS = 50                    # EarlyStopping will halt if no improvement
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT_LAYER = 100       # Unfreeze MobileNetV2 from this layer onwards
DROPOUT_RATE = 0.4

# --- Classification Labels ---
# (1) Crowd Density Level (multi-class — exactly one per image)
CROWD_LEVEL_LABELS = [
    "empty",              # No visible crowd
    "low",                # Sparse — few people/vehicles
    "moderate",           # Noticeable crowd, manageable flow
    "high",               # Dense crowd, slow movement
    "congested",          # Severe crowding, near standstill
]

# (2) Crowd Activity / Location Type (multi-class)
CROWD_ACTIVITY_LABELS = [
    "checkpoint_boarding",    # Crowd inside customs before boarding bus to JB
    "checkpoint_alighting",   # Crowd inside customs after arriving from JB
    "causeway_walking_to_jb", # Pedestrians walking on causeway towards JB
    "causeway_walking_from_jb", # Pedestrians walking on causeway from JB
]

# (3) Weather / Visibility Conditions (multi-label — multiple can be True)
WEATHER_LABELS = [
    "clear",        # Clear skies, good visibility
    "cloudy",       # Overcast but no rain
    "rainy",        # Active rain visible
    "wet_road",     # Wet surfaces post-rain
    "hazy",         # Haze / poor visibility (common in region)
    "night",        # Nighttime / low-light conditions
]

# Convenience counts
NUM_CROWD_LEVELS = len(CROWD_LEVEL_LABELS)
NUM_CROWD_ACTIVITIES = len(CROWD_ACTIVITY_LABELS)
NUM_WEATHER_CLASSES = len(WEATHER_LABELS)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def setup_all_directories():
    """Create all required directory structures."""
    dirs = [MODEL_DIR, PREDICTION_OUTPUT_DIR, LOG_DIR]

    # Training data subdirectories per camera
    for cam_id, cam_name in TARGET_CAMERAS.items():
        cam_dir = os.path.join(TRAINING_DATA_DIR, f"{cam_id}_{cam_name}")
        dirs.append(cam_dir)

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory ready: {d}")

    # Create labeling template if it doesn't exist
    template_path = os.path.join(TRAINING_DATA_DIR, "labels.json")
    if not os.path.exists(template_path):
        _create_label_template(template_path)

    return template_path


def _create_label_template(path: str):
    """
    Generate an empty labels.json template for manual annotation.

    LABELING GUIDE:
    ───────────────
    You must manually label your collected images before training.

    Expected structure of labels.json:
    {
        "images": [
            {
                "filename": "2701_Woodlands_Causeway_Towards_Johor_20260313_103000_original.png",
                "camera_id": "2701",
                "crowd_level": "moderate",            ← pick ONE from CROWD_LEVEL_LABELS
                "crowd_activity": "causeway_walking_to_jb",  ← pick ONE from CROWD_ACTIVITY_LABELS
                "weather": ["clear"],                  ← pick ONE or MORE from WEATHER_LABELS
                "notes": "optional free-text"
            },
            ...
        ]
    }
    """
    template = {
        "_labeling_guide": {
            "crowd_level_options": CROWD_LEVEL_LABELS,
            "crowd_activity_options": CROWD_ACTIVITY_LABELS,
            "weather_options (multi-select)": WEATHER_LABELS,
            "instructions": (
                "For each image collected by the extractor, add an entry below. "
                "crowd_level and crowd_activity are single-choice. "
                "weather is multi-select (list of strings). "
                "Copy images from 'originals/' or 'keras_224x224/' into 'training_data/<camera_folder>/'."
            ),
        },
        "images": [
            {
                "filename": "EXAMPLE_2701_Woodlands_Causeway_20260313_080000_original.png",
                "camera_id": "2701",
                "crowd_level": "moderate",
                "crowd_activity": "causeway_walking_to_jb",
                "weather": ["clear", "hazy"],
                "notes": "Morning rush, slight haze"
            }
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=4, ensure_ascii=False)
    logger.info(f"Label template created: {path}")
    logger.info(">>> IMPORTANT: Manually label your collected images in labels.json before training!")


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_labeled_dataset(labels_path: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]
]:
    """
    Load labeled images and encode labels for multi-output training.

    Args:
        labels_path: Path to labels.json

    Returns:
        images:           np.ndarray (N, 224, 224, 3) — preprocessed pixel values
        crowd_levels:     np.ndarray (N, NUM_CROWD_LEVELS) — one-hot
        crowd_activities: np.ndarray (N, NUM_CROWD_ACTIVITIES) — one-hot
        weather:          np.ndarray (N, NUM_WEATHER_CLASSES) — binary multi-label
        filenames:        List of source filenames
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = [e for e in data.get("images", []) if not e.get("filename", "").startswith("EXAMPLE_")]

    if len(entries) == 0:
        raise ValueError(
            "No labeled images found in labels.json!\n"
            "Please label your collected images first (see the template)."
        )

    logger.info(f"Loading {len(entries)} labeled images...")

    images = []
    y_crowd_level = []
    y_crowd_activity = []
    y_weather = []
    filenames = []

    for entry in entries:
        fname = entry["filename"]
        cam_id = entry.get("camera_id", "")
        cam_name = TARGET_CAMERAS.get(cam_id, f"Camera_{cam_id}")
        cam_folder = f"{cam_id}_{cam_name}"

        # Search for image file
        img_path = None
        for search_dir in [
            os.path.join(TRAINING_DATA_DIR, cam_folder),
            os.path.join(TRAINING_DATA_DIR),
            os.path.join(BASE_DIR, "originals", cam_folder),
            os.path.join(BASE_DIR, f"keras_{IMG_SIZE[0]}x{IMG_SIZE[1]}", cam_folder),
        ]:
            candidate = os.path.join(search_dir, fname)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            logger.warning(f"  Image not found: {fname} — skipping")
            continue

        # Load and preprocess image
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)  # (224, 224, 3) as float32
            images.append(img_array)
        except Exception as e:
            logger.warning(f"  Failed to load {fname}: {e}")
            continue

        # Encode crowd level (multi-class → one-hot)
        cl = entry.get("crowd_level", "low")
        cl_idx = CROWD_LEVEL_LABELS.index(cl) if cl in CROWD_LEVEL_LABELS else 1
        y_crowd_level.append(cl_idx)

        # Encode crowd activity (multi-class → one-hot)
        ca = entry.get("crowd_activity", "checkpoint_boarding")
        ca_idx = CROWD_ACTIVITY_LABELS.index(ca) if ca in CROWD_ACTIVITY_LABELS else 0
        y_crowd_activity.append(ca_idx)

        # Encode weather (multi-label → binary vector)
        weather_tags = entry.get("weather", ["clear"])
        weather_vec = np.zeros(NUM_WEATHER_CLASSES, dtype=np.float32)
        for tag in weather_tags:
            if tag in WEATHER_LABELS:
                weather_vec[WEATHER_LABELS.index(tag)] = 1.0
        y_weather.append(weather_vec)

        filenames.append(fname)

    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32)
    X = preprocess_input(X)  # Scale to [-1, 1] for MobileNetV2

    y_cl = to_categorical(np.array(y_crowd_level), num_classes=NUM_CROWD_LEVELS)
    y_ca = to_categorical(np.array(y_crowd_activity), num_classes=NUM_CROWD_ACTIVITIES)
    y_w = np.array(y_weather, dtype=np.float32)

    logger.info(f"Dataset loaded: {X.shape[0]} images, shape={X.shape[1:]}")
    logger.info(f"  Crowd levels distribution:    {dict(zip(CROWD_LEVEL_LABELS, np.sum(y_cl, axis=0).astype(int)))}")
    logger.info(f"  Crowd activity distribution:  {dict(zip(CROWD_ACTIVITY_LABELS, np.sum(y_ca, axis=0).astype(int)))}")
    logger.info(f"  Weather tag counts:           {dict(zip(WEATHER_LABELS, np.sum(y_w, axis=0).astype(int)))}")

    return X, y_cl, y_ca, y_w, filenames


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_augmentation_generator() -> ImageDataGenerator:
    """
    Build an augmentation pipeline optimized for traffic camera images.

    Key augmentations:
    - Brightness shifts (simulate time-of-day / weather changes)
    - Horizontal flip (cameras sometimes mirror)
    - Slight rotation/shift (camera vibration)
    - Channel shift (simulate haze, rain tint)

    NOTE: preprocess_input is applied BEFORE augmentation in our pipeline,
    so pixel values are already in [-1, 1]. Augmentations here are additive.
    """
    return ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,
        horizontal_flip=True,
        zoom_range=0.05,
        fill_mode="nearest",
    )


# ============================================================
# MODEL ARCHITECTURE — Multi-Output MobileNetV2
# ============================================================

def build_multi_output_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Build a multi-output classification model using MobileNetV2 transfer learning.

    Architecture:
    ┌─────────────────────────────────────┐
    │  Input (224 × 224 × 3)             │
    │  ↓                                  │
    │  MobileNetV2 (ImageNet pretrained)  │  ← Frozen initially
    │  ↓                                  │
    │  GlobalAveragePooling2D             │
    │  ↓                                  │
    │  Dense(512) + BN + Dropout          │  ← Shared feature layer
    │  ↓                 ↓          ↓     │
    │  ┌─────────┐ ┌──────────┐ ┌──────┐ │
    │  │Crowd    │ │Crowd     │ │Weathr│ │
    │  │Level    │ │Activity  │ │Cond. │ │
    │  │softmax  │ │softmax   │ │sigmoi│ │
    │  │(5)      │ │(4)       │ │d (6) │ │
    │  └─────────┘ └──────────┘ └──────┘ │
    └─────────────────────────────────────┘

    Returns:
        Compiled Keras Model with 3 output heads
    """
    # ── Base Model ──
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # Freeze base model for initial training (transfer learning phase 1)
    base_model.trainable = False

    # ── Input ──
    inputs = Input(shape=input_shape, name="image_input")

    # ── Feature Extraction ──
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)

    # ── Shared Dense Layers ──
    x = Dense(512, activation="relu", name="shared_dense_1")(x)
    x = BatchNormalization(name="shared_bn_1")(x)
    x = Dropout(dropout_rate, name="shared_dropout_1")(x)

    x = Dense(256, activation="relu", name="shared_dense_2")(x)
    x = BatchNormalization(name="shared_bn_2")(x)
    x = Dropout(dropout_rate * 0.75, name="shared_dropout_2")(x)

    # ── Output Head 1: Crowd Density Level (multi-class) ──
    crowd_level_branch = Dense(128, activation="relu", name="cl_dense")(x)
    crowd_level_branch = Dropout(0.3, name="cl_dropout")(crowd_level_branch)
    crowd_level_output = Dense(
        NUM_CROWD_LEVELS, activation="softmax", name="crowd_level"
    )(crowd_level_branch)

    # ── Output Head 2: Crowd Activity Type (multi-class) ──
    crowd_activity_branch = Dense(128, activation="relu", name="ca_dense")(x)
    crowd_activity_branch = Dropout(0.3, name="ca_dropout")(crowd_activity_branch)
    crowd_activity_output = Dense(
        NUM_CROWD_ACTIVITIES, activation="softmax", name="crowd_activity"
    )(crowd_activity_branch)

    # ── Output Head 3: Weather Conditions (multi-label) ──
    weather_branch = Dense(128, activation="relu", name="wx_dense")(x)
    weather_branch = Dropout(0.3, name="wx_dropout")(weather_branch)
    weather_output = Dense(
        NUM_WEATHER_CLASSES, activation="sigmoid", name="weather"
    )(weather_branch)

    # ── Assemble Model ──
    model = Model(
        inputs=inputs,
        outputs=[crowd_level_output, crowd_activity_output, weather_output],
        name="JalanNow_TrafficClassifier"
    )

    # ── Compile ──
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={
            "crowd_level": "categorical_crossentropy",
            "crowd_activity": "categorical_crossentropy",
            "weather": "binary_crossentropy",
        },
        loss_weights={
            "crowd_level": 1.0,
            "crowd_activity": 1.0,
            "weather": 0.5,  # Lower weight — auxiliary task
        },
        metrics={
            "crowd_level": ["accuracy"],
            "crowd_activity": ["accuracy"],
            "weather": [
                tf.keras.metrics.BinaryAccuracy(name="binary_acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        },
    )

    model.summary(print_fn=logger.info)
    return model


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_model(labels_path: str) -> Tuple[Model, dict]:
    """
    Full training pipeline: load data → train → fine-tune → save.

    Phase 1: Train only the new classification heads (base frozen)
    Phase 2: Fine-tune top layers of MobileNetV2 with low learning rate

    Args:
        labels_path: Path to labels.json with annotated images

    Returns:
        (trained_model, training_history_dict)
    """
    # ── Load Data ──
    X, y_cl, y_ca, y_w, filenames = load_labeled_dataset(labels_path)

    # ── Train/Validation Split (stratified on crowd_level) ──
    cl_indices = np.argmax(y_cl, axis=1)
    X_train, X_val, ycl_train, ycl_val, yca_train, yca_val, yw_train, yw_val = \
        train_test_split(
            X, y_cl, y_ca, y_w,
            test_size=0.2,
            random_state=42,
            stratify=cl_indices,
        )

    logger.info(f"Training set:   {X_train.shape[0]} images")
    logger.info(f"Validation set: {X_val.shape[0]} images")

    # ── Compute Class Weights (handle imbalanced crowd levels) ──
    cl_train_indices = np.argmax(ycl_train, axis=1)
    class_weights_cl = compute_class_weight(
        "balanced", classes=np.arange(NUM_CROWD_LEVELS), y=cl_train_indices
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_cl)}
    logger.info(f"Crowd level class weights: {class_weight_dict}")

    # ── Build Model ──
    model = build_multi_output_model()

    # ── Callbacks ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"best_model_{timestamp}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(LOG_DIR, f"run_{timestamp}"),
            histogram_freq=1
        ),
    ]

    # ══════════════════════════════════════════════
    # PHASE 1: Transfer Learning (base model frozen)
    # ══════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Transfer Learning (MobileNetV2 frozen)")
    logger.info("=" * 60)

    history_phase1 = model.fit(
        X_train,
        {
            "crowd_level": ycl_train,
            "crowd_activity": yca_train,
            "weather": yw_train,
        },
        validation_data=(
            X_val,
            {
                "crowd_level": ycl_val,
                "crowd_activity": yca_val,
                "weather": yw_val,
            },
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ══════════════════════════════════════════════
    # PHASE 2: Fine-Tuning (unfreeze top MobileNetV2 layers)
    # ══════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Fine-Tuning (unfreezing from layer {FINE_TUNE_AT_LAYER})")
    logger.info("=" * 60)

    # Unfreeze base model from specified layer
    base_model = model.layers[1]  # MobileNetV2 layer
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss={
            "crowd_level": "categorical_crossentropy",
            "crowd_activity": "categorical_crossentropy",
            "weather": "binary_crossentropy",
        },
        loss_weights={
            "crowd_level": 1.0,
            "crowd_activity": 1.0,
            "weather": 0.5,
        },
        metrics={
            "crowd_level": ["accuracy"],
            "crowd_activity": ["accuracy"],
            "weather": [
                tf.keras.metrics.BinaryAccuracy(name="binary_acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        },
    )

    # Update callbacks for phase 2
    callbacks_ft = [
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-8, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"best_model_finetuned_{timestamp}.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    history_phase2 = model.fit(
        X_train,
        {
            "crowd_level": ycl_train,
            "crowd_activity": yca_train,
            "weather": yw_train,
        },
        validation_data=(
            X_val,
            {
                "crowd_level": ycl_val,
                "crowd_activity": yca_val,
                "weather": yw_val,
            },
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_ft,
        verbose=1,
    )

    # ── Save Final Model ──
    final_model_path = os.path.join(MODEL_DIR, f"jalannow_classifier_{timestamp}.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    # ── Save Training History ──
    combined_history = {}
    for key in history_phase1.history:
        combined_history[key] = history_phase1.history[key] + history_phase2.history.get(key, [])

    history_path = os.path.join(MODEL_DIR, f"training_history_{timestamp}.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in combined_history.items()}, f, indent=2)
    logger.info(f"Training history saved: {history_path}")

    # ── Plot Training Curves ──
    _plot_training_history(combined_history, timestamp)

    return model, combined_history


def _plot_training_history(history: dict, timestamp: str):
    """Generate and save training metric plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("JalanNow Traffic Classifier — Training History", fontsize=14)

    # Plot 1: Overall Loss
    axes[0, 0].plot(history.get("loss", []), label="Train Loss")
    axes[0, 0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Crowd Level Accuracy
    axes[0, 1].plot(history.get("crowd_level_accuracy", []), label="Train")
    axes[0, 1].plot(history.get("val_crowd_level_accuracy", []), label="Val")
    axes[0, 1].set_title("Crowd Level Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Crowd Activity Accuracy
    axes[1, 0].plot(history.get("crowd_activity_accuracy", []), label="Train")
    axes[1, 0].plot(history.get("val_crowd_activity_accuracy", []), label="Val")
    axes[1, 0].set_title("Crowd Activity Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Weather Precision & Recall
    axes[1, 1].plot(history.get("weather_precision", []), label="Train Precision")
    axes[1, 1].plot(history.get("val_weather_precision", []), label="Val Precision")
    axes[1, 1].plot(history.get("weather_recall", []), label="Train Recall")
    axes[1, 1].plot(history.get("val_weather_recall", []), label="Val Recall")
    axes[1, 1].set_title("Weather Detection Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, f"training_curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Training plots saved: {plot_path}")


# ============================================================
# LIVE IMAGE FETCHING (from data.gov.sg API)
# ============================================================

def fetch_latest_camera_image(camera_id: str = "2701") -> Tuple[Optional[np.ndarray], dict]:
    """
    Fetch the latest live traffic camera image from data.gov.sg API.

    The API is free, requires no authentication, and updates every ~20 seconds.
    Image URLs are temporary (expire in ~5 minutes), so we download immediately.

    Args:
        camera_id: Camera ID ("2701" for Causeway, "2702" for Checkpoint)

    Returns:
        (preprocessed_image_array, metadata_dict) or (None, {}) on failure
    """
    logger.info(f"Fetching latest image for Camera {camera_id} from data.gov.sg API...")

    try:
        response = requests.get(
            API_URL,
            headers={"Accept": "application/json", "User-Agent": "JalanNow-ML/2.0"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Find our target camera
        cameras = data.get("items", [{}])[0].get("cameras", [])
        target = None
        for cam in cameras:
            if cam.get("camera_id") == camera_id:
                target = cam
                break

        if target is None:
            logger.error(f"Camera {camera_id} not found in API response!")
            return None, {}

        image_url = target.get("image", "")
        timestamp = target.get("timestamp", "")
        location = target.get("location", {})
        metadata_info = target.get("image_metadata", {})

        metadata = {
            "camera_id": camera_id,
            "camera_name": TARGET_CAMERAS.get(camera_id, f"Camera_{camera_id}"),
            "timestamp": timestamp,
            "latitude": location.get("latitude", 0),
            "longitude": location.get("longitude", 0),
            "original_width": metadata_info.get("width", 0),
            "original_height": metadata_info.get("height", 0),
            "api_md5": metadata_info.get("md5", ""),
            "image_url": image_url,
        }

        logger.info(f"  Camera: {metadata['camera_name']}")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Resolution: {metadata['original_width']}x{metadata['original_height']}")

        # Download image (URLs expire in ~5 min)
        img_response = requests.get(
            image_url,
            headers={"User-Agent": "JalanNow-ML/2.0"},
            timeout=30,
        )
        img_response.raise_for_status()

        # Process for model input
        img = Image.open(BytesIO(img_response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save raw image for the prediction report
        raw_save_dir = os.path.join(PREDICTION_OUTPUT_DIR, "raw_images")
        os.makedirs(raw_save_dir, exist_ok=True)
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            dt = datetime.now()
        raw_filename = f"{camera_id}_{dt.strftime('%Y%m%d_%H%M%S')}_raw.png"
        raw_path = os.path.join(raw_save_dir, raw_filename)
        img.save(raw_path, format="PNG")
        metadata["raw_image_path"] = raw_path

        # Resize for model
        img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array.copy())

        logger.info(f"  Image preprocessed: shape={img_preprocessed.shape}")
        return img_preprocessed, metadata

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None, {}
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return None, {}


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_single_image(
    model: Model,
    image: np.ndarray,
    metadata: dict,
    weather_threshold: float = 0.4,
) -> dict:
    """
    Run prediction on a single preprocessed image.

    Args:
        model: Trained Keras model with 3 output heads
        image: Preprocessed image array (1, 224, 224, 3)
        metadata: Camera/image metadata dict
        weather_threshold: Probability threshold for weather multi-label

    Returns:
        Comprehensive prediction results dict
    """
    logger.info("Running model prediction...")
    start_time = time.time()

    # ── Predict ──
    predictions = model.predict(image, verbose=0)
    inference_time = time.time() - start_time

    crowd_level_probs = predictions[0][0]
    crowd_activity_probs = predictions[1][0]
    weather_probs = predictions[2][0]

    # ── Decode Crowd Level ──
    cl_idx = np.argmax(crowd_level_probs)
    cl_label = CROWD_LEVEL_LABELS[cl_idx]
    cl_confidence = float(crowd_level_probs[cl_idx])

    # ── Decode Crowd Activity ──
    ca_idx = np.argmax(crowd_activity_probs)
    ca_label = CROWD_ACTIVITY_LABELS[ca_idx]
    ca_confidence = float(crowd_activity_probs[ca_idx])

    # ── Decode Weather (multi-label with threshold) ──
    wx_detected = []
    for i, prob in enumerate(weather_probs):
        if prob >= weather_threshold:
            wx_detected.append({
                "condition": WEATHER_LABELS[i],
                "confidence": float(prob),
            })

    if not wx_detected:
        # Fallback: take the single highest weather prediction
        wx_top_idx = np.argmax(weather_probs)
        wx_detected.append({
            "condition": WEATHER_LABELS[wx_top_idx],
            "confidence": float(weather_probs[wx_top_idx]),
        })

    # ── Build Travel Advisory ──
    advisory = _generate_travel_advisory(cl_label, ca_label, wx_detected, metadata)

    results = {
        "prediction_timestamp": datetime.now().isoformat(),
        "inference_time_ms": round(inference_time * 1000, 1),
        "camera_metadata": metadata,
        "crowd_level": {
            "predicted_class": cl_label,
            "confidence": round(cl_confidence, 4),
            "all_probabilities": {
                CROWD_LEVEL_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(crowd_level_probs)
            },
        },
        "crowd_activity": {
            "predicted_class": ca_label,
            "confidence": round(ca_confidence, 4),
            "all_probabilities": {
                CROWD_ACTIVITY_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(crowd_activity_probs)
            },
        },
        "weather_conditions": {
            "detected": wx_detected,
            "all_probabilities": {
                WEATHER_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(weather_probs)
            },
        },
        "travel_advisory": advisory,
    }

    return results


def _generate_travel_advisory(
    crowd_level: str,
    crowd_activity: str,
    weather: list,
    metadata: dict,
) -> dict:
    """
    Generate human-readable travel advisory based on predictions.
    This provides actionable insights for travelers using the Woodlands Causeway.
    """
    advisory_lines = []
    risk_score = 0  # 0-100 scale

    camera_name = metadata.get("camera_name", "Unknown")

    # ── Crowd Level Assessment ──
    crowd_descriptions = {
        "empty": ("✅ Clear passage — minimal to no crowd detected.", 0),
        "low": ("✅ Light traffic — smooth flow expected.", 10),
        "moderate": ("⚠️ Moderate crowd — expect some waiting time at customs.", 35),
        "high": ("🔴 Heavy crowd — significant delays likely. Consider alternative timing.", 65),
        "congested": ("🔴🔴 SEVERE CONGESTION — expect 1-2+ hour delays. Strongly consider postponing.", 90),
    }
    desc, score = crowd_descriptions.get(crowd_level, ("Unknown crowd level.", 30))
    advisory_lines.append(f"CROWD STATUS: {desc}")
    risk_score += score * 0.4

    # ── Crowd Activity Assessment ──
    activity_descriptions = {
        "checkpoint_boarding": (
            "📍 Crowd concentrated INSIDE Woodlands Checkpoint — "
            "travelers appear to be queuing to board buses/transport towards JB.",
            10
        ),
        "checkpoint_alighting": (
            "📍 Crowd inside Woodlands Checkpoint — "
            "travelers arriving FROM JB and proceeding through Singapore customs.",
            10
        ),
        "causeway_walking_to_jb": (
            "🚶 Pedestrians detected WALKING ON THE CAUSEWAY towards JB — "
            "this typically indicates heavy congestion forcing travelers to walk.",
            25
        ),
        "causeway_walking_from_jb": (
            "🚶 Pedestrians detected WALKING ON THE CAUSEWAY from JB — "
            "return traffic is heavy; expect delays at SG customs.",
            25
        ),
    }
    act_desc, act_score = activity_descriptions.get(
        crowd_activity, ("Activity unclear.", 15)
    )
    advisory_lines.append(f"ACTIVITY: {act_desc}")
    risk_score += act_score * 0.3

    # ── Weather Impact Assessment ──
    weather_names = [w["condition"] for w in weather]
    weather_impact_lines = []

    if "rainy" in weather_names:
        weather_impact_lines.append(
            "🌧️ RAIN DETECTED — Causeway pedestrians will be affected. "
            "Walking travelers should bring umbrellas. "
            "Slippery conditions may slow traffic further."
        )
        risk_score += 20 * 0.3

    if "hazy" in weather_names:
        weather_impact_lines.append(
            "🌫️ HAZE/LOW VISIBILITY — Reduced visibility may cause "
            "cautious driving and slower checkpoint processing."
        )
        risk_score += 15 * 0.3

    if "wet_road" in weather_names:
        weather_impact_lines.append(
            "💧 WET ROAD conditions — Recent rain. Roads and causeway may be slippery."
        )
        risk_score += 10 * 0.3

    if "night" in weather_names:
        weather_impact_lines.append(
            "🌙 NIGHTTIME — Low-light conditions. Pedestrians on causeway should "
            "exercise extra caution."
        )
        risk_score += 5 * 0.3

    if "cloudy" in weather_names and "rainy" not in weather_names:
        weather_impact_lines.append(
            "☁️ Overcast skies — No immediate rain impact but may develop."
        )

    if "clear" in weather_names:
        weather_impact_lines.append(
            "☀️ Clear weather — Good conditions for travel."
        )

    if not weather_impact_lines:
        weather_impact_lines.append("Weather conditions appear normal.")

    advisory_lines.append("WEATHER IMPACT: " + " | ".join(weather_impact_lines))

    # ── Overall Risk Score ──
    risk_score = min(100, max(0, int(risk_score)))
    if risk_score <= 20:
        risk_level = "LOW"
        risk_emoji = "🟢"
    elif risk_score <= 45:
        risk_level = "MODERATE"
        risk_emoji = "🟡"
    elif risk_score <= 70:
        risk_level = "HIGH"
        risk_emoji = "🟠"
    else:
        risk_level = "CRITICAL"
        risk_emoji = "🔴"

    advisory_lines.insert(0, f"{risk_emoji} OVERALL TRAVEL RISK: {risk_level} ({risk_score}/100)")

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "advisory_text": advisory_lines,
    }


# ============================================================
# TXT REPORT GENERATION
# ============================================================

def generate_prediction_report(results: dict, output_dir: str = PREDICTION_OUTPUT_DIR) -> str:
    """
    Generate a comprehensive prediction report as a .txt file.

    The report includes:
    - Camera metadata and timestamp
    - Crowd density level with confidence scores
    - Crowd activity classification
    - Weather condition analysis
    - Travel advisory with risk score
    - Raw probability distributions for all classes

    Args:
        results: Prediction results dict from predict_single_image()
        output_dir: Directory to save the report

    Returns:
        Path to the generated .txt report file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    cam_id = results["camera_metadata"].get("camera_id", "unknown")
    cam_name = results["camera_metadata"].get("camera_name", "unknown")
    pred_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{cam_id}_{cam_name}_{pred_time}.txt"
    filepath = os.path.join(output_dir, filename)

    # Build report content
    meta = results["camera_metadata"]
    cl = results["crowd_level"]
    ca = results["crowd_activity"]
    wx = results["weather_conditions"]
    adv = results["travel_advisory"]

    lines = []
    w = 80  # line width

    lines.append("=" * w)
    lines.append("  JALANNOW TRAFFIC CAMERA — AI PREDICTION REPORT")
    lines.append("=" * w)
    lines.append("")

    # ── Section 1: Metadata ──
    lines.append("─" * w)
    lines.append("  CAMERA & IMAGE INFORMATION")
    lines.append("─" * w)
    lines.append(f"  Prediction Time    : {results.get('prediction_timestamp', 'N/A')}")
    lines.append(f"  Inference Time     : {results.get('inference_time_ms', 'N/A')} ms")
    lines.append(f"  Camera ID          : {meta.get('camera_id', 'N/A')}")
    lines.append(f"  Camera Name        : {meta.get('camera_name', 'N/A')}")
    lines.append(f"  Image Timestamp    : {meta.get('timestamp', 'N/A')}")
    lines.append(f"  Location (lat/lon) : {meta.get('latitude', 'N/A')}, {meta.get('longitude', 'N/A')}")
    lines.append(f"  Original Resolution: {meta.get('original_width', '?')}x{meta.get('original_height', '?')}")
    lines.append(f"  Raw Image Saved    : {meta.get('raw_image_path', 'N/A')}")
    lines.append(f"  API Source         : {API_URL}")
    lines.append("")

    # ── Section 2: Travel Advisory (most important — first) ──
    lines.append("─" * w)
    lines.append("  🚀 TRAVEL ADVISORY")
    lines.append("─" * w)
    for line in adv.get("advisory_text", []):
        lines.append(f"  {line}")
    lines.append("")

    # ── Section 3: Crowd Density Level ──
    lines.append("─" * w)
    lines.append("  📊 CROWD DENSITY LEVEL (Prediction 1 of 3)")
    lines.append("─" * w)
    lines.append(f"  Predicted Class : {cl['predicted_class'].upper()}")
    lines.append(f"  Confidence      : {cl['confidence'] * 100:.1f}%")
    lines.append("")
    lines.append("  Probability Distribution:")
    for label, prob in cl["all_probabilities"].items():
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ◄── PREDICTED" if label == cl["predicted_class"] else ""
        lines.append(f"    {label:<12} [{bar}] {prob * 100:5.1f}%{marker}")
    lines.append("")
    lines.append("  Interpretation:")
    crowd_interp = {
        "empty": "No visible pedestrians or vehicles. Checkpoint appears vacant.",
        "low": "Few travelers visible. Minimal queue expected at customs.",
        "moderate": "Noticeable number of travelers. Some queuing at customs counters.",
        "high": "Dense crowd visible. Long queues and slow processing expected.",
        "congested": "Severe overcrowding. Possible spillover onto causeway. Major delays.",
    }
    lines.append(f"    → {crowd_interp.get(cl['predicted_class'], 'N/A')}")
    lines.append("")

    # ── Section 4: Crowd Activity ──
    lines.append("─" * w)
    lines.append("  🚶 CROWD ACTIVITY TYPE (Prediction 2 of 3)")
    lines.append("─" * w)
    lines.append(f"  Predicted Class : {ca['predicted_class']}")
    lines.append(f"  Confidence      : {ca['confidence'] * 100:.1f}%")
    lines.append("")
    lines.append("  Probability Distribution:")
    for label, prob in ca["all_probabilities"].items():
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ◄── PREDICTED" if label == ca["predicted_class"] else ""
        lines.append(f"    {label:<28} [{bar}] {prob * 100:5.1f}%{marker}")
    lines.append("")
    lines.append("  Interpretation:")
    activity_interp = {
        "checkpoint_boarding": (
            "Crowd is concentrated INSIDE Woodlands Checkpoint area.\n"
            "    Travelers appear to be in queue before boarding public transport (bus)\n"
            "    heading towards Johor Bahru, Malaysia."
        ),
        "checkpoint_alighting": (
            "Crowd is inside the Woodlands Checkpoint arrival zone.\n"
            "    Travelers are arriving FROM Johor Bahru and proceeding through\n"
            "    Singapore immigration/customs clearance."
        ),
        "causeway_walking_to_jb": (
            "Pedestrians are WALKING on the Woodlands Causeway TOWARDS Johor Bahru.\n"
            "    This typically occurs during peak periods when vehicle traffic is\n"
            "    severely congested, and travelers choose to cross on foot (~1.1 km).\n"
            "    Weather conditions significantly affect these pedestrians."
        ),
        "causeway_walking_from_jb": (
            "Pedestrians are WALKING on the Woodlands Causeway FROM Johor Bahru\n"
            "    back towards Singapore. Return traffic is heavy.\n"
            "    Expect additional delays at Singapore customs upon arrival."
        ),
    }
    interp_text = activity_interp.get(ca["predicted_class"], "Activity type unclear.")
    for interp_line in interp_text.split("\n"):
        lines.append(f"    → {interp_line.strip()}")
    lines.append("")

    # ── Section 5: Weather Conditions ──
    lines.append("─" * w)
    lines.append("  🌤️ WEATHER & VISIBILITY CONDITIONS (Prediction 3 of 3)")
    lines.append("─" * w)
    lines.append(f"  Detected Conditions:")
    for wx_item in wx["detected"]:
        lines.append(f"    • {wx_item['condition'].upper()} (confidence: {wx_item['confidence'] * 100:.1f}%)")
    lines.append("")
    lines.append("  Full Probability Distribution:")
    for label, prob in wx["all_probabilities"].items():
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        detected_marker = " ◄── DETECTED" if any(
            d["condition"] == label for d in wx["detected"]
        ) else ""
        lines.append(f"    {label:<10} [{bar}] {prob * 100:5.1f}%{detected_marker}")
    lines.append("")
    lines.append("  Weather Impact on Travel:")
    if any(d["condition"] == "rainy" for d in wx["detected"]):
        lines.append("    ⚠️ RAIN: Causeway pedestrians will get wet. Bring umbrella/raincoat.")
        lines.append("       Slippery road conditions. Vehicle traffic may be slower.")
        lines.append("       Consider postponing walking across causeway.")
    if any(d["condition"] == "hazy" for d in wx["detected"]):
        lines.append("    ⚠️ HAZE: Reduced visibility affects driving safety.")
        lines.append("       Checkpoint operations may slow down.")
    if any(d["condition"] == "night" for d in wx["detected"]):
        lines.append("    ℹ️ NIGHT: Low-light conditions. Exercise caution if walking.")
    if any(d["condition"] == "clear" for d in wx["detected"]):
        lines.append("    ✅ CLEAR: Good weather for travel. No weather-related delays expected.")
    lines.append("")

    # ── Section 6: Summary ──
    lines.append("─" * w)
    lines.append("  📋 PREDICTION SUMMARY")
    lines.append("─" * w)
    lines.append(f"  ┌{'─' * (w - 4)}┐")
    lines.append(f"  │ {'Risk Level:':<20} {adv['risk_level']:<15} (Score: {adv['risk_score']}/100){' ' * (w - 59)}│")
    lines.append(f"  │ {'Crowd Density:':<20} {cl['predicted_class'].upper():<15} ({cl['confidence'] * 100:.0f}% conf.){' ' * (w - 60)}│")
    lines.append(f"  │ {'Activity:':<20} {ca['predicted_class']:<35}{' ' * (w - 59)}│")
    wx_str = ", ".join(d["condition"] for d in wx["detected"])
    lines.append(f"  │ {'Weather:':<20} {wx_str:<35}{' ' * max(0, w - 59)}│")
    lines.append(f"  └{'─' * (w - 4)}┘")
    lines.append("")

    # ── Footer ──
    lines.append("─" * w)
    lines.append("  NOTES")
    lines.append("─" * w)
    lines.append("  • Predictions are AI-generated estimates based on traffic camera imagery.")
    lines.append("  • Confidence scores indicate model certainty (higher = more confident).")
    lines.append("  • Weather detection is multi-label (multiple conditions can coexist).")
    lines.append("  • For most accurate results, ensure the model is trained with ≥500 labeled images.")
    lines.append(f"  • Model architecture: MobileNetV2 (transfer learning) with multi-output heads.")
    lines.append(f"  • Image source: data.gov.sg Traffic Images API (Camera {cam_id})")
    lines.append("")
    lines.append("=" * w)
    lines.append(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  JalanNow AI Traffic Classifier v2.0 — by YongWei")
    lines.append("=" * w)

    # Write to file
    report_content = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Prediction report saved: {filepath}")

    # Also print to console
    print("\n" + report_content)

    return filepath


# ============================================================
# BATCH PREDICTION (all cameras)
# ============================================================

def predict_all_cameras(model: Model) -> List[str]:
    """
    Fetch the latest image from ALL target cameras and run predictions.

    Returns:
        List of generated report file paths
    """
    report_paths = []

    for cam_id, cam_name in TARGET_CAMERAS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing Camera {cam_id}: {cam_name}")
        logger.info(f"{'=' * 60}")

        image, metadata = fetch_latest_camera_image(camera_id=cam_id)

        if image is None:
            logger.error(f"Failed to fetch image for Camera {cam_id}. Skipping.")
            continue

        results = predict_single_image(model, image, metadata)
        report_path = generate_prediction_report(results)
        report_paths.append(report_path)

        # Save results as JSON too (for programmatic access)
        json_path = report_path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON results saved: {json_path}")

    return report_paths


# ============================================================
# UTILITY: Find Latest Model
# ============================================================

def find_latest_model(model_dir: str = MODEL_DIR) -> Optional[str]:
    """Find the most recently saved model file."""
    if not os.path.exists(model_dir):
        return None

    model_files = []
    for f in os.listdir(model_dir):
        if f.endswith(".keras") or f.endswith(".h5"):
            full_path = os.path.join(model_dir, f)
            model_files.append((full_path, os.path.getmtime(full_path)))

    if not model_files:
        return None

    model_files.sort(key=lambda x: x[1], reverse=True)
    return model_files[0][0]


# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def main_train():
    """Entry point: Train the model from labeled data."""
    logger.info("=" * 60)
    logger.info("JalanNow Traffic Classifier — TRAINING MODE")
    logger.info("=" * 60)

    labels_path = setup_all_directories()
    model, history = train_model(labels_path)

    logger.info("\n✅ Training complete!")
    logger.info(f"Model saved in: {MODEL_DIR}")
    logger.info("You can now run prediction mode with: main_predict()")


def main_predict():
    """Entry point: Load trained model and predict on latest live images."""
    logger.info("=" * 60)
    logger.info("JalanNow Traffic Classifier — PREDICTION MODE")
    logger.info("=" * 60)

    setup_all_directories()

    # Find and load the latest trained model
    model_path = find_latest_model()
    if model_path is None:
        logger.error(
            "No trained model found!\n"
            "Please run training first: main_train()\n"
            f"Expected model directory: {MODEL_DIR}"
        )
        return

    logger.info(f"Loading model: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully!")

    # Run prediction on all target cameras
    report_paths = predict_all_cameras(model)

    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION SESSION COMPLETE")
    logger.info("=" * 60)
    for rp in report_paths:
        logger.info(f"  Report: {rp}")
    logger.info(f"  Output directory: {PREDICTION_OUTPUT_DIR}")


def main_predict_single(camera_id: str = "2701"):
    """Entry point: Predict on a single camera's latest image."""
    setup_all_directories()

    model_path = find_latest_model()
    if model_path is None:
        logger.error("No trained model found! Run main_train() first.")
        return

    model = load_model(model_path)
    image, metadata = fetch_latest_camera_image(camera_id=camera_id)

    if image is None:
        logger.error("Failed to fetch camera image.")
        return

    results = predict_single_image(model, image, metadata)
    report_path = generate_prediction_report(results)

    logger.info(f"\n✅ Report saved: {report_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Usage:
        python jalannow_classifier.py train      → Train model from labeled data
        python jalannow_classifier.py predict     → Predict all cameras (latest images)
        python jalannow_classifier.py predict2701 → Predict Camera 2701 only
        python jalannow_classifier.py predict2702 → Predict Camera 2702 only
    """
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python jalannow_classifier.py train        Train model")
        print("  python jalannow_classifier.py predict      Predict all cameras")
        print("  python jalannow_classifier.py predict2701  Predict Camera 2701 only")
        print("  python jalannow_classifier.py predict2702  Predict Camera 2702 only")
        sys.exit(0)

    command = sys.argv[1].lower().strip()

    if command == "train":
        main_train()
    elif command == "predict":
        main_predict()
    elif command == "predict2701":
        main_predict_single("2701")
    elif command == "predict2702":
        main_predict_single("2702")
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: train, predict, predict2701, predict2702")