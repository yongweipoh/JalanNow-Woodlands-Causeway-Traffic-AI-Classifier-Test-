# JalanNow-Woodlands-Causeway-Traffic-AI-Classifier-Test-

[ReadMe.txt](https://github.com/user-attachments/files/25971123/ReadMe.txt)
 **README.md** 

```markdown
# 🚦 JalanNow — Woodlands Causeway Traffic AI Classifier

> AI-powered traffic analysis for the Singapore–Johor Bahru Woodlands Causeway using
> live camera feeds from Singapore's [data.gov.sg](https://data.gov.sg) API and
> TensorFlow Keras deep learning.

Predicts **crowd density**, **crowd activity** (walking on causeway / inside checkpoint),
and **weather impact** on travelers — all from a single traffic camera image.


**Camera Sources** ([LTA DataMall / data.gov.sg](https://lta.gov.sg)):
| Camera ID | Location | View |
|-----------|----------|------|
| **2701** | Woodlands Causeway | Towards Johor Bahru |
| **2702** | Woodlands Checkpoint | Customs / Immigration |

**Author:** YongWei


## 📑 Table of Contents

1. [Prerequisites](#1--prerequisites)
2. [Installation](#2--installation)
3. [Project File Structure](#3--project-file-structure)
4. [Step 1 — Extract Images from data.gov.sg](#4--step-1--extract-images-from-datagovsg)
5. [Step 2 — Label Images](#5--step-2--label-images)
6. [Step 3 — Validate Labels](#6--step-3--validate-labels)
7. [Step 4 — Train the Model](#7--step-4--train-the-model)
8. [Step 5 — Predict on Live Images](#8--step-5--predict-on-live-images)
9. [Understanding the Output Report](#9--understanding-the-output-report)
10. [Troubleshooting](#10--troubleshooting)
11. [Classification Labels Reference](#11--classification-labels-reference)
12. [Technical Architecture](#12--technical-architecture)
13. [License](#13--license)

## 1. 📋 Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any modern x64 processor | Intel i5 / AMD Ryzen 5 or better |
| RAM | 8 GB | 16 GB |
| GPU | Not required (CPU works) | NVIDIA GPU with CUDA support |
| Storage | 2 GB free | 10 GB free (for 1000+ images) |
| Internet | Required for image extraction and prediction | Stable connection |

### Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10 or 3.11 | Runtime ([python.org](https://python.org)) |
| **pip** | Latest | Package manager (comes with Python) |
| **Git** | Any | Version control (optional) |
| **VS Code** or **Notepad++** | Any | Editing labels.json (optional) |

> ⚠️ **Windows GPU Note**: TensorFlow 2.11+ does not support native Windows GPU.
> Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) for GPU acceleration,
> or train on CPU (slower but works fine). See the
> [TensorFlow GPU Guide](https://www.tensorflow.org/install/pip) for details.

---

## 2. 🛠️ Installation

### 2.1 — Clone or Download This Project

```bash
# Option A: Git clone
git clone https://github.com/YongWei/jalannow-traffic-classifier.git
cd jalannow-traffic-classifier

# Option B: Or just create the project folder manually
mkdir JalanNow
cd JalanNow
```

### 2.2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 2.3 — Install Dependencies

```bash
pip install --upgrade pip
pip install tensorflow requests Pillow numpy matplotlib scikit-learn schedule
```

**Verify TensorFlow installed correctly:**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

Expected output:
```
TensorFlow 2.17.0
```

**(Optional) Verify GPU is detected:**

```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### 2.4 — Project Scripts

Make sure you have these **4 Python files** in your project folder:

| File | Purpose | When to Run |
|------|---------|-------------|
| `jalannow_extractor.py` | Extracts traffic camera images from API | Step 1 |
| `annotation_tool.py` | GUI tool for labeling images | Step 2 |
| `validate_labels.py` | Checks labels.json for errors | Step 3 |
| `jalannow_classifier.py` | Trains model & runs predictions | Steps 4 & 5 |

---

## 3. 📁 Project File Structure

After running all steps, your folder will look like this:

```
JalanNow/
│
├── jalannow_extractor.py          ← Image extraction script
├── annotation_tool.py             ← GUI labeling tool
├── validate_labels.py             ← Label validation script
├── jalannow_classifier.py         ← Training & prediction script
├── README.md                      ← This file
│
├── originals/                     ← Raw images from API (auto-created)
│   ├── 2701_Woodlands_Causeway_Towards_Johor/
│   │   ├── 2701_..._20260313_080000_original.png
│   │   ├── 2701_..._20260313_083000_original.png
│   │   └── ...
│   └── 2702_Woodlands_Checkpoint/
│       └── ...
│
├── keras_224x224/                  ← Resized images for model (auto-created)
│   ├── 2701_Woodlands_Causeway_Towards_Johor/
│   └── 2702_Woodlands_Checkpoint/
│
├── training_data/                 ← Labeled images + labels.json
│   ├── 2701_Woodlands_Causeway_Towards_Johor/
│   ├── 2702_Woodlands_Checkpoint/
│   └── labels.json                ← YOUR ANNOTATIONS GO HERE
│
├── models/                        ← Saved trained models (auto-created)
│   ├── jalannow_classifier_20260315_143022.keras
│   ├── training_history_20260315_143022.json
│   └── training_curves_20260315_143022.png
│
├── predictions/                   ← Prediction output reports (auto-created)
│   ├── raw_images/
│   ├── prediction_2701_..._20260316_080045.txt
│   └── prediction_2701_..._20260316_080045.json
│
├── training_logs/                 ← TensorBoard logs (auto-created)
├── timestamps.txt                 ← Extraction timestamp log
├── capture_log.txt                ← Extraction debug log
└── image_hashes.json              ← Duplicate tracking
```

---

## 4. 📷 Step 1 — Extract Images from data.gov.sg

### What This Does

Connects to Singapore's free [data.gov.sg Traffic Images API](https://data.gov.sg/datasets/d_4288e2bbb5d04f64de0c5e4bd665c19c/view) every 30 minutes
and saves images from Camera 2701 (Woodlands Causeway) and Camera 2702
(Woodlands Checkpoint). No API key is required ([lta.gov.sg](https://lta.gov.sg)).

### How to Run

```bash
python jalannow_extractor.py
```

### What You'll See

```
============================================================
JalanNow Traffic Camera Collector (API-Based)
============================================================
API Endpoint:     https://api.data.gov.sg/v1/transport/traffic-images
Target Cameras:   ['2701', '2702']
Save Directory:   C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow
Capture Interval: Every 30 minutes
Max Captures:     Unlimited
============================================================

>>> First capture starting immediately...

============================================================
CAPTURE CYCLE @ 2026-03-13 08:00:00
============================================================
Fetching traffic images from API...
  Found Camera 2701 (Woodlands_Causeway_Towards_Johor): 2006x1504 @ 2026-03-13T08:00:04+08:00
  Found Camera 2702 (Woodlands_Checkpoint): 2006x1504 @ 2026-03-13T08:00:04+08:00

Processing Camera 2701: Woodlands_Causeway_Towards_Johor
  Downloaded: 312.5 KB
  ✓ Original saved: originals/2701_.../2701_..._original.png (312.5 KB)
  ✓ Keras saved:    keras_224x224/2701_.../2701_..._keras_224x224.png (48.2 KB)

Capture cycle complete: 2/2 images saved

Scheduler running. Next capture in 30 minutes.
Press Ctrl+C to stop.
```

### How Long to Run

| Duration | Approx. Images | Quality |
|----------|---------------|---------|
| 1 day | ~96 images | Too few — just for testing |
| 3 days | ~288 images | Minimum for basic training |
| **7 days** | **~672 images** | **Recommended starting point** |
| 14 days | ~1,344 images | Better model accuracy |
| 30 days | ~2,880 images | Best accuracy & weather diversity |

> 💡 **Tips:**
> - Run across **weekdays AND weekends** — traffic patterns differ significantly.
> - Run through **different weather** — rain, haze, clear skies, night.
> - Leave running on a desktop / laptop with power connected.
> - Press `Ctrl+C` anytime to stop gracefully.

### Configuration (Optional)

Edit the top of `jalannow_extractor.py` to change settings:

```python
CAPTURE_INTERVAL_MINUTES = 30   # Change to 15 for more images, 60 for fewer
MAX_CAPTURES = 0                # Set to e.g. 100 to auto-stop after 100 captures
SAVE_DIR = r"C:\your\custom\path"  # Change save location
```

---

## 5. 🏷️ Step 2 — Label Images

### What This Does

You assign 3 labels to each collected image by looking at it and deciding:
1. **How crowded** is it? (crowd level)
2. **What are people doing?** (crowd activity)
3. **What's the weather?** (weather conditions)

### 5.1 — Copy Images to Training Folder

First, copy your collected images into the `training_data` folder:

**Windows (Command Prompt):**

```cmd
REM Create training directories
mkdir "training_data\2701_Woodlands_Causeway_Towards_Johor"
mkdir "training_data\2702_Woodlands_Checkpoint"

REM Copy images
xcopy "originals\2701_Woodlands_Causeway_Towards_Johor\*" "training_data\2701_Woodlands_Causeway_Towards_Johor\" /Y
xcopy "originals\2702_Woodlands_Checkpoint\*" "training_data\2702_Woodlands_Checkpoint\" /Y
```

**Windows (PowerShell):**

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "training_data\2701_Woodlands_Causeway_Towards_Johor"
New-Item -ItemType Directory -Force -Path "training_data\2702_Woodlands_Checkpoint"

# Copy images
Copy-Item "originals\2701_Woodlands_Causeway_Towards_Johor\*" "training_data\2701_Woodlands_Causeway_Towards_Johor\" -Force
Copy-Item "originals\2702_Woodlands_Checkpoint\*" "training_data\2702_Woodlands_Checkpoint\" -Force
```

### 5.2 — Run the GUI Annotation Tool (Recommended Method)

```bash
python annotation_tool.py
```

**The GUI window opens:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Image 1 of 672 (in current view)       Total labeled: 0 images           │
├────────────────────────────────┬────────────────────────────────────────────┤
│                                │  📊 CROWD DENSITY LEVEL (pick one)       │
│                                │  ─────────────────────────────────        │
│                                │  ○ ⬜ EMPTY                              │
│                                │  ○ 🟢 LOW                                │
│     [TRAFFIC CAMERA IMAGE]     │  ○ 🟡 MODERATE                           │
│                                │  ○ 🟠 HIGH                               │
│     (your image displays       │  ○ 🔴 CONGESTED                          │
│      here automatically)       │                                          │
│                                │  🚶 CROWD ACTIVITY (pick one)            │
│                                │  ─────────────────────────────────        │
│                                │  ○ 🚌 Checkpoint → Boarding bus TO JB    │
│                                │  ○ 🛬 Checkpoint → Arriving FROM JB     │
│                                │  ○ 🚶➡️ Walking on causeway TO JB        │
│                                │  ○ 🚶⬅️ Walking on causeway FROM JB     │
│                                │                                          │
│                                │  🌤️ WEATHER CONDITIONS (select all)      │
│                                │  ─────────────────────────────────        │
│                                │  ☐ ☀️ CLEAR                              │
│                                │  ☐ ☁️ CLOUDY                             │
│                                │  ☐ 🌧️ RAINY                             │
│                                │  ☐ 💧 WET_ROAD                          │
│                                │  ☐ 🌫️ HAZY                              │
│                                │  ☐ 🌙 NIGHT                             │
│                                │                                          │
│                                │  📝 NOTES (optional)                     │
│                                │  [________________________]              │
│                                │                                          │
│                                │  ┌──────────────────────────────┐        │
│                                │  │     ✅ SAVE & NEXT →         │        │
│                                │  └──────────────────────────────┘        │
│                                │  [◄ PREV]  [SKIP ►]  [🗑️ CLEAR]        │
├────────────────────────────────┴────────────────────────────────────────────┤
│  Shortcuts: Enter=Save&Next | ←→=Navigate | Esc=Clear | 1-5=Crowd Level  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 — Labeling Workflow (for each image)

**Repeat these 4 actions for every image:**

```
 👀 LOOK at the image on the left
       │
       ▼
 1️⃣  SELECT crowd level        → click one radio button (or press 1-5)
       │
       ▼
 2️⃣  SELECT crowd activity     → click one radio button
       │
       ▼
 3️⃣  CHECK weather boxes       → tick one or more checkboxes
       │
       ▼
 ⏎  PRESS ENTER                → saves label, loads next image
```

### 5.4 — How to Decide Each Label

#### Crowd Level — "How many people/vehicles do I see?"

```
 EMPTY          LOW            MODERATE         HIGH            CONGESTED
 ┌─────┐       ┌─────┐       ┌─────┐         ┌─────┐         ┌─────┐
 │     │       │  .  │       │ . . │         │.....│         │█████│
 │     │       │     │       │  .  │         │.....│         │█████│
 │     │       │ .   │       │. . .│         │.....│         │█████│
 └─────┘       └─────┘       └─────┘         └─────┘         └─────┘
 No people      Few people    Stream of       Packed,         Wall-to-wall,
 or vehicles    scattered     people,         long queues     no gaps,
 visible        around        some queues     visible         standstill
```

#### Crowd Activity — "What are people doing and where?"

```
 Camera 2701 (Causeway):                  Camera 2702 (Checkpoint):
 ┌────────────────────────────────┐      ┌────────────────────────────────┐
 │                                │      │                                │
 │   SG ◄──🚶🚶🚶── JB          │      │   [CUSTOMS BUILDING]           │
 │   causeway_walking_from_jb     │      │   🚶🚶🚶 → 🚌 → JB           │
 │                                │      │   checkpoint_boarding          │
 │   SG ──🚶🚶🚶──► JB          │      │                                │
 │   causeway_walking_to_jb       │      │   JB → 🚌 → 🚶🚶🚶          │
 │                                │      │   checkpoint_alighting         │
 └────────────────────────────────┘      └────────────────────────────────┘
```

> 📝 **Quick rule of thumb:**
> - Camera **2701** images → usually `causeway_walking_to_jb` or `causeway_walking_from_jb`
> - Camera **2702** images → usually `checkpoint_boarding` or `checkpoint_alighting`
> - **Direction of movement** determines `to_jb` vs `from_jb`

#### Weather — "What are the sky/road conditions?"

```
 ☀️ clear       ☁️ cloudy      🌧️ rainy       💧 wet_road     🌫️ hazy        🌙 night
 ──────────    ──────────    ──────────    ──────────     ──────────    ──────────
 Bright sky    Grey/over-   Rain drops     Dark shiny     Washed-out   Dark image,
 Sharp         cast sky     visible,       road surface   low contrast  street-
 shadows       No rain      blurry image   (post-rain)    distant blur  lights on
```

> ⚠️ **Weather is multi-select!** Common combinations:
> - `["rainy", "wet_road"]` — active rain
> - `["cloudy", "wet_road"]` — rain just stopped
> - `["rainy", "night"]` — rain at night
> - `["hazy"]` — Southeast Asian haze season
> - `["clear"]` — normal good weather

### 5.5 — Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Save current labels and go to next image |
| `→` Right Arrow | Skip image (no save) |
| `←` Left Arrow | Go back to previous image |
| `Esc` | Clear all selections |
| `1` | Set crowd level → empty |
| `2` | Set crowd level → low |
| `3` | Set crowd level → moderate |
| `4` | Set crowd level → high |
| `5` | Set crowd level → congested |

### 5.6 — How Long Does Labeling Take?

| Images | Estimated Time | Tip |
|--------|---------------|-----|
| 100 | ~10 minutes | Quick test run |
| 300 | ~30 minutes | Minimum viable dataset |
| **500** | **~45 minutes** | **Recommended minimum** |
| 1,000 | ~1.5 hours | Good accuracy |

> 💡 **Speed tip**: Label images in **time-of-day batches**. All 6am–9am images
> will have similar crowd levels, so decisions are faster. Then do 9am–12pm, etc.

### 5.7 — Alternative: Manual JSON Editing

If you prefer not to use the GUI, you can directly edit `training_data/labels.json`:

```json
{
    "images": [
        {
            "filename": "2701_Woodlands_Causeway_Towards_Johor_20260313_080000_original.png",
            "camera_id": "2701",
            "crowd_level": "moderate",
            "crowd_activity": "causeway_walking_to_jb",
            "weather": ["clear"],
            "notes": "Morning rush, clear weather"
        },
        {
            "filename": "2701_Woodlands_Causeway_Towards_Johor_20260313_190000_original.png",
            "camera_id": "2701",
            "crowd_level": "congested",
            "crowd_activity": "causeway_walking_from_jb",
            "weather": ["rainy", "wet_road", "night"],
            "notes": "Friday evening return, raining"
        }
    ]
}
```

---

## 6. ✅ Step 3 — Validate Labels

### What This Does

Automatically checks your `labels.json` for:
- ❌ Missing or invalid label values
- ❌ Image files referenced in JSON but not found on disk
- ❌ Empty weather arrays
- 📊 Shows class distribution (helps spot imbalanced data)

### How to Run

```bash
python validate_labels.py
```

### What You'll See (Success)

```
============================================================
  LABEL VALIDATION REPORT
============================================================

  Total labeled entries: 523

────────────────────────────────────────────────────────────
  📊 CROWD LEVEL DISTRIBUTION:
────────────────────────────────────────────────────────────
    empty        [████░░░░░░░░░░░░░░░░░░░░░]   45 images
    low          [████████░░░░░░░░░░░░░░░░░]  112 images
    moderate     [█████████████░░░░░░░░░░░░]  187 images
    high         [███████░░░░░░░░░░░░░░░░░░]  128 images
    congested    [██░░░░░░░░░░░░░░░░░░░░░░░]   51 images

────────────────────────────────────────────────────────────
  🚶 CROWD ACTIVITY DISTRIBUTION:
────────────────────────────────────────────────────────────
    checkpoint_boarding            [████████░░░░░░░░░░░░░░░░░]  135
    checkpoint_alighting           [██████░░░░░░░░░░░░░░░░░░░]  118
    causeway_walking_to_jb         [███████░░░░░░░░░░░░░░░░░░]  142
    causeway_walking_from_jb       [██████░░░░░░░░░░░░░░░░░░░]  128

────────────────────────────────────────────────────────────
  🌤️ WEATHER TAG DISTRIBUTION:
────────────────────────────────────────────────────────────
    clear      [█████████████░░░░░░░░░░░░░]  198
    cloudy     [████████░░░░░░░░░░░░░░░░░░]  121
    rainy      [███░░░░░░░░░░░░░░░░░░░░░░░]   67
    wet_road   [████░░░░░░░░░░░░░░░░░░░░░░]   89
    hazy       [██░░░░░░░░░░░░░░░░░░░░░░░░]   34
    night      [██████░░░░░░░░░░░░░░░░░░░░]  109

============================================================
  SUMMARY
============================================================
    Total entries:     523
    Missing files:     0
    Validation errors: 0

    ✅ ALL CHECKS PASSED — Ready for training!
    Run:  python jalannow_classifier.py train
============================================================
```

### What You'll See (Errors)

```
  ⚠️  FILE NOT FOUND: 2701_Woodlands_..._20260313_080000_original.png
  ❌ Invalid crowd_level 'medium' in: 2701_..._20260314_120000_original.png
  ❌ Empty/invalid weather in: 2702_..._20260314_180000_original.png

    ⚠️  Please fix the issues above before training.
```

**Common fixes:**
| Error | Fix |
|-------|-----|
| `FILE NOT FOUND` | Copy the image into `training_data/<camera_folder>/` |
| `Invalid crowd_level 'medium'` | Change to `moderate` (must match exactly) |
| `Empty weather` | Add at least one weather tag, e.g. `["clear"]` |

---

## 7. 🧠 Step 4 — Train the Model

### What This Does

Trains a **MobileNetV2** neural network using transfer learning in two phases:

```
 PHASE 1: Transfer Learning (10-20 min)
 ┌───────────────────────────────────────┐
 │  MobileNetV2 base   →  FROZEN ❄️      │ ← Pre-trained ImageNet weights stay fixed
 │  New classification →  TRAINING 🔥    │ ← Only new layers learn your data
 │  heads (3 outputs)                    │
 └───────────────────────────────────────┘
                    │
                    ▼
 PHASE 2: Fine-Tuning (15-30 min)
 ┌───────────────────────────────────────┐
 │  MobileNetV2 top    →  UNFROZEN 🔥   │ ← Top CNN layers adapt to your images
 │  layers              (low LR: 1e-5)  │
 │  Classification     →  TRAINING 🔥   │ ← Heads continue learning
 │  heads                               │
 └───────────────────────────────────────┘
                    │
                    ▼
 OUTPUT: Saved .keras model file + training plots
```

### How to Run

```bash
python jalannow_classifier.py train
```

### What You'll See

```
============================================================
JalanNow Traffic Classifier — TRAINING MODE
============================================================
Loading 523 labeled images...
Dataset loaded: 523 images, shape=(224, 224, 3)
  Crowd levels distribution:    {'empty': 45, 'low': 112, 'moderate': 187, ...}
  Crowd activity distribution:  {'checkpoint_boarding': 135, ...}
  Weather tag counts:           {'clear': 198, 'cloudy': 121, ...}

Training set:   418 images
Validation set: 105 images

============================================================
PHASE 1: Transfer Learning (MobileNetV2 frozen)
============================================================
Epoch 1/50
27/27 [======] - 15s - loss: 2.8432 - crowd_level_accuracy: 0.2344 ...
Epoch 2/50
27/27 [======] - 12s - loss: 2.1205 - crowd_level_accuracy: 0.4102 ...
...
Epoch 18/50
27/27 [======] - 12s - loss: 0.6821 - crowd_level_accuracy: 0.7812 ...

============================================================
PHASE 2: Fine-Tuning (unfreezing from layer 100)
============================================================
Epoch 1/50
27/27 [======] - 18s - loss: 0.5934 - crowd_level_accuracy: 0.8125 ...
...
Epoch 12/50
EarlyStopping: Restoring best weights from epoch 8.

Final model saved: models/jalannow_classifier_20260315_143022.keras
Training history saved: models/training_history_20260315_143022.json
Training plots saved: models/training_curves_20260315_143022.png

✅ Training complete!
```

### Training Time Estimates

| Dataset Size | CPU Only | NVIDIA GPU |
|-------------|----------|------------|
| 300 images | ~30 min | ~8 min |
| 500 images | ~50 min | ~15 min |
| 1,000 images | ~1.5 hours | ~25 min |

### Training Output Files

After training completes, check the `models/` folder:

```
models/
├── jalannow_classifier_20260315_143022.keras      ← THE TRAINED MODEL
├── best_model_20260315_143022.keras               ← Best Phase 1 checkpoint
├── best_model_finetuned_20260315_143022.keras     ← Best Phase 2 checkpoint
├── training_history_20260315_143022.json           ← Loss/accuracy per epoch
└── training_curves_20260315_143022.png             ← Visual training plots
```

> 💡 **Check training_curves.png** — if validation loss is much higher than training
> loss, your model may be overfitting. Collect more diverse images and retrain.

---

## 8. 🔮 Step 5 — Predict on Live Images

### What This Does

1. Fetches the **latest live image** from Camera 2701 and/or 2702 via the API
2. Runs the trained model on the image
3. Generates a **prediction report** as a `.txt` file with:
   - Crowd density level with confidence percentages
   - Crowd activity classification
   - Weather conditions detected
   - Travel advisory with risk score

### How to Run

```bash
# Predict on ALL cameras (2701 + 2702)
python jalannow_classifier.py predict

# Predict on Camera 2701 only (Woodlands Causeway)
python jalannow_classifier.py predict2701

# Predict on Camera 2702 only (Woodlands Checkpoint)
python jalannow_classifier.py predict2702
```

### What You'll See

```
============================================================
JalanNow Traffic Classifier — PREDICTION MODE
============================================================
Loading model: models/jalannow_classifier_20260315_143022.keras
Model loaded successfully!

============================================================
Processing Camera 2701: Woodlands_Causeway_Towards_Johor
============================================================
Fetching latest image for Camera 2701 from data.gov.sg API...
  Camera: Woodlands_Causeway_Towards_Johor
  Timestamp: 2026-03-16T08:00:04+08:00
  Resolution: 2006x1504
  Image preprocessed: shape=(1, 224, 224, 3)
Running model prediction...

================================================================================
  JALANNOW TRAFFIC CAMERA — AI PREDICTION REPORT
================================================================================
  🚀 TRAVEL ADVISORY
  ──────────────────
  🟠 OVERALL TRAVEL RISK: HIGH (62/100)
  CROWD STATUS: 🔴 Heavy crowd — significant delays likely.
  ACTIVITY: 🚶 Pedestrians WALKING ON CAUSEWAY towards JB
  WEATHER: 🌧️ RAIN — Bring umbrella. Slippery conditions.

  📊 CROWD DENSITY: HIGH (78.3% confidence)
  🚶 CROWD ACTIVITY: causeway_walking_to_jb (85.1% confidence)
  🌤️ WEATHER: rainy (72.4%), wet_road (89.1%)
================================================================================

Prediction report saved: predictions/prediction_2701_..._20260316_080045.txt
JSON results saved:      predictions/prediction_2701_..._20260316_080045.json
```

### Output Files

Each prediction generates two files in the `predictions/` folder:

| File | Format | Purpose |
|------|--------|---------|
| `prediction_2701_..._20260316_080045.txt` | Human-readable text | Full report with visuals |
| `prediction_2701_..._20260316_080045.json` | Machine-readable JSON | For downstream apps |

---

## 9. 📄 Understanding the Output Report

The `.txt` report contains **6 sections**:

```
Section 1: CAMERA & IMAGE INFORMATION
         → Camera ID, timestamp, GPS coordinates, resolution

Section 2: 🚀 TRAVEL ADVISORY
         → Overall risk level (LOW/MODERATE/HIGH/CRITICAL)
         → Plain-English advice for travelers

Section 3: 📊 CROWD DENSITY LEVEL
         → Predicted class + confidence
         → Bar chart of all class probabilities
         → What the prediction means for travelers

Section 4: 🚶 CROWD ACTIVITY TYPE
         → Whether people are in checkpoint or walking on causeway
         → Direction of travel (to JB or from JB)

Section 5: 🌤️ WEATHER & VISIBILITY CONDITIONS
         → All detected weather conditions
         → Impact on travelers (especially causeway pedestrians)

Section 6: 📋 PREDICTION SUMMARY
         → Quick-reference box with all key results
```

---

## 10. 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install tensorflow` |
| `No trained model found!` | Run `python jalannow_classifier.py train` first |
| `No labeled images found in labels.json` | Complete Step 2 (labeling) first |
| `API request failed: ConnectionError` | Check internet connection; API may be temporarily down |
| `Image not found: <filename>` | Copy image from `originals/` to `training_data/` folder |
| `Invalid crowd_level 'medium'` | Use exact spelling: `moderate` (not `medium`) |
| Training is very slow | Normal on CPU. Use GPU or reduce `EPOCHS` to 20 |
| `CUDA out of memory` | Reduce `BATCH_SIZE` from 16 to 8 in `jalannow_classifier.py` |
| Low prediction accuracy | Need more labeled images (aim for 500+); ensure label consistency |

### Checking Your Setup

```bash
# Check Python version (need 3.10 or 3.11)
python --version

# Check all packages are installed
python -c "
import tensorflow as tf
import requests
import PIL
import numpy as np
import matplotlib
import sklearn
import schedule
print('All packages OK!')
print(f'TensorFlow: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
"
```

---

## 11. 📋 Classification Labels Reference

### Crowd Level (single-select)

| Value | Description | Typical Time |
|-------|-------------|-------------|
| `empty` | No visible crowd | 1am – 5am |
| `low` | Few people, lots of open space | 6am – 7am, late evening |
| `moderate` | Noticeable stream, some queuing | Normal weekday |
| `high` | Dense crowd, slow movement | Friday evening, weekend morning |
| `congested` | Wall-to-wall, near standstill | Public holidays, long weekends |

### Crowd Activity (single-select)

| Value | Description | Camera |
|-------|-------------|--------|
| `checkpoint_boarding` | Inside customs, queuing to board bus TO JB | 2702 |
| `checkpoint_alighting` | Inside customs, arriving FROM JB | 2702 |
| `causeway_walking_to_jb` | Walking on causeway towards JB | 2701 |
| `causeway_walking_from_jb` | Walking on causeway from JB | 2701 |

### Weather (multi-select)

| Value | Description |
|-------|-------------|
| `clear` | Bright sky, sharp shadows, dry road |
| `cloudy` | Overcast, no rain |
| `rainy` | Active rain visible, blurry image |
| `wet_road` | Shiny/reflective road surface (post-rain) |
| `hazy` | Low contrast, distant blur |
| `night` | Dark image, artificial lighting |

---

## 12. 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────┐
│           MobileNetV2 (ImageNet)            │
│        Pre-trained Convolutional Base        │
│         (~3.4M parameters, frozen)          │
├─────────────────────────────────────────────┤
│          GlobalAveragePooling2D             │
├─────────────────────────────────────────────┤
│   Dense(512) → BatchNorm → Dropout(0.4)    │
│   Dense(256) → BatchNorm → Dropout(0.3)    │
├──────────┬──────────────┬───────────────────┤
│ Output 1 │  Output 2    │    Output 3       │
│ Crowd    │  Crowd       │    Weather        │
│ Level    │  Activity    │    Conditions     │
│ softmax  │  softmax     │    sigmoid        │
│ (5)      │  (4)         │    (6)            │
│ cat_xent │  cat_xent    │    bin_xent       │
└──────────┴──────────────┴───────────────────┘
```

| Component | Detail |
|-----------|--------|
| Base Model | MobileNetV2 (ImageNet pre-trained) |
| Input Size | 224 × 224 × 3 (RGB) |
| Output Heads | 3 (crowd level, crowd activity, weather) |
| Optimizer | Adam (1e-4 → 1e-5 for fine-tuning) |
| Image Source | [data.gov.sg Traffic Images API](https://data.gov.sg) (free, no key) |
| Framework | TensorFlow / Keras |

---

## 13. 📜 License

This project is for educational and research purposes.

- Traffic camera images are sourced from Singapore's [data.gov.sg](https://data.gov.sg)
  open data platform under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).
- The [data.gov.sg Traffic Images API](https://data.gov.sg) is free and requires no API key.
- MobileNetV2 architecture and ImageNet weights are provided by
  [TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Install
pip install tensorflow requests Pillow numpy matplotlib scikit-learn schedule

# 2. Extract images (let run for 3-7 days, Ctrl+C to stop)
python jalannow_extractor.py

# 3. Copy images to training folder, then label them
python annotation_tool.py

# 4. Validate your labels
python validate_labels.py

# 5. Train the model
python jalannow_classifier.py train

# 6. Predict on latest live image
python jalannow_classifier.py predict
```

**That's it!** Your prediction report `.txt` will be in the `predictions/` folder. 🎉
```

---

This README follows GitHub documentation best practices — clear section structure, table of contents, visual diagrams, copy-pasteable commands, and troubleshooting guidance ([tilburgsciencehub.com](https://tilburgsciencehub.com), [github.blog](https://github.blog), [dev.to](https://dev.to)). The pipeline uses Singapore's free [data.gov.sg](https://data.gov.sg) Traffic Images API ([lta.gov.sg](https://lta.gov.sg)) and MobileNetV2 transfer learning via [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).
