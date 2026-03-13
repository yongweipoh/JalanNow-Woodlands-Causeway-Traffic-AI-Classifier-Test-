"""Quick validation of labels.json before training."""
import json
import os
from collections import Counter

LABELS_PATH = r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow\training_data\labels.json"
IMAGE_DIRS = [
    r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow\training_data\2701_Woodlands_Causeway_Towards_Johor",
    r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow\training_data\2702_Woodlands_Checkpoint",
    r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow\originals\2701_Woodlands_Causeway_Towards_Johor",
    r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow\originals\2702_Woodlands_Checkpoint",
]

VALID_CROWD_LEVELS = {"empty", "low", "moderate", "high", "congested"}
VALID_ACTIVITIES = {"checkpoint_boarding", "checkpoint_alighting",
                    "causeway_walking_to_jb", "causeway_walking_from_jb"}
VALID_WEATHER = {"clear", "cloudy", "rainy", "wet_road", "hazy", "night"}

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

entries = [e for e in data.get("images", []) if not e.get("filename", "").startswith("EXAMPLE_")]

print(f"{'='*60}")
print(f"  LABEL VALIDATION REPORT")
print(f"{'='*60}")
print(f"\n  Total labeled entries: {len(entries)}\n")

# Check each entry
errors = 0
missing_files = 0
cl_counts = Counter()
ca_counts = Counter()
wx_counts = Counter()

for i, entry in enumerate(entries):
    fname = entry.get("filename", "???")

    # Check file exists
    found = False
    for d in IMAGE_DIRS:
        if os.path.exists(os.path.join(d, fname)):
            found = True
            break
    if not found:
        print(f"  ⚠️  FILE NOT FOUND: {fname}")
        missing_files += 1

    # Validate crowd_level
    cl = entry.get("crowd_level", "")
    if cl not in VALID_CROWD_LEVELS:
        print(f"  ❌ Invalid crowd_level '{cl}' in: {fname}")
        errors += 1
    cl_counts[cl] += 1

    # Validate crowd_activity
    ca = entry.get("crowd_activity", "")
    if ca not in VALID_ACTIVITIES:
        print(f"  ❌ Invalid crowd_activity '{ca}' in: {fname}")
        errors += 1
    ca_counts[ca] += 1

    # Validate weather
    weather = entry.get("weather", [])
    if not isinstance(weather, list) or len(weather) == 0:
        print(f"  ❌ Empty/invalid weather in: {fname}")
        errors += 1
    for w in weather:
        if w not in VALID_WEATHER:
            print(f"  ❌ Invalid weather tag '{w}' in: {fname}")
            errors += 1
        wx_counts[w] += 1

# ── Distribution Report ──
print(f"\n{'─'*60}")
print(f"  📊 CROWD LEVEL DISTRIBUTION:")
print(f"{'─'*60}")
for level in ["empty", "low", "moderate", "high", "congested"]:
    count = cl_counts.get(level, 0)
    bar = "█" * (count // 2) + "░" * max(0, 25 - count // 2)
    print(f"    {level:<12} [{bar}] {count:>4} images")

print(f"\n{'─'*60}")
print(f"  🚶 CROWD ACTIVITY DISTRIBUTION:")
print(f"{'─'*60}")
for act in VALID_ACTIVITIES:
    count = ca_counts.get(act, 0)
    bar = "█" * (count // 2) + "░" * max(0, 25 - count // 2)
    print(f"    {act:<30} [{bar}] {count:>4}")

print(f"\n{'─'*60}")
print(f"  🌤️ WEATHER TAG DISTRIBUTION:")
print(f"{'─'*60}")
for w in VALID_WEATHER:
    count = wx_counts.get(w, 0)
    bar = "█" * (count // 2) + "░" * max(0, 25 - count // 2)
    print(f"    {w:<10} [{bar}] {count:>4}")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"    Total entries:    {len(entries)}")
print(f"    Missing files:    {missing_files}")
print(f"    Validation errors: {errors}")
if errors == 0 and missing_files == 0:
    print(f"\n    ✅ ALL CHECKS PASSED — Ready for training!")
    print(f"    Run:  python jalannow_classifier.py train")
else:
    print(f"\n    ⚠️  Please fix the issues above before training.")
print(f"{'='*60}")