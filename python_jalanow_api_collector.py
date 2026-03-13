"""
JalanNow Traffic Camera Image Extractor (API-Based)
=====================================================
Extracts Woodlands Causeway & Checkpoint traffic camera images directly
from Singapore's data.gov.sg API every 30 minutes.

Saves images in optimal quality for Keras ML training/prediction.

API Source: https://api.data.gov.sg/v1/transport/traffic-images
    - Free, no API key required
    - Updates every 20 seconds
    - Camera 2701: Woodlands Causeway (Towards Johor)
    - Camera 2702: Woodlands Checkpoint

Requirements:
    pip install requests Pillow schedule

Author: YongWei
"""

import os
import sys
import json
import time
import hashlib
import logging
import requests
import schedule
from io import BytesIO
from datetime import datetime
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

# API endpoint (free, no API key needed)
API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"

# Camera IDs to capture (matching the JalanNow page layout)
# Image 1: Woodlands Causeway (the large image on the JalanNow page)
# Image 2: Woodlands Checkpoint (the smaller image on the JalanNow page)
TARGET_CAMERAS = {
    "2701": "Woodlands_Causeway_Towards_Johor",     # Image 1
    "2702": "Woodlands_Checkpoint",                  # Image 2
}

# Save directory
SAVE_DIR = r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow"

# Sub-directories per camera (organized for Keras ImageDataGenerator)
USE_CAMERA_SUBDIRS = True

# Timestamp log file
TIMESTAMP_LOG = "timestamps.txt"

# Capture schedule
CAPTURE_INTERVAL_MINUTES = 10   # Capture every 30 minutes
MAX_CAPTURES = 0                # 0 = run indefinitely

# Image quality settings for Keras ML
SAVE_ORIGINAL = True            # Keep full-resolution original (lossless PNG)
SAVE_KERAS_RESIZED = True       # Also save resized version for direct Keras use
KERAS_IMG_SIZE = (224, 224)     # Standard for VGG16, ResNet50, MobileNetV2
IMAGE_FORMAT = "PNG"            # PNG = lossless; best for ML training

# Duplicate detection (skip saving if image hasn't changed)
SKIP_DUPLICATES = True

# Logging configuration
LOG_FILE = os.path.join(SAVE_DIR, "capture_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def setup_directories() -> dict:
    """
    Create the save directory structure.
    
    Structure for Keras ImageDataGenerator compatibility:
        JalanNow/
        ├── originals/
        │   ├── 2701_Woodlands_Causeway_Towards_Johor/
        │   └── 2702_Woodlands_Checkpoint/
        ├── keras_224x224/
        │   ├── 2701_Woodlands_Causeway_Towards_Johor/
        │   └── 2702_Woodlands_Checkpoint/
        ├── timestamps.txt
        └── capture_log.txt
    
    Returns:
        Dict mapping camera_id -> {'original': path, 'keras': path}
    """
    paths = {}
    
    for cam_id, cam_name in TARGET_CAMERAS.items():
        folder_name = f"{cam_id}_{cam_name}"
        
        # Original images directory
        orig_dir = os.path.join(SAVE_DIR, "originals", folder_name) if USE_CAMERA_SUBDIRS \
                   else os.path.join(SAVE_DIR, "originals")
        os.makedirs(orig_dir, exist_ok=True)
        
        # Keras-resized images directory
        keras_dir = os.path.join(SAVE_DIR, f"keras_{KERAS_IMG_SIZE[0]}x{KERAS_IMG_SIZE[1]}", folder_name) \
                    if USE_CAMERA_SUBDIRS else os.path.join(SAVE_DIR, f"keras_{KERAS_IMG_SIZE[0]}x{KERAS_IMG_SIZE[1]}")
        os.makedirs(keras_dir, exist_ok=True)
        
        paths[cam_id] = {"original": orig_dir, "keras": keras_dir}
        logger.info(f"Directory ready: {orig_dir}")
        logger.info(f"Directory ready: {keras_dir}")
    
    return paths


# ============================================================
# HASH TRACKING (avoid saving duplicate images)
# ============================================================

class DuplicateTracker:
    """Track image hashes to avoid saving duplicate/unchanged images."""
    
    def __init__(self, hash_file: str):
        self.hash_file = hash_file
        self.last_hashes = {}
        self._load()
    
    def _load(self):
        """Load previous hashes from file."""
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r') as f:
                    self.last_hashes = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.last_hashes = {}
    
    def _save(self):
        """Persist current hashes to file."""
        with open(self.hash_file, 'w') as f:
            json.dump(self.last_hashes, f)
    
    def is_duplicate(self, camera_id: str, image_bytes: bytes) -> bool:
        """Check if image is same as last captured for this camera."""
        current_hash = hashlib.md5(image_bytes).hexdigest()
        previous_hash = self.last_hashes.get(camera_id)
        
        if current_hash == previous_hash:
            return True
        
        self.last_hashes[camera_id] = current_hash
        self._save()
        return False


# ============================================================
# API INTERACTION
# ============================================================

def fetch_traffic_images() -> dict:
    """
    Fetch live traffic camera data from data.gov.sg API.
    
    API: https://api.data.gov.sg/v1/transport/traffic-images
    - Free, no authentication required
    - Returns JSON with camera images updated every 20 seconds
    - Image URLs are temporary (expire in ~5 minutes)
    
    Returns:
        Dict mapping camera_id -> {
            'image_url': str,
            'timestamp': str,
            'latitude': float,
            'longitude': float,
            'width': int,
            'height': int
        }
    """
    try:
        logger.info(f"Fetching traffic images from API: {API_URL}")
        
        response = requests.get(
            API_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "TrafficCamCollector/1.0"
            },
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Validate API health
        api_status = data.get("api_info", {}).get("status", "unknown")
        if api_status != "healthy":
            logger.warning(f"API status: {api_status}")
        
        # Parse camera data
        cameras = {}
        items = data.get("items", [])
        
        if not items:
            logger.error("No items returned from API")
            return cameras
        
        # Get all cameras from the first (most recent) item
        for camera in items[0].get("cameras", []):
            cam_id = camera.get("camera_id", "")
            
            # Only process our target cameras
            if cam_id in TARGET_CAMERAS:
                metadata = camera.get("image_metadata", {})
                location = camera.get("location", {})
                
                cameras[cam_id] = {
                    "image_url": camera.get("image", ""),
                    "timestamp": camera.get("timestamp", ""),
                    "latitude": location.get("latitude", 0),
                    "longitude": location.get("longitude", 0),
                    "width": metadata.get("width", 0),
                    "height": metadata.get("height", 0),
                    "md5": metadata.get("md5", ""),
                }
                
                logger.info(
                    f"  Found Camera {cam_id} ({TARGET_CAMERAS[cam_id]}): "
                    f"{cameras[cam_id]['width']}x{cameras[cam_id]['height']} "
                    f"@ {cameras[cam_id]['timestamp']}"
                )
        
        # Check if we found all target cameras
        for cam_id in TARGET_CAMERAS:
            if cam_id not in cameras:
                logger.warning(f"  Camera {cam_id} ({TARGET_CAMERAS[cam_id]}) NOT found in API response!")
        
        return cameras
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse API response: {e}")
        return {}


# ============================================================
# IMAGE DOWNLOAD & PROCESSING
# ============================================================

def download_image(image_url: str) -> bytes | None:
    """
    Download image from the temporary S3 URL.
    
    Note: Image URLs from data.gov.sg expire after ~5 minutes,
    so download immediately after fetching.
    
    Args:
        image_url: Temporary URL to the traffic camera image
    
    Returns:
        Raw image bytes, or None if download failed
    """
    try:
        response = requests.get(
            image_url,
            headers={"User-Agent": "TrafficCamCollector/1.0"},
            timeout=30,
            stream=True
        )
        response.raise_for_status()
        
        image_bytes = response.content
        logger.info(f"  Downloaded: {len(image_bytes) / 1024:.1f} KB")
        return image_bytes
    
    except requests.exceptions.RequestException as e:
        logger.error(f"  Image download failed: {e}")
        return None


def process_and_save_image(
    image_bytes: bytes,
    camera_id: str,
    camera_name: str,
    timestamp_str: str,
    paths: dict
) -> dict:
    """
    Process raw image bytes and save in optimal quality for Keras ML.
    
    ML Optimization:
        - Saves as PNG (lossless) to preserve all pixel information
        - Converts to RGB mode (3 channels) for standard CNN input
        - Uses LANCZOS resampling for highest-quality resizing
        - Original + resized versions for flexibility
    
    Args:
        image_bytes: Raw downloaded image bytes
        camera_id: Camera identifier (e.g., "2701")
        camera_name: Human-readable name (e.g., "Woodlands_Causeway_Towards_Johor")
        timestamp_str: ISO timestamp from API (e.g., "2026-03-13T10:44:04+08:00")
        paths: Directory paths dict from setup_directories()
    
    Returns:
        Dict with saved file paths and metadata
    """
    result = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "timestamp": timestamp_str,
        "original_file": None,
        "keras_file": None,
        "original_size": None,
        "status": "failed"
    }
    
    try:
        # Open image with Pillow
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB (drop alpha channel if present)
        # This ensures consistent 3-channel input for Keras models
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        original_size = img.size  # (width, height)
        result["original_size"] = f"{original_size[0]}x{original_size[1]}"
        
        # Create filename from timestamp
        # Parse ISO timestamp: "2026-03-13T10:44:04+08:00"
        try:
            dt = datetime.fromisoformat(timestamp_str)
        except ValueError:
            dt = datetime.now()
        
        ts_filename = dt.strftime("%Y%m%d_%H%M%S")
        base_name = f"{camera_id}_{camera_name}_{ts_filename}"
        
        # ----- Save Original (full resolution, lossless) -----
        if SAVE_ORIGINAL:
            cam_paths = paths.get(camera_id, {})
            orig_dir = cam_paths.get("original", SAVE_DIR)
            orig_path = os.path.join(orig_dir, f"{base_name}_original.png")
            
            img.save(orig_path, format="PNG", optimize=True)
            file_kb = os.path.getsize(orig_path) / 1024
            result["original_file"] = orig_path
            logger.info(f"  ✓ Original saved: {orig_path} ({file_kb:.1f} KB)")
        
        # ----- Save Keras-Resized (e.g., 224x224, lossless) -----
        if SAVE_KERAS_RESIZED:
            cam_paths = paths.get(camera_id, {})
            keras_dir = cam_paths.get("keras", SAVE_DIR)
            keras_path = os.path.join(
                keras_dir,
                f"{base_name}_keras_{KERAS_IMG_SIZE[0]}x{KERAS_IMG_SIZE[1]}.png"
            )
            
            # LANCZOS = highest quality resampling for downscaling
            img_resized = img.resize(KERAS_IMG_SIZE, Image.LANCZOS)
            img_resized.save(keras_path, format="PNG", optimize=True)
            file_kb = os.path.getsize(keras_path) / 1024
            result["keras_file"] = keras_path
            logger.info(f"  ✓ Keras saved:    {keras_path} ({file_kb:.1f} KB)")
        
        result["status"] = "success"
        return result
    
    except Exception as e:
        logger.error(f"  Image processing error: {e}")
        return result


# ============================================================
# TIMESTAMP LOGGING
# ============================================================

def append_timestamp_log(results: list, save_dir: str) -> None:
    """
    Append capture results to the timestamp log file.
    
    Format:
        [Capture Time] Camera ID | Camera Name | Image Timestamp | Original File | Keras File | Status
    
    Args:
        results: List of result dicts from process_and_save_image()
        save_dir: Directory containing the log file
    """
    log_path = os.path.join(save_dir, TIMESTAMP_LOG)
    capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write header if file is new
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    
    with open(log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("=" * 120 + "\n")
            f.write("JalanNow Traffic Camera Capture Log\n")
            f.write(f"Source API: {API_URL}\n")
            f.write(f"Target Cameras: {json.dumps(TARGET_CAMERAS, indent=2)}\n")
            f.write(f"Capture Interval: Every {CAPTURE_INTERVAL_MINUTES} minutes\n")
            f.write(f"Keras Image Size: {KERAS_IMG_SIZE[0]}x{KERAS_IMG_SIZE[1]}\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"{'Capture Time':<22} | {'Cam ID':<7} | {'Camera Name':<40} | "
                    f"{'Image Timestamp':<28} | {'Size':<10} | {'Status':<8} | Files\n")
            f.write("-" * 160 + "\n")
        
        for r in results:
            files = []
            if r.get("original_file"):
                files.append(f"orig: {os.path.basename(r['original_file'])}")
            if r.get("keras_file"):
                files.append(f"keras: {os.path.basename(r['keras_file'])}")
            
            f.write(
                f"{capture_time:<22} | "
                f"{r.get('camera_id', 'N/A'):<7} | "
                f"{r.get('camera_name', 'N/A'):<40} | "
                f"{r.get('timestamp', 'N/A'):<28} | "
                f"{r.get('original_size', 'N/A'):<10} | "
                f"{r.get('status', 'N/A'):<8} | "
                f"{' | '.join(files)}\n"
            )
    
    logger.info(f"Timestamp log updated: {log_path}")


# ============================================================
# MAIN CAPTURE FUNCTION
# ============================================================

def capture_once(paths: dict, tracker: DuplicateTracker) -> int:
    """
    Perform a single capture cycle:
    1. Fetch camera data from API
    2. Download images
    3. Process and save in optimal ML quality
    4. Log timestamps
    
    Args:
        paths: Directory paths from setup_directories()
        tracker: DuplicateTracker instance
    
    Returns:
        Number of images successfully saved
    """
    capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n{'='*60}")
    logger.info(f"CAPTURE CYCLE @ {capture_time}")
    logger.info(f"{'='*60}")
    
    # Step 1: Fetch camera data from API
    cameras = fetch_traffic_images()
    
    if not cameras:
        logger.error("No camera data received. Skipping this cycle.")
        return 0
    
    results = []
    saved_count = 0
    
    for cam_id, cam_data in cameras.items():
        cam_name = TARGET_CAMERAS.get(cam_id, f"Camera_{cam_id}")
        image_url = cam_data.get("image_url", "")
        timestamp = cam_data.get("timestamp", "")
        
        logger.info(f"\nProcessing Camera {cam_id}: {cam_name}")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Image URL: {image_url[:80]}...")
        
        if not image_url:
            logger.warning(f"  No image URL for camera {cam_id}. Skipping.")
            results.append({
                "camera_id": cam_id,
                "camera_name": cam_name,
                "timestamp": timestamp,
                "status": "no_url"
            })
            continue
        
        # Step 2: Download the image
        image_bytes = download_image(image_url)
        
        if image_bytes is None:
            results.append({
                "camera_id": cam_id,
                "camera_name": cam_name,
                "timestamp": timestamp,
                "status": "download_failed"
            })
            continue
        
        # Step 3: Check for duplicates (image unchanged since last capture)
        if SKIP_DUPLICATES and tracker.is_duplicate(cam_id, image_bytes):
            logger.info(f"  ⊘ Duplicate detected (image unchanged). Skipping save.")
            results.append({
                "camera_id": cam_id,
                "camera_name": cam_name,
                "timestamp": timestamp,
                "status": "duplicate_skipped"
            })
            continue
        
        # Step 4: Process and save the image
        result = process_and_save_image(
            image_bytes=image_bytes,
            camera_id=cam_id,
            camera_name=cam_name,
            timestamp_str=timestamp,
            paths=paths
        )
        results.append(result)
        
        if result["status"] == "success":
            saved_count += 1
    
    # Step 5: Log timestamps
    if results:
        append_timestamp_log(results, SAVE_DIR)
    
    logger.info(f"\nCapture cycle complete: {saved_count}/{len(cameras)} images saved")
    return saved_count


# ============================================================
# SCHEDULER
# ============================================================

def run_scheduled(interval_minutes: int = 10, max_captures: int = 0):
    """
    Run image capture on a fixed schedule.
    
    Args:
        interval_minutes: Minutes between captures (default: 30)
        max_captures: Max number of capture cycles (0 = infinite)
    """
    logger.info("=" * 60)
    logger.info("JalanNow Traffic Camera Collector (API-Based)")
    logger.info("=" * 60)
    logger.info(f"API Endpoint:     {API_URL}")
    logger.info(f"Target Cameras:   {list(TARGET_CAMERAS.keys())}")
    logger.info(f"Save Directory:   {SAVE_DIR}")
    logger.info(f"Capture Interval: Every {interval_minutes} minutes")
    logger.info(f"Max Captures:     {'Unlimited' if max_captures == 0 else max_captures}")
    logger.info(f"Keras Size:       {KERAS_IMG_SIZE[0]}x{KERAS_IMG_SIZE[1]}")
    logger.info(f"Image Format:     {IMAGE_FORMAT} (lossless)")
    logger.info(f"Skip Duplicates:  {SKIP_DUPLICATES}")
    logger.info("=" * 60)
    
    # Setup directories
    paths = setup_directories()
    
    # Setup file-based logging (in addition to console)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    # Initialize duplicate tracker
    tracker = DuplicateTracker(os.path.join(SAVE_DIR, "image_hashes.json"))
    
    # Capture count tracker
    capture_count = 0
    total_images = 0
    
    # ---- Perform first capture immediately ----
    logger.info("\n>>> First capture starting immediately...")
    saved = capture_once(paths, tracker)
    capture_count += 1
    total_images += saved
    logger.info(f">>> Capture {capture_count} complete. Total images saved: {total_images}")
    
    if max_captures == 1:
        logger.info("Max captures (1) reached. Exiting.")
        return
    
    # ---- Schedule subsequent captures ----
    def scheduled_capture():
        nonlocal capture_count, total_images
        
        capture_count += 1
        saved = capture_once(paths, tracker)
        total_images += saved
        
        logger.info(f">>> Capture {capture_count} complete. Total images saved: {total_images}")
        
        # Check max captures
        if 0 < max_captures <= capture_count:
            logger.info(f"Max captures ({max_captures}) reached. Stopping scheduler.")
            schedule.clear()
            return schedule.CancelJob
    
    # Schedule the job
    schedule.every(interval_minutes).minutes.do(scheduled_capture)
    
    logger.info(f"\nScheduler running. Next capture in {interval_minutes} minutes.")
    logger.info("Press Ctrl+C to stop.\n")
    
    try:
        while schedule.get_jobs():
            schedule.run_pending()
            time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        logger.info("\n\nCapture stopped by user (Ctrl+C)")
    
    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total capture cycles:  {capture_count}")
    logger.info(f"Total images saved:    {total_images}")
    logger.info(f"Save directory:        {SAVE_DIR}")
    logger.info(f"Timestamp log:         {os.path.join(SAVE_DIR, TIMESTAMP_LOG)}")
    logger.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_scheduled(
        interval_minutes=CAPTURE_INTERVAL_MINUTES,  # Every 10 minutes
        max_captures=MAX_CAPTURES                    # 0 = run forever
    )