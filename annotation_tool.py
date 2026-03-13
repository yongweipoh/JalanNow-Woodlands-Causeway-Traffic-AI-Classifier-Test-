"""
JalanNow Image Annotation Helper — Visual GUI Tool
====================================================
Opens each image, shows it on screen, and lets you click buttons
to assign crowd_level, crowd_activity, and weather labels.
Automatically writes to labels.json.

Requirements:
    pip install Pillow

Author: YongWei
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from datetime import datetime


# ============================================================
# CONFIGURATION (match your project paths)
# ============================================================

BASE_DIR = r"C:\Users\YPOHA\Desktop\YongWei\Practice Module Sem 1\JalanNow"
DEFAULT_IMAGE_DIRS = [
    os.path.join(BASE_DIR, "originals", "2701_Woodlands_Causeway_Towards_Johor"),
    os.path.join(BASE_DIR, "originals", "2702_Woodlands_Checkpoint"),
]
LABELS_FILE = os.path.join(BASE_DIR, "training_data", "labels.json")

# Classification options
CROWD_LEVELS = ["empty", "low", "moderate", "high", "congested"]
CROWD_ACTIVITIES = [
    "checkpoint_boarding",
    "checkpoint_alighting",
    "causeway_walking_to_jb",
    "causeway_walking_from_jb",
]
WEATHER_OPTIONS = ["clear", "cloudy", "rainy", "wet_road", "hazy", "night"]

# Camera ID extraction from filename
def extract_camera_id(filename: str) -> str:
    """Extract camera ID from filename like '2701_Woodlands_..._original.png'"""
    if filename.startswith("2701"):
        return "2701"
    elif filename.startswith("2702"):
        return "2702"
    return "unknown"


# ============================================================
# LABELS FILE MANAGEMENT
# ============================================================

class LabelsManager:
    """Load, manage, and save labels.json"""

    def __init__(self, labels_path: str):
        self.path = labels_path
        self.data = {"images": []}
        self.labeled_filenames = set()
        self._load()

    def _load(self):
        """Load existing labels from JSON file."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {"images": []}

        # Build set of already-labeled filenames (skip examples)
        self.labeled_filenames = {
            entry["filename"]
            for entry in self.data.get("images", [])
            if not entry.get("filename", "").startswith("EXAMPLE_")
        }

    def is_labeled(self, filename: str) -> bool:
        return filename in self.labeled_filenames

    def add_label(self, entry: dict):
        """Add a new label entry and save to disk immediately."""
        # Remove any existing entry for this filename (allow re-labeling)
        self.data["images"] = [
            e for e in self.data["images"]
            if e.get("filename") != entry["filename"]
        ]
        self.data["images"].append(entry)
        self.labeled_filenames.add(entry["filename"])
        self._save()

    def _save(self):
        """Write labels to JSON file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get_count(self) -> int:
        return len([
            e for e in self.data.get("images", [])
            if not e.get("filename", "").startswith("EXAMPLE_")
        ])


# ============================================================
# MAIN GUI APPLICATION
# ============================================================

class AnnotationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JalanNow Image Annotation Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")

        # State
        self.image_files = []       # List of (full_path, filename)
        self.current_index = 0
        self.labels_mgr = LabelsManager(LABELS_FILE)

        # Tkinter variables for selections
        self.crowd_level_var = tk.StringVar(value="")
        self.crowd_activity_var = tk.StringVar(value="")
        self.weather_vars = {w: tk.BooleanVar(value=False) for w in WEATHER_OPTIONS}
        self.notes_var = tk.StringVar(value="")
        self.show_unlabeled_only = tk.BooleanVar(value=True)

        # Build the GUI
        self._build_ui()

        # Load images
        self._scan_images()

        if self.image_files:
            self._show_current_image()

    # ────────────────────────────────────────────
    # UI CONSTRUCTION
    # ────────────────────────────────────────────

    def _build_ui(self):
        """Build the complete user interface."""

        # ── Top Bar (progress + controls) ──
        top_frame = tk.Frame(self.root, bg="#313244", height=50)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        self.progress_label = tk.Label(
            top_frame, text="Loading...", font=("Segoe UI", 11),
            bg="#313244", fg="#cdd6f4"
        )
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.total_label = tk.Label(
            top_frame, text="", font=("Segoe UI", 11),
            bg="#313244", fg="#a6adc8"
        )
        self.total_label.pack(side=tk.LEFT, padx=10)

        # Unlabeled filter checkbox
        cb = tk.Checkbutton(
            top_frame, text="Show unlabeled only",
            variable=self.show_unlabeled_only,
            command=self._scan_images,
            bg="#313244", fg="#cdd6f4", selectcolor="#45475a",
            font=("Segoe UI", 10)
        )
        cb.pack(side=tk.LEFT, padx=20)

        # Load directory button
        btn_load = tk.Button(
            top_frame, text="📁 Load Image Folder",
            command=self._browse_directory,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 10, "bold")
        )
        btn_load.pack(side=tk.RIGHT, padx=10)

        # ── Main Content Area ──
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # LEFT: Image display
        self.image_frame = tk.Frame(main_frame, bg="#11111b", width=700)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.filename_label = tk.Label(
            self.image_frame, text="", font=("Consolas", 9),
            bg="#11111b", fg="#a6adc8", wraplength=680
        )
        self.filename_label.pack(pady=(5, 0))

        self.canvas = tk.Canvas(self.image_frame, bg="#11111b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # RIGHT: Label controls
        control_frame = tk.Frame(main_frame, bg="#313244", width=450)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        control_frame.pack_propagate(False)

        # ── Crowd Level Section ──
        self._add_section_header(control_frame, "📊 CROWD DENSITY LEVEL (pick one)")

        for level in CROWD_LEVELS:
            emoji = {"empty": "⬜", "low": "🟢", "moderate": "🟡", "high": "🟠", "congested": "🔴"}
            rb = tk.Radiobutton(
                control_frame, text=f"  {emoji.get(level, '')} {level.upper()}",
                variable=self.crowd_level_var, value=level,
                bg="#313244", fg="#cdd6f4", selectcolor="#45475a",
                font=("Segoe UI", 11), anchor="w",
                activebackground="#45475a", activeforeground="#cdd6f4"
            )
            rb.pack(fill=tk.X, padx=20, pady=1)

        # ── Crowd Activity Section ──
        self._add_section_header(control_frame, "🚶 CROWD ACTIVITY (pick one)")

        activity_short = {
            "checkpoint_boarding": "🚌 Checkpoint → Boarding bus TO JB",
            "checkpoint_alighting": "🛬 Checkpoint → Arriving FROM JB",
            "causeway_walking_to_jb": "🚶➡️ Walking on causeway TO JB",
            "causeway_walking_from_jb": "🚶⬅️ Walking on causeway FROM JB",
        }
        for activity in CROWD_ACTIVITIES:
            rb = tk.Radiobutton(
                control_frame, text=f"  {activity_short.get(activity, activity)}",
                variable=self.crowd_activity_var, value=activity,
                bg="#313244", fg="#cdd6f4", selectcolor="#45475a",
                font=("Segoe UI", 10), anchor="w",
                activebackground="#45475a", activeforeground="#cdd6f4"
            )
            rb.pack(fill=tk.X, padx=20, pady=1)

        # ── Weather Section (multi-select checkboxes) ──
        self._add_section_header(control_frame, "🌤️ WEATHER CONDITIONS (select all that apply)")

        weather_emoji = {
            "clear": "☀️", "cloudy": "☁️", "rainy": "🌧️",
            "wet_road": "💧", "hazy": "🌫️", "night": "🌙"
        }
        for weather in WEATHER_OPTIONS:
            cb = tk.Checkbutton(
                control_frame,
                text=f"  {weather_emoji.get(weather, '')} {weather.upper()}",
                variable=self.weather_vars[weather],
                bg="#313244", fg="#cdd6f4", selectcolor="#45475a",
                font=("Segoe UI", 11), anchor="w",
                activebackground="#45475a", activeforeground="#cdd6f4"
            )
            cb.pack(fill=tk.X, padx=20, pady=1)

        # ── Notes ──
        self._add_section_header(control_frame, "📝 NOTES (optional)")

        self.notes_entry = tk.Entry(
            control_frame, textvariable=self.notes_var,
            font=("Segoe UI", 10), bg="#45475a", fg="#cdd6f4",
            insertbackground="#cdd6f4"
        )
        self.notes_entry.pack(fill=tk.X, padx=20, pady=5)

        # ── Spacer ──
        tk.Frame(control_frame, height=15, bg="#313244").pack()

        # ── Action Buttons ──
        btn_frame = tk.Frame(control_frame, bg="#313244")
        btn_frame.pack(fill=tk.X, padx=20, pady=5)

        # SAVE & NEXT (main action)
        self.save_btn = tk.Button(
            btn_frame, text="✅ SAVE & NEXT →",
            command=self._save_and_next,
            bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 13, "bold"),
            height=2, cursor="hand2"
        )
        self.save_btn.pack(fill=tk.X, pady=(0, 5))

        # Navigation buttons
        nav_frame = tk.Frame(btn_frame, bg="#313244")
        nav_frame.pack(fill=tk.X)

        tk.Button(
            nav_frame, text="◄ PREV", command=self._prev_image,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
            width=12, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            nav_frame, text="SKIP ►", command=self._skip_image,
            bg="#f9e2af", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
            width=12, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            nav_frame, text="🗑️ CLEAR", command=self._clear_selections,
            bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
            width=12, cursor="hand2"
        ).pack(side=tk.LEFT)

        # ── Keyboard shortcuts info ──
        shortcut_label = tk.Label(
            control_frame,
            text="Shortcuts: Enter=Save&Next | ←→=Navigate | Esc=Clear",
            font=("Consolas", 8), bg="#313244", fg="#6c7086"
        )
        shortcut_label.pack(side=tk.BOTTOM, pady=5)

        # ── Bind keyboard shortcuts ──
        self.root.bind("<Return>", lambda e: self._save_and_next())
        self.root.bind("<Right>", lambda e: self._skip_image())
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Escape>", lambda e: self._clear_selections())

        # Number keys for crowd level quick-select
        for i, level in enumerate(CROWD_LEVELS):
            self.root.bind(str(i + 1), lambda e, l=level: self.crowd_level_var.set(l))

    def _add_section_header(self, parent, text):
        """Add a styled section header."""
        tk.Frame(parent, height=8, bg="#313244").pack()
        tk.Label(
            parent, text=text, font=("Segoe UI", 10, "bold"),
            bg="#313244", fg="#89b4fa", anchor="w"
        ).pack(fill=tk.X, padx=15, pady=(5, 3))
        ttk.Separator(parent).pack(fill=tk.X, padx=15)

    # ────────────────────────────────────────────
    # IMAGE SCANNING & LOADING
    # ────────────────────────────────────────────

    def _browse_directory(self):
        """Open a folder browser to select image directory."""
        folder = filedialog.askdirectory(
            title="Select Image Folder",
            initialdir=os.path.join(BASE_DIR, "originals")
        )
        if folder:
            DEFAULT_IMAGE_DIRS.clear()
            DEFAULT_IMAGE_DIRS.append(folder)
            self._scan_images()
            if self.image_files:
                self.current_index = 0
                self._show_current_image()

    def _scan_images(self):
        """Scan directories for image files."""
        self.image_files = []
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for img_dir in DEFAULT_IMAGE_DIRS:
            if not os.path.exists(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_extensions:
                    continue
                if fname.startswith("EXAMPLE_"):
                    continue

                # Filter: show unlabeled only?
                if self.show_unlabeled_only.get() and self.labels_mgr.is_labeled(fname):
                    continue

                full_path = os.path.join(img_dir, fname)
                self.image_files.append((full_path, fname))

        self._update_progress()

    def _show_current_image(self):
        """Display the current image on the canvas."""
        if not self.image_files:
            self.canvas.delete("all")
            self.canvas.create_text(
                350, 300, text="No images to label!\n\nAll done ✅ or no images found.",
                font=("Segoe UI", 16), fill="#a6adc8"
            )
            self.filename_label.config(text="")
            return

        # Clamp index
        self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
        full_path, fname = self.image_files[self.current_index]

        # Update filename display
        self.filename_label.config(text=fname)

        # Load and display image
        try:
            img = Image.open(full_path)
            # Fit image to canvas
            canvas_w = self.canvas.winfo_width() or 680
            canvas_h = self.canvas.winfo_height() or 550
            img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=self.tk_image, anchor=tk.CENTER
            )
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                350, 300, text=f"Error loading image:\n{e}",
                font=("Segoe UI", 12), fill="#f38ba8"
            )

        # Pre-fill if already labeled (for editing)
        self._clear_selections()
        if self.labels_mgr.is_labeled(fname):
            self._prefill_labels(fname)

        self._update_progress()

    def _prefill_labels(self, fname):
        """Pre-fill selections if image was previously labeled."""
        for entry in self.labels_mgr.data.get("images", []):
            if entry.get("filename") == fname:
                self.crowd_level_var.set(entry.get("crowd_level", ""))
                self.crowd_activity_var.set(entry.get("crowd_activity", ""))
                for w in WEATHER_OPTIONS:
                    self.weather_vars[w].set(w in entry.get("weather", []))
                self.notes_var.set(entry.get("notes", ""))
                break

    def _update_progress(self):
        """Update progress counter display."""
        total_labeled = self.labels_mgr.get_count()
        total_images = len(self.image_files)
        current = self.current_index + 1 if total_images > 0 else 0
        self.progress_label.config(
            text=f"Image {current} of {total_images} (in current view)"
        )
        self.total_label.config(
            text=f"Total labeled: {total_labeled} images"
        )

    # ────────────────────────────────────────────
    # ACTIONS
    # ────────────────────────────────────────────

    def _save_and_next(self):
        """Validate selections, save label, and move to next image."""
        if not self.image_files:
            return

        _, fname = self.image_files[self.current_index]

        # ── Validate ──
        crowd_level = self.crowd_level_var.get()
        crowd_activity = self.crowd_activity_var.get()
        weather_selected = [w for w, var in self.weather_vars.items() if var.get()]

        errors = []
        if not crowd_level:
            errors.append("❌ Please select a CROWD LEVEL")
        if not crowd_activity:
            errors.append("❌ Please select a CROWD ACTIVITY")
        if not weather_selected:
            errors.append("❌ Please select at least one WEATHER condition")

        if errors:
            messagebox.showwarning(
                "Missing Labels",
                "\n".join(errors)
            )
            return

        # ── Save ──
        entry = {
            "filename": fname,
            "camera_id": extract_camera_id(fname),
            "crowd_level": crowd_level,
            "crowd_activity": crowd_activity,
            "weather": weather_selected,
            "notes": self.notes_var.get().strip(),
            "labeled_at": datetime.now().isoformat(),
        }
        self.labels_mgr.add_label(entry)

        # Flash the save button green
        self.save_btn.config(bg="#40a02b", text="✅ SAVED!")
        self.root.after(300, lambda: self.save_btn.config(
            bg="#a6e3a1", text="✅ SAVE & NEXT →"
        ))

        # ── Move to next ──
        if self.show_unlabeled_only.get():
            # Rescan to remove the just-labeled image
            self._scan_images()
            # Stay at same index (next unlabeled image slides into position)
            self.current_index = min(self.current_index, len(self.image_files) - 1)
        else:
            self.current_index += 1

        self._clear_selections()
        self._show_current_image()

    def _skip_image(self):
        """Skip to next image without saving."""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._show_current_image()

    def _prev_image(self):
        """Go to previous image."""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self._show_current_image()

    def _clear_selections(self):
        """Reset all label selections."""
        self.crowd_level_var.set("")
        self.crowd_activity_var.set("")
        for var in self.weather_vars.values():
            var.set(False)
        self.notes_var.set("")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()