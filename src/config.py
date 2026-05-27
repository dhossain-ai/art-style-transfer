import torch
import os

# ── Device ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Base path (goes one level up from src/) ──────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Image settings ───────────────────────────────────
IMAGE_SIZE = 512
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# ── Paths ────────────────────────────────────────────
CONTENT_IMAGE_PATH = os.path.join(BASE_DIR, "images", "content", "my_photo.jpg")

STYLE_IMAGES = {
    "vangogh" : os.path.join(BASE_DIR, "images", "style", "vangogh_starry_night.jpg"),
    "monet"   : os.path.join(BASE_DIR, "images", "style", "monet_water_lilies.jpg"),
    "picasso" : os.path.join(BASE_DIR, "images", "style", "picasso_cubism.jpg"),
    "warhol"  : os.path.join(BASE_DIR, "images", "style", "warhol_popart.jpg"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "images", "output")

# ── Transfer settings ────────────────────────────────
NUM_STEPS        = 600
CONTENT_WEIGHT   = 1
STYLE_WEIGHT     = 50000000
LEARNING_RATE    = 0.01