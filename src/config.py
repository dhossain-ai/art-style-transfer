import torch

# ── Device ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Image settings ──────────────────────────────────
IMAGE_SIZE = 512          # pixels (height & width)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# ── Paths ────────────────────────────────────────────
CONTENT_IMAGE_PATH = "images/content/my_photo.jpg"

STYLE_IMAGES = {
    "vangogh" : "images/style/vangogh_starry_night.jpg",
    "monet"   : "images/style/monet_water_lilies.jpg",
    "picasso" : "images/style/picasso_cubism.jpg",
    "warhol"  : "images/style/warhol_popart.jpg",
}

OUTPUT_DIR = "images/output/"

# ── Transfer settings ────────────────────────────────
NUM_STEPS        = 300    # optimization steps
CONTENT_WEIGHT   = 1      # how much to keep your photo
STYLE_WEIGHT     = 1000000  # how much to apply art style
LEARNING_RATE    = 0.003