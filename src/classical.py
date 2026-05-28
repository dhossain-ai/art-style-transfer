import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image):
    """Convert PIL image to OpenCV format."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image):
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def load_image_pil(path, size=512):
    """Load and resize image as PIL."""
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size))
    return image


# ── Method 1: Reinhard Color Transfer ───────────────────────
def reinhard_color_transfer(content_pil, style_pil):
    """
    Transfer color statistics (mean & std) from style to content.
    Works in LAB color space — closest to human perception.
    """
    # Convert to LAB color space
    content_cv = pil_to_cv2(content_pil).astype(np.float32)
    style_cv   = pil_to_cv2(style_pil).astype(np.float32)

    content_lab = cv2.cvtColor(content_cv, cv2.COLOR_BGR2LAB)
    style_lab   = cv2.cvtColor(style_cv,   cv2.COLOR_BGR2LAB)

    # Split into channels
    c_l, c_a, c_b = cv2.split(content_lab)
    s_l, s_a, s_b = cv2.split(style_lab)

    # Transfer mean and std for each channel
    def transfer_channel(c_chan, s_chan):
        c_mean, c_std = c_chan.mean(), c_chan.std()
        s_mean, s_std = s_chan.mean(), s_chan.std()
        return ((c_chan - c_mean) * (s_std / (c_std + 1e-6))) + s_mean

    result_l = transfer_channel(c_l, s_l)
    result_a = transfer_channel(c_a, s_a)
    result_b = transfer_channel(c_b, s_b)

    # Merge and convert back
    result_lab = cv2.merge([result_l, result_a, result_b])
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

    return cv2_to_pil(result_bgr)


# ── Method 2: Histogram Matching ────────────────────────────
def histogram_matching(content_pil, style_pil):
    """
    Match the histogram of content image to style image.
    Each color channel is matched separately.
    """
    content_cv = pil_to_cv2(content_pil)
    style_cv   = pil_to_cv2(style_pil)

    result = np.zeros_like(content_cv)

    # Match each channel independently
    for channel in range(3):
        c_chan = content_cv[:, :, channel]
        s_chan = style_cv[:, :, channel]

        # Calculate CDFs
        c_hist, _ = np.histogram(c_chan.flatten(), 256, [0, 256])
        s_hist, _ = np.histogram(s_chan.flatten(), 256, [0, 256])

        c_cdf = c_hist.cumsum()
        s_cdf = s_hist.cumsum()

        # Normalize CDFs
        c_cdf = c_cdf / c_cdf[-1]
        s_cdf = s_cdf / s_cdf[-1]

        # Build lookup table
        lookup = np.zeros(256, dtype=np.uint8)
        s_idx = 0
        for c_idx in range(256):
            while s_idx < 255 and s_cdf[s_idx] < c_cdf[c_idx]:
                s_idx += 1
            lookup[c_idx] = s_idx

        result[:, :, channel] = lookup[c_chan]

    return cv2_to_pil(result)


# ── Method 3: Pencil Sketch ──────────────────────────────────
def pencil_sketch(content_pil, style_pil=None):
    """
    Convert content image into a pencil sketch.
    Style image is not used — purely edge-based.
    """
    content_cv = pil_to_cv2(content_pil)

    # Convert to grayscale
    gray = cv2.cvtColor(content_cv, cv2.COLOR_BGR2GRAY)

    # Invert grayscale
    inverted = cv2.bitwise_not(gray)

    # Blur the inverted image
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Blend to get sketch effect
    sketch = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    # Convert back to RGB PIL
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(sketch_rgb)


# ── Run any classical method ─────────────────────────────────
CLASSICAL_METHODS = {
    "reinhard" : reinhard_color_transfer,
    "histogram": histogram_matching,
    "sketch"   : pencil_sketch,
}

def run_classical(method_name, content_pil, style_pil=None):
    """Run selected classical method and return PIL result."""
    if method_name not in CLASSICAL_METHODS:
        raise ValueError(f"Unknown method: {method_name}. "
                         f"Choose from: {list(CLASSICAL_METHODS.keys())}")

    print(f"\n🖌️  Running classical method: {method_name}")
    result = CLASSICAL_METHODS[method_name](content_pil, style_pil)
    print(f"   ✅ Done: {method_name}")
    return result