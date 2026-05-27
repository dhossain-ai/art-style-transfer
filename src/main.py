import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONTENT_IMAGE_PATH, STYLE_IMAGES, OUTPUT_DIR
from utils import load_image, save_image, show_images
from style_transfer import run_style_transfer


def main():
    print("=" * 55)
    print("        🎨 ART STYLE TRANSFER PROJECT")
    print("=" * 55)

    # ── Check for specific style argument ───────────────
    selected = sys.argv[1] if len(sys.argv) > 1 else None

    if selected:
        if selected not in STYLE_IMAGES:
            print(f"\n❌ Unknown style: '{selected}'")
            print(f"   Available styles: {', '.join(STYLE_IMAGES.keys())}")
            return
        styles_to_run = {selected: STYLE_IMAGES[selected]}
        print(f"\n🎯 Running single style: {selected}")
    else:
        styles_to_run = STYLE_IMAGES
        print(f"\n🎯 Running all {len(STYLE_IMAGES)} styles")

    # ── Check content image exists ──────────────────────
    if not os.path.exists(CONTENT_IMAGE_PATH):
        print(f"\n❌ Content image not found!")
        print(f"   Please add your photo to:")
        print(f"   {CONTENT_IMAGE_PATH}")
        return

    # ── Load content image ──────────────────────────────
    print(f"\n📸 Loading content image...")
    content_image = load_image(CONTENT_IMAGE_PATH)
    print(f"   ✅ Loaded: {CONTENT_IMAGE_PATH}")

    # ── Run style transfer for each art style ───────────
    results = {}

    for style_name, style_path in styles_to_run.items():

        # Check style image exists
        if not os.path.exists(style_path):
            print(f"\n⚠️  Style image not found, skipping: {style_path}")
            continue

        # Load style image
        print(f"\n🖼️  Loading style: {style_name}")
        style_image = load_image(style_path)

        # Run style transfer
        result = run_style_transfer(
            content_image,
            style_image,
            style_name=style_name
        )

        # Save result
        output_path = os.path.join(OUTPUT_DIR, f"result_{style_name}.jpg")
        save_image(result, output_path)

        # Show content + style + result
        show_images(content_image, style_image, result)

        results[style_name] = result

    # ── Summary ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("        ✅ COMPLETED!")
    print("=" * 55)
    print(f"\n📁 Results saved in: {OUTPUT_DIR}")
    for style_name in results:
        print(f"   • result_{style_name}.jpg")


if __name__ == "__main__":
    main()