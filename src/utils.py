import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from config import DEVICE, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

def load_image(path, size=IMAGE_SIZE):
    """Load image, resize and convert to tensor."""
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

    # Add batch dimension → shape becomes [1, 3, H, W]
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE)


def tensor_to_image(tensor):
    """Convert tensor back to viewable image."""
    image = tensor.clone().detach()
    image = image.squeeze(0)  # remove batch dimension

    # Reverse the normalization
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGE_STD).view(3, 1, 1)
    image = image.cpu() * std + mean

    # Clamp values to valid range [0, 1]
    image = torch.clamp(image, 0, 1)

    # Convert to PIL image
    image = transforms.ToPILImage()(image)
    return image


def save_image(tensor, path):
    """Save tensor as image file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = tensor_to_image(tensor)
    image.save(path)
    print(f"Image saved → {path}")


def show_images(content, style, output=None):
    """Display content, style and result side by side."""
    cols = 3 if output is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(15, 5))

    axes[0].imshow(tensor_to_image(content))
    axes[0].set_title("Content (Your Photo)")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_image(style))
    axes[1].set_title("Style (Artwork)")
    axes[1].axis("off")

    if output is not None:
        axes[2].imshow(tensor_to_image(output))
        axes[2].set_title("Result")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()