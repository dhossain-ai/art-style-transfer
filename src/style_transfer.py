import torch
import torch.optim as optim
from config import (DEVICE, NUM_STEPS, CONTENT_WEIGHT,
                    STYLE_WEIGHT, LEARNING_RATE)
from model import get_vgg19, get_features, gram_matrix


def compute_content_loss(target_features, content_features):
    return torch.mean(
        (target_features['conv4_2'] - content_features['conv4_2']) ** 2
    )


def compute_style_loss(target_features, style_features):
    style_loss = 0
    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    for layer in layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram  = gram_matrix(style_features[layer])
        _, c, h, w  = target_features[layer].shape
        layer_loss  = torch.mean((target_gram - style_gram) ** 2)
        layer_loss  = layer_loss / (4 * (c * h * w) ** 2)
        style_loss += layer_loss

    return style_loss


def run_style_transfer(content_image, style_image,
                       style_name="output", progress_callback=None):
    print(f"\n🎨 Starting style transfer: {style_name}")
    print(f"   Steps: {NUM_STEPS} | Device: {DEVICE}")

    model            = get_vgg19()
    content_features = get_features(content_image, model)
    style_features   = get_features(style_image,   model)
    target           = content_image.clone().requires_grad_(True)
    optimizer        = optim.Adam([target], lr=LEARNING_RATE)

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        target_features = get_features(target, model)
        content_loss    = compute_content_loss(target_features, content_features)
        style_loss      = compute_style_loss(target_features, style_features)
        total_loss      = (CONTENT_WEIGHT * content_loss +
                           STYLE_WEIGHT   * style_loss)
        total_loss.backward()
        optimizer.step()

        # ── Progress callback for UI ──
        if progress_callback:
            progress_callback(step, NUM_STEPS)

        if step % 50 == 0:
            print(f"   Step {step:>3}/{NUM_STEPS} | "
                  f"Content Loss: {content_loss.item():.4f} | "
                  f"Style Loss: {style_loss.item():.4f} | "
                  f"Total: {total_loss.item():.4f}")

    print(f"✅ Done: {style_name}")
    return target