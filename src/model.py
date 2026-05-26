import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE

def get_vgg19():
    """Load pretrained VGG19 and freeze all weights."""
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    vgg = vgg.to(DEVICE)

    # Freeze — we NEVER train VGG19, only use it
    for param in vgg.parameters():
        param.requires_grad_(False)

    return vgg


# These layers are used to extract features
# (these are the best layers found by researchers)
CONTENT_LAYERS = {
    '21': 'conv4_2'   # captures content/structure
}

STYLE_LAYERS = {
    '0' : 'conv1_1',  # captures fine textures
    '5' : 'conv2_1',  # captures patterns
    '10': 'conv3_1',  # captures complex textures
    '19': 'conv4_1',  # captures style elements
    '28': 'conv5_1'   # captures high-level style
}


def get_features(image, model):
    """Pass image through VGG19 and collect features at key layers."""
    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in CONTENT_LAYERS:
            features[CONTENT_LAYERS[name]] = x
        if name in STYLE_LAYERS:
            features[STYLE_LAYERS[name]] = x

    return features


def gram_matrix(tensor):
    """
    Gram matrix captures style (textures, colors).
    It measures which features appear together.
    """
    _, channels, height, width = tensor.size()
    tensor = tensor.view(channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram