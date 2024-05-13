import torch
from PIL import Image
import numpy as np
from einops import rearrange


def show_originalimage(image):
    image = torch.clamp(image, -1, 1)
    img = image.cpu().numpy().copy()
    img *= np.array([0.229, 0.224, 0.225])[:, None, None]
    img += np.array([0.485, 0.456, 0.406])[:, None, None]

    img = rearrange(img, "c h w -> h w c")
    img = img * 255
    img = img.astype(np.uint8)
    return img
