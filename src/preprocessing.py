# src/preprocessing.py
from PIL import Image
import numpy as np

RESIZE_MAX = 800

def load_image_as_rgb(path, resize_max=RESIZE_MAX):
    """
    Returns numpy array (H, W, 3) RGB of the image.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, resize_max / max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return np.array(img)

def pil_from_array(arr):
    from PIL import Image
    return Image.fromarray(arr.astype('uint8'), 'RGB')
