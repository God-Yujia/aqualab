"""Classical CPU baseline for AquaLab's internal experiments."""

import cv2
import numpy as np
from PIL import Image


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def gray_world_clahe(image: Image.Image) -> Image.Image:
    """White-balance a prepared RGB image, then equalize its Lab lightness."""
    rgb = np.asarray(image, dtype=np.float32)
    channel_means = rgb.mean(axis=(0, 1), dtype=np.float64)
    gains = np.divide(
        channel_means.mean(), channel_means,
        out=np.ones_like(channel_means), where=channel_means > 0,
    )
    balanced = np.rint(np.clip(rgb * gains, 0, 255)).astype(np.uint8)

    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
