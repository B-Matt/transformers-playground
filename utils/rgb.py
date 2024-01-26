import numpy as np
from blend_modes import multiply

LABEL_COLORS = {
    0: [0, 0, 0],           # Background
    1: [139, 189, 7],       # Fire
    2: [198, 36, 125]       # Smoke
}

LABEL_BLACK_WHITE = {
    0: [0, 0, 0],            # Background
    1: [255, 255, 255],      # Fire
    2: [255, 255, 255]       # Smoke
}

LABEL_TRESHOLDED = {
    0: [0, 0, 0],            # Background
    255: [139, 189, 7],       # Fire
}

def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))
    for k,v in LABEL_COLORS.items():
        mask[np.all(rgb==v, axis=2)] = k

    mask[mask==2] = 0
    return mask

def mask2rgb(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
        rgb[mask==i] = LABEL_COLORS[i]
    return rgb

def mask2bw(mask):
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
        rgb[mask==i] = LABEL_BLACK_WHITE[i]
    return rgb

def bw2rgb(bw):
    mask = np.zeros(bw.shape+(3,), dtype=np.uint8)
    for i in np.unique(bw):
        mask[bw==i] = LABEL_TRESHOLDED[i]
    return mask

def colorize_mask(mask, color):
    colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 4), np.float32)
    colorized_mask[:] = color
    a = multiply(mask, mask, 1.0)
    b = multiply(a, colorized_mask, 1.0)
    return b
