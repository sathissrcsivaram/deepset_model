import numpy as np


def make_gaussian_heatmap(x, y, height=50, width=50, sigma=1.5):
    yy, xx = np.mgrid[0:height, 0:width]

    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    heatmap = heatmap.astype(np.float32)
    heatmap /= heatmap.max() + 1e-8

    return heatmap


def heatmap_to_xy(heatmap):
    idx = np.argmax(heatmap)
    y, x = np.unravel_index(idx, heatmap.shape)
    return int(x), int(y)

