# src/color_extraction.py
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_colors(image_np, k=5, sample_frac=1.0, random_state=42):
    """
    image_np: HxWx3 uint8 RGB
    returns centers (k x 3 int), fractions (k,)
    """
    h, w, c = image_np.shape
    pixels = image_np.reshape(-1, 3).astype(float)

    if sample_frac < 1.0:
        # random sample to speed up
        idx = np.random.choice(pixels.shape[0], int(pixels.shape[0]*sample_frac), replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels_sample)
    centers = kmeans.cluster_centers_.astype(int)

    # If sampled, predict on full to get correct counts
    if pixels_sample.shape[0] != pixels.shape[0]:
        labels_full = kmeans.predict(pixels)
    else:
        labels_full = labels

    counts = np.bincount(labels_full, minlength=k)
    order = np.argsort(counts)[::-1]
    centers = centers[order]
    counts = counts[order]
    fractions = counts / counts.sum()
    return centers, fractions
