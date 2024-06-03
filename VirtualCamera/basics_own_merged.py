#pip install imageio imageio-ffmpeg numpy numba pyvirtualcam pillow scipy matplotlib scikit-image

import imageio 
import numpy as np
from numba import jit
import pyvirtualcam
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage.filters import gabor


@jit(nopython=True)
def calculate_statistics(frame):
    """Calculate statistics using numba for speedup."""
    mean_value = np.mean(frame)
    std_dev = np.std(frame)
    max_value = np.max(frame)
    min_value = np.min(frame)
    return mean_value, std_dev, max_value, min_value

def calculate_mode(arr):
    """Calculate the mode of an array."""
    flattened = arr.flatten()
    data = Counter(flattened)
    mode = data.most_common(1)
    return mode[0][0]

def calculate_entropy(frame):
    """Calculate the entropy of a grayscale frame."""
    histogram, _ = np.histogram(frame, bins=256, range=(0, 256), density=True)
    return entropy(histogram)

def apply_linear_transformation(image_array, scale_factor):
    """Apply a simple linear transformation (scaling) to the image."""
    height, width, _ = image_array.shape
    new_size = (int(width * scale_factor), int(height * scale_factor))
    image = Image.fromarray(image_array)
    image = image.resize(new_size, Image.ANTIALIAS)
    return np.array(image)

def equalize_histogram(image_array):
    """Equalize the histogram of an image."""
    image = Image.fromarray(image_array)
    equalized_image = image.convert("L").point(lambda x: x * 256 // 255)
    return np.array(equalized_image)

def plot_histogram(image_array, ax):
    """Plot histogram for each RGB channel."""
    fig, ax = plt.subplots()
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            image_array[:, :, i], bins=256, range=(0, 255)
        )
        ax.plot(bin_edges[0:-1], histogram, color=color)
    ax.set_xlim(0, 255)
    ax.set_title('Histogram for RGB channels')
    fig.canvas.draw()
    hist_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    hist_image = hist_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return hist_image

def apply_sobel_filter(image_array):
    """Apply Sobel edge detection filter to the image."""
    gray_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    sx = sobel(gray_image, axis=0, mode='constant')
    sy = sobel(gray_image, axis=1, mode='constant')
    sobel_edges = np.hypot(sx, sy)
    sobel_edges = (sobel_edges / np.max(sobel_edges) * 255).astype(np.uint8)
    return np.stack([sobel_edges]*3, axis=-1)

def apply_gabor_filter(image_array, frequency=0.6):
    """Apply Gabor filter to the image."""
    gray_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    filt_real, filt_imag = gabor(gray_image, frequency=frequency)
    gabor_magnitude = np.sqrt(filt_real**2 + filt_imag**2)
    gabor_magnitude = (gabor_magnitude / np.max(gabor_magnitude) * 255).astype(np.uint8)
    return np.stack([gabor_magnitude]*3, axis=-1)
