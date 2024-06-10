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
import cv2

GABOR_VALUES_1 = {
    "size":17, "lmbda":7, "theta":130, "omega":2, "sigma":1, "gamma":0.7
}

GABOR_VALUES_2 = {
    "size":17, "lmbda":13, "theta":60, "omega":11, "sigma":1, "gamma":1.2
}
GABOR_VALUES_3 = {
    "size":19, "lmbda":8, "theta":130, "omega":5, "sigma":2, "gamma":0.8
}

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

def apply_linear_transformation(image_array, brightness, contrast):
    """Apply a simple linear transformation adjusting the contrast and brightness to the image."""
    transformed = image_array * contrast + brightness * contrast
    transformed[transformed > 255] = 255
    transformed[transformed < 0] = 0
    return transformed.astype(np.uint8)

def equalize_histogram(image_array):
    """Equalize the histogram of an image."""
    image = Image.fromarray(image_array)
    equalized_image = image.convert("L").point(lambda x: x * 256 // 255)
    return np.stack([np.array(equalized_image).astype(np.uint8)]*3, axis=-1)

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

def apply_sobel_own_implentation(image_array, mode = 0 ):
    """Apply Sobel edge detection filter to the image.
    mode = 0: filter on both x and y axis
    mode = 1: filter on x axis
    mode = 2: filter on y axis
    """
    # Convert the image to grayscale
    gray_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    # Define the Sobel filters
    vertical_edge_sobel = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    horizontal_edge_sobel = np.transpose(vertical_edge_sobel)
    if mode == 0:
        vertical_edges = cv2.filter2D(gray_image, -1, vertical_edge_sobel)
        horizontal_edges = cv2.filter2D(gray_image, -1, horizontal_edge_sobel)
    # Apply the Sobel filters
    if mode == 1:
        vertical_edges = cv2.filter2D(gray_image, -1, vertical_edge_sobel)
        horizontal_edges = np.zeros_like(vertical_edges)
    if mode == 2:
        horizontal_edges = cv2.filter2D(gray_image, -1, horizontal_edge_sobel)
        vertical_edges = np.zeros_like(horizontal_edges)
    # Calculate the magnitude of the edges
    sobel_edges = np.sqrt(vertical_edges**2 + horizontal_edges**2)
    return np.stack([sobel_edges.astype(np.uint8)]*3, axis=-1)
    

def apply_gabor_filter(image_array, frequency=0.6):
    """Apply Gabor filter to the image."""
    gray_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    filt_real, filt_imag = gabor(gray_image, frequency=frequency)
    gabor_magnitude = np.sqrt(filt_real**2 + filt_imag**2)
    gabor_magnitude = (gabor_magnitude / np.max(gabor_magnitude) * 255).astype(np.uint8)
    return np.stack([gabor_magnitude]*3, axis=-1)


def create_gabor_filter(size, lmbda=10, theta=30, omega=10, sigma=10, gamma=0.5):
    theta = np.radians(theta)

    center_coor = size // 2
    xx, yy = np.mgrid[:size, :size]
    xx -= center_coor
    yy -= center_coor

    x_prime = xx * np.cos(theta) + yy * np.sin(theta)
    y_prime = -xx * np.sin(theta) + yy * np.cos(theta)

    exp_part = np.exp(
        -(x_prime ** 2 + gamma ** 2 * y_prime ** 2) / (2 * sigma ** 2)
    )
    cos_part = np.cos(2 * np.pi * (x_prime / lmbda) + omega)
    return exp_part * cos_part


def apply_gabor_own_implementation(image_array, size=19, lmbda=10, theta=30, omega=10, sigma=10, gamma=0.5):
    """Apply Gabor filter to the image."""
    image_gray =  image_array.mean(axis=2)
    gabor_filter = create_gabor_filter(size, lmbda, theta, omega, sigma, gamma)
    image_gabor = cv2.filter2D(image_gray, -1, gabor_filter)
    return np.stack([image_gabor]*3, axis=-1).astype(np.uint8)

