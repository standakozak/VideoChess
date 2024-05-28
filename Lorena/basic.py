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


# Initialize video reader for OBS Virtual Camera
#video_source = 'video=OBS Virtual Camera'  # Adjust this based on your OS and virtual camera setup
video_source = '0'
try:
    #video_reader = imageio.get_reader(f'dshow://{video_source}', 'ffmpeg')
    video_reader = imageio.get_reader(f'avfoundation:{video_source}', 'ffmpeg')
except Exception as e:
    print(f"Error: {e}")
    print("Make sure the virtual camera is running and the correct path is provided.")
    exit()

# Iterate over frames in the video
with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')
    for i, frame in enumerate(video_reader):
        # Apply linear transformation (e.g., scaling by a factor of 0.5)
        transformed_frame = apply_linear_transformation(frame, scale_factor=0.5)
        # Convert frame to grayscale if needed
        gray_frame = np.dot(transformed_frame[..., :3], [0.299, 0.587, 0.114])

        # Equalize the histogram of the grayscale frame
        equalized_frame = equalize_histogram(gray_frame)

        # Calculate statistics using numba
        mean_value, std_dev, max_value, min_value = calculate_statistics(equalized_frame)

        # Calculate mode
        mode_value = calculate_mode(equalized_frame)

        # Calculate entropy
        entropy_value = calculate_entropy(equalized_frame)

        # Convert the frame to an image
        image = Image.fromarray(equalized_frame)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        # Overlay statistics on the frame
        overlay_text = (
            f"Frame {i+1}\n"
            f"Mean: {mean_value:.2f}\n"
            f"Mode: {mode_value}\n"
            f"Std Dev: {std_dev:.2f}\n"
            f"Max: {max_value}\n"
            f"Min: {min_value}"
            f"Entropy: {entropy_value:.2f}"
        )
        y_text = 10
        for line in overlay_text.split('\n'):
            draw.text((10, y_text), line, font=font, fill=(255, 255, 255))
            y_text += 20

        # Convert the image back to a numpy array
        frame_with_text = np.array(image)

        # Plot histogram
        hist_image = plot_histogram(transformed_frame)

        # Combine the frame and histogram image
        combined_image = np.vstack((frame_with_text, hist_image))

        # Send the frame to the virtual camera
        cam.send(combined_image)
        cam.sleep_until_next_frame()


# Close the video reader
video_reader.close()
