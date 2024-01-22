import cv2
import numpy as np
from scipy.ndimage import sobel
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor

EPSILON = 1e-5


def get_max_hue(image, n_bin=180):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the hue channel
    hue_channel = hsv_image[:, :, 0]

    # Compute the histogram of the hue channel -> hue is between 0 and 180
    hist, _ = np.histogram(hue_channel.flatten(), bins=n_bin, range=[0, 180])

    # Return the bin index with the maximal occurrence
    return np.argmax(hist)


def filter_image_by_hue_band(image, max_bin_index, band_width=20):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the hue channel
    hue_channel = hsv_image[:, :, 0]

    # Create a binary mask for pixels within the specified band around the maximal HUE occurrence
    lower_bound = float(max_bin_index - band_width)
    upper_bound = float(max_bin_index + band_width)
    mask = cv2.inRange(hue_channel, lower_bound, upper_bound)

    # Return the masked original image
    return cv2.bitwise_and(image, image, mask=mask)


def take_top_n(indices, n_top, top_distance=5):
    # top_distance corresponds to the minimum distance in pixels between 2 tops

    n_horizontal, n_vertical = indices.shape
    top_n = np.full((n_top, n_vertical), -top_distance)

    for x in range(n_vertical):
        k_top = 0
        y = n_horizontal - 1  # start to the end (with the largest gradients)
        while k_top < n_top and y >= 0:
            top = indices[y, x]
            if np.min(np.abs(top - top_n[:, x])) > top_distance:
                top_n[k_top, x] = top
                k_top += 1
            y -= 1

    return top_n


def estimate_horizon(image, n_vertical=8, n_top=2):
    # Normalize between 0 and 1
    image = image / 255.

    # Compute some useful values
    height, width, *_ = image.shape
    vertical_step = int(width / n_vertical)

    # Compute the absolute value of the horizontal gradient and normalize it
    horizontal_grad = np.abs(sobel(image, axis=0))
    horizontal_grad = horizontal_grad / np.max(horizontal_grad)

    # Filter top n_top largest gradient points
    indices = np.argsort(horizontal_grad[:, ::vertical_step][:, :n_vertical], axis=0)
    x = (np.arange(n_vertical * n_top) // n_top) * vertical_step
    y = np.ravel(take_top_n(indices, n_top), order='F')

    # Reshape for visualization purposes
    horizontal_point_x = x.reshape(-1, 1)
    horizontal_point_y = y.reshape(-1, 1)

    # Compute the best line fitting our point using RANSAC
    ransac = RANSACRegressor()
    ransac.fit(horizontal_point_x, horizontal_point_y)
    ransac_pred_y = ransac.predict(np.array([[0.], [width - 1.]]))

    return horizontal_point_x, horizontal_point_y, ransac_pred_y


def compute_saliency_mask(image_bgr, saliency_ceil=0.85):
    channel = [1]  # 0=H ; 1=S ; 2=V
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, channel]
    mean_hsv = np.mean(image_hsv, axis=(0, 1), keepdims=True)
    saliency = np.linalg.norm(image_hsv - mean_hsv, axis=2)
    return (saliency / np.max(saliency) > saliency_ceil).astype(np.uint8)


def update_tracking_window(x, y, s, ps, size_step=16, min_window_size=32):
    # points_old_frame : list of 1x2 arrays -> (n, 1, 2)
    # maybe update size based on (x, y) displacement until a fix size is reached

    if ps is not None:
        center = np.array([x, y]).reshape((1, 1, 2))
        mask = (np.abs(ps - center) < s)
        mask = mask[:, 0, 0] & mask[:, 0, 1]
        barycenter = np.sum(ps[mask], axis=0)

        n_point = np.sum(mask)
        if n_point > 0:
            barycenter /= n_point
            x, y = barycenter.flatten()
            s -= size_step / np.log2(1 + n_point)
        else:
            s += size_step

    return x, y, max(min_window_size, s)
