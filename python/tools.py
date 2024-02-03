import cv2
import numpy as np
from scipy.ndimage import sobel
from sklearn.linear_model import RANSACRegressor

EPSILON = 1e-5


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


class HorizonEstimator:
    def __init__(self, n_vertical=8, n_top=3, max_residual=4., max_slope_change=6e-2, max_bias_change=10.):
        self.n_vertical = n_vertical
        self.n_top = n_top

        self.max_residual = max_residual
        self.max_slope_change = max_slope_change  # no units
        self.max_bias_change = max_bias_change  # in pixels

        self.slope = 0.  # looking for a horizontal line
        self.bias = None  # no ideas what this value should be

    def compute(self, image):
        def is_data_valid(X, Y):
            dx = X[-1, 0] - X[0, 0]
            dy = Y[-1, 0] - Y[0, 0]

            if self.bias is None:
                return True
            else:
                if abs(dx) > EPSILON:
                    a = (dy / dx)
                    b = Y[0, 0] - a * X[0, 0]
                    return (abs(self.slope - a) < self.max_slope_change) \
                           and (abs(self.bias - b) < self.max_bias_change)
                else:
                    return False

        # Normalize between 0 and 1
        image = image / 255.

        # Compute some useful values
        height, width, *_ = image.shape
        horizontal_half_step = int(0.5 * width / self.n_vertical)

        # Compute the horizontal gradient -> higher = from dark to bright -> Guerledan hypothesis
        # horizontal_grad = sobel(image, axis=0)
        horizontal_grad = np.abs(sobel(image, axis=0))

        # Filter top n_top largest gradient points
        indices = np.argsort(horizontal_grad[:, horizontal_half_step:][:, ::2 * horizontal_half_step], axis=0)
        x = horizontal_half_step + (np.arange(self.n_vertical * self.n_top) // self.n_top) * (2 * horizontal_half_step)
        y = np.ravel(take_top_n(indices, self.n_top), order='F')

        # Reshape for visualization purposes
        horizontal_point_x = x.reshape(-1, 1)
        horizontal_point_y = y.reshape(-1, 1)

        # Compute the best line fitting our point using RANSAC
        ransac = RANSACRegressor(residual_threshold=self.max_residual, is_data_valid=is_data_valid)

        try:
            ransac.fit(horizontal_point_x, horizontal_point_y)
        except ValueError:
            print('RANSAC value error!')
            self.slope = 0.
            self.bias = None
            return self.compute(image)
        else:
            # Update slope
            ransac_pred_y = ransac.predict(np.array([[0.], [width - 1.]]))
            self.bias, y_w = ransac_pred_y.flatten()
            self.slope = (y_w - self.bias) / width

            # Compute horizon mask
            mask = (np.arange(height).reshape(-1, 1) > ransac.predict(np.arange(width).reshape(-1, 1)).T)

            return horizontal_point_x, horizontal_point_y, ransac_pred_y.reshape(-1, 1, 1), mask.astype(np.uint8)


class SaliencyEstimator:
    def __init__(self, n_sigma=3.5):
        self.n_sigma = n_sigma

    def compute(self, img_bgr, mask):
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = mask.astype(bool)

        mean_hsv = np.mean(img_hsv[mask], axis=0).reshape((1, 1, 3))
        std_hsv = np.std(img_hsv[mask], axis=0).reshape((1, 1, 3))

        saliency = np.linalg.norm((img_hsv - mean_hsv) / std_hsv, axis=2)
        saliency_mask = (saliency > self.n_sigma).astype(np.uint8)

        return saliency_mask


class Buffer:
    def __init__(self, n_buffer, shape):
        self.k_buffer = 0
        self.n_buffer = n_buffer
        self.is_buffer_full = False
        self.memory = np.empty(shape=(n_buffer, *shape))

    def push(self, values):
        n_values = values.shape[0]

        if n_values > self.n_buffer:
            n_values = self.n_buffer
            values = values[:n_values]

        if self.k_buffer + n_values < self.n_buffer:
            self.memory[self.k_buffer:self.k_buffer + n_values] = values
        else:
            delta = self.n_buffer - self.k_buffer
            self.memory[self.k_buffer:self.n_buffer] = values[:delta]
            self.memory[:n_values-delta] = values[delta:]

        self.k_buffer += n_values
        if self.k_buffer >= self.n_buffer:
            self.is_buffer_full = True
            self.k_buffer -= self.n_buffer

        if self.is_buffer_full:
            return self.memory
        else:
            return self.memory[:self.k_buffer]


class TrackingWindow:
    def __init__(self, height, width, alpha=0.5, size_scale=3., size_search_factor=1.25):
        self.alpha = alpha
        self.size_scale = size_scale
        self.size_search_factor = size_search_factor

        self.size_min = np.array([width, height]).reshape((1, 1, 2)) / 64.

        self.center = np.array([width, height]).reshape((1, 1, 2)) / 2.
        self.size = np.array([width, height]).reshape((1, 1, 2)) / 2.

        self.buffer = Buffer(n_buffer=(width // 6), shape=(1, 2))

    def update(self, ps, output_scale=1.):
        if ps is not None:
            mask = (np.abs(ps - self.center) < self.size)
            mask = mask[:, 0, 0] & mask[:, 0, 1]

            points_in_window = ps[mask]
            if points_in_window.size > 0:
                self.center = (1. - self.alpha) * self.center + self.alpha * np.mean(points_in_window, axis=0)

                buffered_ps = self.buffer.push(points_in_window)
                buffer_size = np.std(buffered_ps, axis=0) / 2.  # convert into radius
                self.size = self.size_min + buffer_size * self.size_scale
            else:
                self.size *= self.size_search_factor

        return *(self.center * output_scale).flatten(), *(self.size * output_scale).flatten()
