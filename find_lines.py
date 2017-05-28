from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg

N_WINDOWS = 9
MIN_PX_RESIZE_WINDOW = 50  # Minimum pixels to resize window
MARGIN = 100
WINDOW_COLOR = (0, 255, 0)
YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

Window = namedtuple("Window", "y_low y_high x_low x_high")


class LaneNotFoundException(Exception):
    pass


def find_lines_with_sliding_window(img):
    """ Returns two lane lines by sliding a window across a top down image. """
    leftx_base, rightx_base = _window_centers(img)

    out_img = np.dstack((img, img, img)) * 255
    left_lane_indices = _slide_window(img, leftx_base, out_img)
    right_lane_indices = _slide_window(img, rightx_base, out_img)

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    try:
        left_fit = fit_poly(img, left_lane_indices)
        right_fit = fit_poly(img, right_lane_indices)
    except:
        raise LaneNotFoundException()
    return left_fit, right_fit, out_img


def find_line_from_prior(img, previous_line):
    active_indices = _active_pixels_in_line_margin(img, previous_line, 70)
    try:
        new_fit = fit_poly(img, active_indices)
    except TypeError:
        raise LaneNotFoundException()
    return new_fit
    # ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # return new_fit[0] * ploty ** 2 + new_fit[1] * ploty + new_fit[2]


def _active_pixels_in_line_margin(img, line, margin=MARGIN):
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    return ((nonzero_x > (line[0] * (nonzero_y ** 2) + line[1] * nonzero_y + line[2] - margin)) &
            (nonzero_x < (line[0] * (nonzero_y ** 2) + line[1] * nonzero_y + line[2] + margin)))


def _window_centers(img):
    """ Finds the x locations where we want to start our sliding window.
    We do this by splitting the window in half and taking the x location with the most
    active pixels. This is where we believe the line will occur. """

    img_height, img_width = img.shape[:2]

    vertical_center = np.int(img_height / 2)
    histogram = np.sum(img[vertical_center:, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


def _slide_window(img, start_window_center_x, out_img=None):
    img_height, img_width = img.shape[:2]
    window_height = np.int(img_height / N_WINDOWS)
    x_current = start_window_center_x
    lane_indices = []

    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    for window_ii in range(N_WINDOWS):
        win_y_low = img_height - (window_ii + 1) * window_height
        win_y_high = img_height - window_ii * window_height
        window = Window(
            y_low=win_y_low,
            y_high=win_y_high,
            x_low=x_current - MARGIN,
            x_high=x_current + MARGIN
        )

        if out_img is not None:
            cv2.rectangle(out_img, (window.x_low, window.y_low), (window.x_high, window.y_high), WINDOW_COLOR, 2)

        active_pixel_indices = active_pixels_in_window(window, nonzero_x, nonzero_y)
        lane_indices.append(active_pixel_indices)

        if len(active_pixel_indices) > MIN_PX_RESIZE_WINDOW:
            x_current = np.int(np.mean(nonzero_x[active_pixel_indices]))

    return lane_indices


def sliding_window(img):
    left_fit, right_fit, out_img = find_lines_with_sliding_window(img)

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # out_img[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
    # out_img[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def calculate_fitx(img_shape, fit):
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    return fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]


def fit_poly(img, lane_indices) -> np.ndarray:
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Extract left and right line pixel positions
    x_indices = nonzero_x[lane_indices]
    y_indices = nonzero_y[lane_indices]
    return np.polyfit(y_indices, x_indices, 2)


def curve_radius(lane_fit):
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    return ((1 + (2 * lane_fit[0] * y_eval + lane_fit[1]) ** 2) ** 1.5) / np.absolute(2 * lane_fit[0])


def camera_center(img_shape, left_fit, right_fit):
    left_fitx = calculate_fitx(img_shape, left_fit)
    right_fitx = calculate_fitx(img_shape, right_fit)
    camera_pos = (left_fitx[-1] + right_fitx[-1]) / 2
    img_x_center = img_shape[1] / 2
    center_diff = (camera_pos - img_x_center) * XM_PER_PIX
    return center_diff


def active_pixels_in_window(window: Window, nonzero_x, nonzero_y):
    ys_in_window = (nonzero_y >= window.y_low) & (nonzero_y < window.y_high)
    xs_in_window = (nonzero_x >= window.x_low) & (nonzero_x < window.x_high)
    return (ys_in_window & xs_in_window).nonzero()[0]


def main():
    img = mpimg.imread('./test_images/straight_lines1.jpg')
    # img = Pipeline().transform(img)
    # sliding_window(img)
    pass


if __name__ == '__main__':
    main()
