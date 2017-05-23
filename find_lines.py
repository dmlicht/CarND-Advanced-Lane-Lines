from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
from pipeline import Pipeline

N_WINDOWS = 9
MIN_PX_RESIZE_WINDOW = 50  # Minimum pixels to resize window
MARGIN = 100
WINDOW_COLOR = (0, 255, 0)

Window = namedtuple("Window", "y_low y_high x_low x_high")


def find_line_from_prior(img, prior_fit):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_inds = ((nonzerox > (prior_fit[0] * (nonzeroy ** 2) + prior_fit[1] * nonzeroy + prior_fit[2] - MARGIN)) & (
        nonzerox < (prior_fit[0] * (nonzeroy ** 2) + prior_fit[1] * nonzeroy + prior_fit[2] + MARGIN)))

    lane_x = nonzerox[lane_inds]
    lane_y = nonzeroy[lane_inds]
    new_fit = np.polyfit(lane_y, lane_x, 2)

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    return new_fit[0] * ploty ** 2 + new_fit[1] * ploty + new_fit[2]


def find_lines_with_sliding_window(img):
    img_height, img_width = img.shape[:2]

    vertical_center = np.int(img_height / 2)
    histogram = np.sum(img[vertical_center:, :], axis=0)
    out_img = np.dstack((img, img, img)) * 255
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_lane_indices = slide_window(img, leftx_base, out_img)
    right_lane_indices = slide_window(img, rightx_base, out_img)

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    left_fit = fit_poly(nonzero_x, nonzero_y, left_lane_indices)
    right_fit = fit_poly(nonzero_x, nonzero_y, right_lane_indices)
    return left_fit, right_fit, out_img


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


def slide_window(img, start_x, out_img=None):
    img_height, img_width = img.shape[:2]
    window_height = np.int(img_height / N_WINDOWS)
    x_current = start_x
    lane_indices = []

    # TODO: does it actually make a performance different to turn these into numpy arrays?
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

        nonzero_indices = nonzero_indices_in_window(window, nonzero_x, nonzero_y)
        lane_indices.append(nonzero_indices)

        if len(nonzero_indices) > MIN_PX_RESIZE_WINDOW:
            x_current = np.int(np.mean(nonzero_x[nonzero_indices]))

    return lane_indices


def nonzero_indices_in_window(window: Window, nonzero_x, nonzero_y):
    return ((nonzero_y >= window.y_low) & (nonzero_y < window.y_high) & (nonzero_x >= window.x_low) & (
        nonzero_x < window.x_high)).nonzero()[0]


def fit_poly(nonzero_x, nonzero_y, lane_indices):
    # Extract left and right line pixel positions
    x_indices = nonzero_x[lane_indices]
    y_indices = nonzero_y[lane_indices]
    return np.polyfit(y_indices, x_indices, 2)


def main():
    img = mpimg.imread('./test_images/straight_lines1.jpg')
    img = Pipeline().transform(img)
    sliding_window(img)
    pass


if __name__ == '__main__':
    main()
