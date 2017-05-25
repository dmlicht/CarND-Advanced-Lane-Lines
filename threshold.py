import cv2
import numpy as np

SATURATION_MIN_THRESHOLD = 80
SATURATION_MAX_THRESHOLD = 355


def to_binary(img: np.ndarray):
    """ Very simple filler threshold function to get the ball rolling on the pipeline """
    thresholded_sat = threshold(saturation(img), SATURATION_MIN_THRESHOLD, SATURATION_MAX_THRESHOLD)

    sobelx = gradient(img, kernel_size=5)
    sobely = gradient(img, direction='y', kernel_size=5)
    sobelxy = gradient_magnitude(sobelx, sobely)
    thresholdedxy = threshold(sobelxy, 10, 20)

    combined = np.zeros(thresholdedxy.shape)
    combined[(thresholded_sat == 1) | (thresholdedxy == 1)] = 1

    return combined


def gradient(img, direction="x", kernel_size=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if direction == "x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif direction == "y":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    else:
        raise Exception("Bad Direction")
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    return scaled_sobel


def threshold(img, thresh_min, thresh_max):
    in_threshold = np.zeros(img.shape)
    in_threshold[(img >= thresh_min) & (img <= thresh_max)] = 1
    return in_threshold


def gradient_magnitude(gradx, grady):
    return np.sqrt(gradx ** 2 + grady ** 2)


def gradient_direction(gradx, grady):
    return np.arctan2(grady, gradx)


def saturation(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls[:, :, 2]
