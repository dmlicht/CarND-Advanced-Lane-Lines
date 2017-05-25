import cv2
import numpy as np


def threshold(img: np.ndarray, thresh=(80, 355)):
    """ Very simple filler threshold function to get the ball rolling on the pipeline """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    s_thresh = np.zeros_like(S)
    s_thresh[(S > thresh[0]) & (S <= thresh[1])] = 1
    return s_thresh
