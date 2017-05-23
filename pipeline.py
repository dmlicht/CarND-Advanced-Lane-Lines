import cv2
import numpy as np
import glob
from distortion_correction import DistortionCorrection, path_to_image_gen

bottom_left = (250, 680)
bottom_right = (1060, 680)
top_left = (595, 450)
top_right = (685, 450)


class Pipeline:
    def __init__(self):
        n_rows = 6
        n_cols = 9
        chessboard_paths = glob.glob('./camera_cal/calibration*.jpg')
        chessboards = path_to_image_gen(chessboard_paths)
        self.distortion_correction = DistortionCorrection()
        self.distortion_correction.fit_to_chessboards(chessboards, n_rows, n_cols)

    def transform(self, img: np.ndarray) -> np.ndarray:
        corrected = self.distortion_correction.transform(img)
        thresholded = threshold(corrected, thresh=(30, 200))
        perspective_thresholded, M = perspective_transform(thresholded)
        return perspective_thresholded


def perspective_transform(img: np.ndarray):
    height, width = img.shape[:2]

    src = np.float32([bottom_left, top_left, top_right, bottom_right])
    dst = np.float32([[200, height], [200, 200], [width - 200, 200], [width - 200, height]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, M


def threshold(img: np.ndarray, thresh=(90, 255)):
    """ Very simple filler threshold function to get the ball rolling on the pipeline """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    s_thresh = np.zeros_like(S)
    s_thresh[(S > thresh[0]) & (S <= thresh[1])] = 1
    return s_thresh
