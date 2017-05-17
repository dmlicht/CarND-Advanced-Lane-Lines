from typing import Iterable
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob


class CouldNotFindChessboardCorners(Exception):
    """ A marker for not finding chessboard corners. Ask for forgiveness not permission. """
    pass


def _find_chessboard_corners(img, n_rows, n_cols):
    """ Returns the locations of chessboard corners.
    If we cannot find the corners of the image. Throw CouldNotFindChessboardCorners exception."""

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(grayscale_img, (n_cols, n_rows), None)
    if ret is False:
        raise CouldNotFindChessboardCorners()
    return corners


def _object_points(n_rows, n_cols):
    objp = np.zeros((n_rows * n_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    return objp


def _grayscale_shape(img: np.ndarray) -> tuple:
    return img.shape[:-1][::-1]


class DistortionCorrection:
    def __init__(self):
        self.mtx = None
        self.dist = None

    def fit_to_chessboards(self, images: Iterable[np.ndarray], n_rows: int, n_cols: int) -> None:
        """ Takes a collection of chessboard images with n_rows and n_columns and fits distortion to images ."""

        objpoints = []
        imgpoints = []

        objp = _object_points(n_rows, n_cols)
        img_for_shape = None

        for img in images:
            img_for_shape = img
            try:
                corners = _find_chessboard_corners(img, n_rows, n_cols)
                imgpoints.append(corners)
                objpoints.append(objp)
            except CouldNotFindChessboardCorners:
                pass

        shape = _grayscale_shape(img_for_shape)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        self.mtx = mtx
        self.dist = dist

    def transform(self, img: np.ndarray) -> np.ndarray:
        if self.mtx is None or self.dist is None:
            raise Exception("Did not fit distortion correction yet. Must call `fit_to_chessboards` first.")

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


def path_to_image_gen(image_paths: Iterable[str]):
    for path in image_paths:
        yield mpimg.imread(path)


def test_distortion_correction():
    n_rows = 6
    n_cols = 9
    chessboard_paths = glob.glob('./camera_cal/calibration*.jpg')
    chessboards = path_to_image_gen(chessboard_paths)
    distortion_correction = DistortionCorrection()
    distortion_correction.fit_to_chessboards(chessboards, n_rows, n_cols)


if __name__ == '__main__':
    test_distortion_correction()
