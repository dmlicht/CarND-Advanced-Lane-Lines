import cv2
import numpy as np
import glob
from distortion_correction import DistortionCorrection, path_to_image_gen
from find_lines import find_lines_with_sliding_window, LaneNotFoundException, find_line_from_prior
from perspective_transform import PerspectiveTransform
from threshold import threshold
from collections import deque

SATURATION_THRESHOLD = (50, 355)
DEFAULT_LINE_BUFFER_LEN = 5


class Pipeline:
    def __init__(self, img_shape):
        n_rows = 6
        n_cols = 9
        chessboard_paths = glob.glob('./camera_cal/calibration*.jpg')
        chessboards = path_to_image_gen(chessboard_paths)
        self._distortion_correction = DistortionCorrection()
        self._distortion_correction.fit_to_chessboards(chessboards, n_rows, n_cols)
        self._perspective_transform = PerspectiveTransform(img_shape)

    def highlight_lane(self, forward_view_img: np.ndarray) -> np.ndarray:
        distortion_corrected = self._distortion_correction.transform(forward_view_img)
        forward_view_edges = threshold(distortion_corrected, thresh=SATURATION_THRESHOLD)
        top_down = self._perspective_transform.transform(forward_view_edges)

        try:
            left_fit, right_fit, _ = find_lines_with_sliding_window(top_down)
        except LaneNotFoundException:
            return distortion_corrected
        top_down_lane_highlight = color_between_lines(top_down.shape, left_fit, right_fit)

        forward_view_lane_highlight = self._perspective_transform.inverse_transform(top_down_lane_highlight)
        return cv2.addWeighted(distortion_corrected, 1, forward_view_lane_highlight, 0.3, 0)

    def show_top_down(self, forward_view_img: np.ndarray) -> np.ndarray:
        distortion_corrected = self._distortion_correction.transform(forward_view_img)
        forward_view_edges = threshold(distortion_corrected, thresh=SATURATION_THRESHOLD)
        return self._perspective_transform.transform(forward_view_edges)


class AveragingPipeline(Pipeline):
    def __init__(self, img_shape, buffer_len=DEFAULT_LINE_BUFFER_LEN):
        super().__init__(img_shape)
        self.left_lane_buffer = deque(maxlen=buffer_len)
        self.right_lane_buffer = deque(maxlen=buffer_len)

    def highlight_lane(self, forward_view_img: np.ndarray) -> np.ndarray:
        distortion_corrected = self._distortion_correction.transform(forward_view_img)
        forward_view_edges = threshold(distortion_corrected, thresh=SATURATION_THRESHOLD)
        top_down = self._perspective_transform.transform(forward_view_edges)

        try:
            if len(self.left_lane_buffer) > 0:
                left_fit = find_line_from_prior(top_down, self.left_lane_buffer[0])
                right_fit = find_line_from_prior(top_down, self.right_lane_buffer[0])
            else:
                left_fit, right_fit, _ = find_lines_with_sliding_window(top_down)
            self.left_lane_buffer.appendleft(left_fit)
            self.right_lane_buffer.appendleft(right_fit)
        except LaneNotFoundException:
            return distortion_corrected

        if len(left_fit) < 1:  # return a blank image we've never detected lanes
            return distortion_corrected

        left_avg = avg_poly(self.left_lane_buffer)
        right_avg = avg_poly(self.right_lane_buffer)

        top_down_lane_highlight = color_between_lines(top_down.shape, left_avg, right_avg)

        forward_view_lane_highlight = self._perspective_transform.inverse_transform(top_down_lane_highlight)
        return cv2.addWeighted(distortion_corrected, 1, forward_view_lane_highlight, 0.3, 0)


def avg_poly(polys):
    n_polys = len(polys)
    n_dimension = len(polys[0])

    return [sum([poly[ii] for poly in polys]) / n_polys for ii in range(n_dimension)]


def color_between_lines(shape, left_fit, right_fit):
    ploty = np.linspace(0, shape[0] - 1, shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Create an image to draw the lines on
    warp_zero = np.zeros(shape).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    return color_warp
