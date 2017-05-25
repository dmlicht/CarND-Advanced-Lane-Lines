from collections import deque

import numpy as np

BUFFER_LEN = 5


class LaneHistory:
    def __init__(self):
        self.detected = False  # was the line detected in the last iteration?
        self.recent_polys = deque(BUFFER_LEN)
        # self.recent_xfitted = deque(BUFFER_LEN)  # x values of the last n fits of the line
        # self.bestx = None  # average x values of the fitted line over the last n iterations
        # self.best_fit = None  # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]  # polynomial coefficients for the most recent fit
        self.radius_of_curvature = None  # radius of curvature of the line in some units
        self.line_base_pos = None  # distance in meters of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')  # difference in fit coefficients between last and new fits
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels

    def has_previous(self):
        return len(self.recent_polys) > 0

    def last_fit(self):
        return self.recent_polys[0]
    
    def new_lane_difference(self, new_lane_poly):
        """ Difference in fit coefficients between last and new fits """
        pass
