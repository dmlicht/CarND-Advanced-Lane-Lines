import numpy as np
import cv2

# BOTTOM_LEFT = (250, 680)
# BOTTOM_RIGHT = (1090, 680)
# TOP_LEFT = (595, 450)
# TOP_RIGHT = (715, 450)

BOTTOM_LEFT = (250, 680)
BOTTOM_RIGHT = (1070, 680)
TOP_LEFT = (595, 450)
TOP_RIGHT = (690, 450)


class PerspectiveTransform:
    def __init__(self, shape, padding=200):
        self.height, self.width = shape[:2]
        src = np.float32(
            [BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT])
        dst = np.float32([[padding, self.height], [padding, padding], [self.width - padding, padding],
                          [self.width - padding, self.height]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, (self.width, self.height))

    def inverse_transform(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M_inv, (self.width, self.height))
