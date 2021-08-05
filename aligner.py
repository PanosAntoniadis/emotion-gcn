import cv2
import numpy as np
from PIL import Image


class FaceAligner:
    """A class that aligns a given facial image based on the position of the eyes."""

    def angle(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def align(self, image, left_eye_center, right_eye_center):
        """
        Args:
            image: The RGB input image.
            left_eye_center: The coordinates of the left eye.
            right_eye_center: The coordinates of the right eye.
        """

        angle = self.angle(left_eye_center, right_eye_center)
        x1, y1 = left_eye_center
        x2, y2 = right_eye_center
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        rotated = cv2.warpAffine(
            image, M,  (image.shape[0], image.shape[1]), flags=cv2.INTER_CUBIC)

        return Image.fromarray(rotated), M
