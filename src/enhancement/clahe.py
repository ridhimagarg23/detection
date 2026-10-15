import cv2
import numpy as np


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Enhance local image contrast using CLAHE.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image loaded using OpenCV.

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image.
    """

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l_channel, a_channel, b_channel)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image