import cv2
import numpy as np


def assess_image_quality(image: np.ndarray) -> dict:
    """
    Calculate basic image-quality indicators.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    dict
        Brightness, contrast, and sharpness measurements.
    """

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(grayscale))

    contrast = float(np.std(grayscale))

    sharpness = float(
        cv2.Laplacian(grayscale, cv2.CV_64F).var()
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }