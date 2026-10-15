import cv2
import numpy as np


def assess_person_crop(image: np.ndarray) -> dict:
    """
    Assess the basic visual quality of a person crop.

    Parameters
    ----------
    image : np.ndarray
        BGR person crop.

    Returns
    -------
    dict
        Basic brightness, contrast, sharpness, width, and height.
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid person crop.")

    height, width = image.shape[:2]

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(grayscale))
    contrast = float(np.std(grayscale))
    sharpness = float(
        cv2.Laplacian(grayscale, cv2.CV_64F).var()
    )

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }