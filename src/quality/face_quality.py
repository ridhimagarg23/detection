import cv2
import numpy as np


def assess_face_quality(face_image):
    """
    Assess the quality of a detected face crop.

    The assessment considers:
    - spatial resolution
    - brightness
    - contrast
    - sharpness

    Returns:
        dict: Face quality metrics, component scores,
        overall score, and quality classification.
    """

    if face_image is None or face_image.size == 0:
        raise ValueError("Invalid or empty face image.")

    # --------------------------------------------------
    # 1. Basic resolution
    # --------------------------------------------------

    height, width = face_image.shape[:2]

    # --------------------------------------------------
    # 2. Convert to grayscale
    # --------------------------------------------------

    gray = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # 3. Brightness
    # --------------------------------------------------

    brightness = float(
        np.mean(gray)
    )

    # --------------------------------------------------
    # 4. Contrast
    # --------------------------------------------------

    contrast = float(
        np.std(gray)
    )

    # --------------------------------------------------
    # 5. Sharpness
    # --------------------------------------------------

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # --------------------------------------------------
    # 6. Resolution score
    # --------------------------------------------------

    min_dimension = min(
        width,
        height
    )

    if min_dimension >= 100:
        resolution_score = 100.0

    elif min_dimension >= 70:
        resolution_score = 70.0

    elif min_dimension >= 50:
        resolution_score = 40.0

    elif min_dimension >= 40:
        resolution_score = 25.0

    else:
        resolution_score = 10.0

    # --------------------------------------------------
    # 7. Brightness score
    # --------------------------------------------------

    brightness_distance = abs(
        brightness - 128.0
    )

    brightness_score = max(
        0.0,
        100.0
        - (brightness_distance / 128.0) * 100.0
    )

    # --------------------------------------------------
    # 8. Contrast score
    # --------------------------------------------------

    contrast_score = min(
        100.0,
        (contrast / 64.0) * 100.0
    )

    # --------------------------------------------------
    # 9. Sharpness score
    # --------------------------------------------------

    sharpness_score = min(
        100.0,
        (sharpness / 200.0) * 100.0
    )

    # --------------------------------------------------
    # 10. Overall quality score
    # --------------------------------------------------

    quality_score = (
        0.25 * resolution_score
        + 0.25 * brightness_score
        + 0.20 * contrast_score
        + 0.30 * sharpness_score
    )

    quality_score = float(
        round(quality_score, 2)
    )

    # --------------------------------------------------
    # 11. Quality classification
    # --------------------------------------------------

    if quality_score >= 70:
        quality_status = "GOOD"

    elif quality_score >= 45:
        quality_status = "FAIR"

    elif quality_score >= 25:
        quality_status = "POOR"

    else:
        quality_status = "UNUSABLE"

    # --------------------------------------------------
    # 12. Enhancement recommendation
    # --------------------------------------------------

    if quality_status == "GOOD":
        enhancement_required = False
        recommendation = "FACE_READY"

    elif quality_status == "FAIR":
        enhancement_required = True
        recommendation = "ENHANCE_OPTIONAL"

    elif quality_status == "POOR":
        enhancement_required = True
        recommendation = "ENHANCE_REQUIRED"

    else:
        enhancement_required = True
        recommendation = "LOW_QUALITY"

    return {
        "width": width,
        "height": height,

        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,

        "resolution_score": resolution_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "sharpness_score": sharpness_score,

        "quality_score": quality_score,
        "quality_status": quality_status,

        "enhancement_required": enhancement_required,
        "recommendation": recommendation,
    }