import cv2


def enhance_face(face_image):
    """
    Enhance a low-quality face crop for downstream processing.

    Pipeline:
    1. 4x upscaling
    2. Mild denoising
    3. LAB-based brightness/contrast enhancement
    4. Controlled sharpening
    """

    if face_image is None or face_image.size == 0:
        raise ValueError("Invalid or empty face image.")

    # --------------------------------------------------
    # 1. Upscale
    # --------------------------------------------------

    height, width = face_image.shape[:2]

    scale_factor = 4

    upscaled = cv2.resize(
        face_image,
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------
    # 2. Mild denoising
    # --------------------------------------------------

    denoised = cv2.bilateralFilter(
        upscaled,
        d=5,
        sigmaColor=35,
        sigmaSpace=35
    )

    # --------------------------------------------------
    # 3. LAB brightness + contrast enhancement
    # --------------------------------------------------

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.2,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )

    contrast_enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------
    # 4. Controlled sharpening
    # --------------------------------------------------

    blurred = cv2.GaussianBlur(
        contrast_enhanced,
        (0, 0),
        sigmaX=1.2
    )

    sharpened = cv2.addWeighted(
        contrast_enhanced,
        1.35,
        blurred,
        -0.35,
        0
    )

    # --------------------------------------------------
    # 5. Final safe intensity range
    # --------------------------------------------------

    enhanced_face = cv2.normalize(
        sharpened,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return enhanced_face.astype("uint8")