import cv2
from pathlib import Path

from enhancement.clahe import apply_clahe
from detection.person_detector import PersonDetector
from detection.face_detector import FaceDetector
from quality.quality_assessment import assess_image_quality
from quality.person_quality import assess_person_crop
from quality.face_quality import assess_face_quality


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_PATH = Path("data/input/cctv_test.png")

ENHANCED_OUTPUT_PATH = Path(
    "data/output/cctv_enhanced.png"
)

DETECTION_OUTPUT_PATH = Path(
    "data/output/cctv_detected.png"
)

PERSON_CROPS_DIR = Path(
    "data/output/person_crops"
)

FACE_CROPS_DIR = Path(
    "data/output/face_crops"
)


# --------------------------------------------------
# Create output directories
# --------------------------------------------------

PERSON_CROPS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FACE_CROPS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 1. Load original image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Image loaded successfully!")
print(
    f"Original image shape: {image.shape}"
)


# --------------------------------------------------
# 2. Assess original image quality
# --------------------------------------------------

original_quality = assess_image_quality(image)

print("\nOriginal image quality:")
print(
    f"Brightness: "
    f"{original_quality['brightness']:.2f}"
)
print(
    f"Contrast: "
    f"{original_quality['contrast']:.2f}"
)
print(
    f"Sharpness: "
    f"{original_quality['sharpness']:.2f}"
)


# --------------------------------------------------
# 3. Enhance original image
# --------------------------------------------------

enhanced_image = apply_clahe(image)

print("\nImage enhancement completed.")


# --------------------------------------------------
# 4. Assess enhanced image quality
# --------------------------------------------------

enhanced_quality = assess_image_quality(
    enhanced_image
)

print("\nEnhanced image quality:")
print(
    f"Brightness: "
    f"{enhanced_quality['brightness']:.2f}"
)
print(
    f"Contrast: "
    f"{enhanced_quality['contrast']:.2f}"
)
print(
    f"Sharpness: "
    f"{enhanced_quality['sharpness']:.2f}"
)


# --------------------------------------------------
# 5. Save enhanced image
# --------------------------------------------------

success = cv2.imwrite(
    str(ENHANCED_OUTPUT_PATH),
    enhanced_image
)

if not success:
    raise IOError(
        f"Could not save enhanced image: "
        f"{ENHANCED_OUTPUT_PATH}"
    )

print(
    f"Enhanced image saved to: "
    f"{ENHANCED_OUTPUT_PATH}"
)


# --------------------------------------------------
# 6. Initialize detectors
# --------------------------------------------------

person_detector = PersonDetector()
face_detector = FaceDetector()


# --------------------------------------------------
# 7. Detect people
# --------------------------------------------------

detections = person_detector.detect(
    enhanced_image
)

print(
    f"\nNumber of people detected: "
    f"{len(detections)}"
)


# --------------------------------------------------
# 8. Prepare detection visualization
# --------------------------------------------------

detected_image = enhanced_image.copy()


# --------------------------------------------------
# 9. Process each detected person
# --------------------------------------------------

for index, detection in enumerate(
    detections,
    start=1
):

    x1, y1, x2, y2 = map(
        int,
        detection["bbox"]
    )

    confidence = detection["confidence"]


    # --------------------------------------------------
    # Draw person bounding box
    # --------------------------------------------------

    cv2.rectangle(
        detected_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"person {confidence:.2f}"
    )

    cv2.putText(
        detected_image,
        label,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # Extract person crop
    # --------------------------------------------------

    person_crop = enhanced_image[
        y1:y2,
        x1:x2
    ]

    if person_crop.size == 0:

        print(
            f"Person {index}: "
            f"invalid crop, skipped."
        )

        continue


    # --------------------------------------------------
    # Assess person crop quality
    # --------------------------------------------------

    person_quality = assess_person_crop(
        person_crop
    )

    print(
        f"\nPerson {index}: "
        f"{person_quality['width']}x"
        f"{person_quality['height']}, "
        f"brightness="
        f"{person_quality['brightness']:.2f}, "
        f"contrast="
        f"{person_quality['contrast']:.2f}, "
        f"sharpness="
        f"{person_quality['sharpness']:.2f}"
    )


    # --------------------------------------------------
    # Save person crop
    # --------------------------------------------------

    person_path = (
        PERSON_CROPS_DIR
        / f"person_{index:02d}.png"
    )

    success = cv2.imwrite(
        str(person_path),
        person_crop
    )

    if success:
        print(
            f"Saved person crop: "
            f"{person_path}"
        )


    # --------------------------------------------------
    # Detect face inside person crop
    # --------------------------------------------------

    face_detections = face_detector.detect(
        person_crop
    )

    print(
        f"Person {index}: "
        f"{len(face_detections)} "
        f"face(s) detected"
    )


    # --------------------------------------------------
    # Process each detected face
    # --------------------------------------------------

    for face_index, face_detection in enumerate(
        face_detections,
        start=1
    ):

        fx1, fy1, fx2, fy2 = map(
            int,
            face_detection["bbox"]
        )


        # --------------------------------------------------
        # Extract face crop
        # --------------------------------------------------

        face_crop = person_crop[
            fy1:fy2,
            fx1:fx2
        ]

        if face_crop.size == 0:

            print(
                f"Person {index}, "
                f"Face {face_index}: "
                f"invalid crop, skipped."
            )

            continue


        # --------------------------------------------------
        # Assess face quality
        # --------------------------------------------------

        face_quality = assess_face_quality(
            face_crop
        )

        print(
            f"Face {face_index} of Person {index}: "
            f"{face_quality['width']}x"
            f"{face_quality['height']}, "
            f"brightness="
            f"{face_quality['brightness']:.2f}, "
            f"contrast="
            f"{face_quality['contrast']:.2f}, "
            f"sharpness="
            f"{face_quality['sharpness']:.2f}, "
            f"score="
            f"{face_quality['quality_score']:.2f}, "
            f"status="
            f"{face_quality['quality_status']}"
        )


        # --------------------------------------------------
        # Save face crop
        # --------------------------------------------------

        face_path = (
            FACE_CROPS_DIR
            / (
                f"person_{index:02d}"
                f"_face_{face_index:02d}.png"
            )
        )

        success = cv2.imwrite(
            str(face_path),
            face_crop
        )

        if success:
            print(
                f"Saved face crop: "
                f"{face_path}"
            )


# --------------------------------------------------
# 10. Save final detection image
# --------------------------------------------------

success = cv2.imwrite(
    str(DETECTION_OUTPUT_PATH),
    detected_image
)

if not success:
    raise IOError(
        f"Could not save detection image: "
        f"{DETECTION_OUTPUT_PATH}"
    )

print(
    f"\nDetection result saved to: "
    f"{DETECTION_OUTPUT_PATH}"
)