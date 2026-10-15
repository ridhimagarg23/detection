from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path(
    "models/face_detector/yolov8n-face.pt"
)


class FaceDetector:
    """
    Detect faces using a pretrained YOLOv8 face detector.
    """

    def __init__(self, model_path: Path = MODEL_PATH):

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face detection model not found: {model_path}"
            )

        self.model = YOLO(str(model_path))

    def detect(self, image):

        results = self.model(image)

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                detections.append(
                    {
                        "confidence": confidence,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }
                )

        return detections