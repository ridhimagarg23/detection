from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"


class PersonDetector:
    """
    Detect people in an image using a YOLO model.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect people in the input image.

        Returns
        -------
        list[dict]
            Detection results containing bounding boxes and confidence scores.
        """

        results = self.model(image)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # COCO class ID 0 corresponds to "person".
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "class": "person",
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return detections