import cv2
import numpy as np
from pathlib import Path

from insightface.app import FaceAnalysis


class FaceRecognizer:
    """
    Generate face embeddings and compare faces
    using InsightFace.
    """

    def __init__(self):
        print("Initializing Face Recognition model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("Face Recognition model initialized successfully!")

    # --------------------------------------------------
    # Generate embedding
    # --------------------------------------------------

    def get_embedding(self, face_image):
        """
        Generate a normalized embedding for a face image.
        """

        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid or empty face image.")

        faces = self.app.get(face_image)

        if len(faces) == 0:
            return None

        # Select the largest detected face
        face = max(
            faces,
            key=lambda f: (
                f.bbox[2] - f.bbox[0]
            ) * (
                f.bbox[3] - f.bbox[1]
            )
        )

        embedding = face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)

    # --------------------------------------------------
    # Compare two embeddings
    # --------------------------------------------------

    @staticmethod
    def cosine_similarity(
        embedding1,
        embedding2
    ):
        """
        Calculate cosine similarity between
        two normalized face embeddings.
        """

        if embedding1 is None or embedding2 is None:
            return 0.0

        embedding1 = embedding1 / np.linalg.norm(
            embedding1
        )

        embedding2 = embedding2 / np.linalg.norm(
            embedding2
        )

        similarity = np.dot(
            embedding1,
            embedding2
        )

        return float(similarity)

    # --------------------------------------------------
    # Compare face images directly
    # --------------------------------------------------

    def compare_faces(
        self,
        face_image1,
        face_image2
    ):
        """
        Generate embeddings for two face images
        and return their similarity.
        """

        embedding1 = self.get_embedding(
            face_image1
        )

        embedding2 = self.get_embedding(
            face_image2
        )

        if embedding1 is None:
            return {
                "similarity": 0.0,
                "match": False,
                "status": "NO_FACE_IN_IMAGE_1"
            }

        if embedding2 is None:
            return {
                "similarity": 0.0,
                "match": False,
                "status": "NO_FACE_IN_IMAGE_2"
            }

        similarity = self.cosine_similarity(
            embedding1,
            embedding2
        )

        # Initial threshold.
        # We will calibrate this later using our dataset.
        threshold = 0.45

        match = similarity >= threshold

        return {
            "similarity": similarity,
            "match": match,
            "status": "MATCH" if match else "NO_MATCH"
        }