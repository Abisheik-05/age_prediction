from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("MediaPipe is required. Install it with: pip install mediapipe") from exc


class FaceDetector:
    """Detect and crop the largest face from an uploaded image."""

    def __init__(self) -> None:
        self.mp_face_detection = mp.solutions.face_detection
        self._face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )

    def detect_largest_face(self, image: Image.Image) -> Optional[tuple[int, int, int, int]]:
        """Return the largest face bounding box as (x, y, w, h) or None if not found."""
        rgb_image = np.array(image.convert("RGB"))
        results = self._face_detector.process(rgb_image)

        if results.detections is None or len(results.detections) == 0:
            return None

        best_box: Optional[tuple[int, int, int, int]] = None
        best_area = -1

        for detection in results.detections:
            relative_bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = rgb_image.shape
            x = int(relative_bbox.xmin * iw)
            y = int(relative_bbox.ymin * ih)
            width = int(relative_bbox.width * iw)
            height = int(relative_bbox.height * ih)
            area = width * height

            if area > best_area:
                best_area = area
                best_box = (x, y, width, height)

        return best_box

    def crop_largest_face(self, image: Image.Image, margin_ratio: float = 0.15) -> Optional[Image.Image]:
        """Crop and return the largest detected face with a small margin."""
        bounding_box = self.detect_largest_face(image)
        if bounding_box is None:
            return None

        x, y, width, height = bounding_box
        if width <= 0 or height <= 0:
            return None

        image_width, image_height = image.size
        margin_x = max(int(width * margin_ratio), 10)
        margin_y = max(int(height * margin_ratio), 10)

        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(image_width, x + width + margin_x)
        bottom = min(image_height, y + height + margin_y)

        return image.crop((left, top, right, bottom))
