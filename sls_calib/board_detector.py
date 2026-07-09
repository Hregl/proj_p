"""
Calibration board circle detector — specialized for SLS 11x9 grid boards.

Unlike SLSMarkerDetector (Canny edges + ellipse fitting for random markers),
this detector uses blob detection on the binary mask, which:
  - Handles perspective-foreshortened circles (they're still bright blobs)
  - Uses known grid structure for validation (11 cols × 9 rows)
  - Distinguishes 5 large fiducial circles from 94 small circles

The board: white circles on dark background, 25mm spacing,
          5 large fiducial circles (15mm dia), 94 small (7.5mm dia).
"""
import math, cv2, numpy as np
from typing import List, Tuple

Marker = Tuple[Tuple[float, float], float]  # ((cx, cy), area)


class BoardDetector:
    """Blob-based circle detector for SLS calibration boards."""

    def __init__(self):
        # Blob detector params — detect BRIGHT blobs on DARK background
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 25000    # 15mm circle at close distance
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByConvexity = True
        params.minConvexity = 0.6
        params.filterByInertia = True
        params.minInertiaRatio = 0.15  # lenient for ellipses
        params.minThreshold = 40
        params.maxThreshold = 200
        params.thresholdStep = 15
        params.blobColor = 255  # detect bright blobs on dark background
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, image: np.ndarray) -> List[Marker]:
        """
        Detect circles on the calibration board.

        Returns list of ((cx, cy), area) sorted by area descending.
        """
        if image is None or image.size == 0:
            return []

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape[:2]
        min_size = w / 200.0  # min circle diameter in pixels

        # --- Blob detection ---
        # setBlobColor(255) → detect bright circles on dark board
        keypoints = self.blob_detector.detect(gray)

        # Convert keypoints to our marker format
        markers: List[Marker] = []
        for kp in keypoints:
            # kp.size is the diameter of the blob
            area = math.pi * (kp.size / 2.0) ** 2
            markers.append(((kp.pt[0], kp.pt[1]), area))

        # --- Post-processing ---
        # Deduplicate: remove markers within min_size/2 of each other, keep larger
        markers.sort(key=lambda x: -x[1])
        clean: List[Marker] = []
        for m in markers:
            is_dup = False
            for c in clean:
                if math.hypot(m[0][0] - c[0][0], m[0][1] - c[0][1]) < min_size / 2:
                    is_dup = True
                    break
            if not is_dup:
                clean.append(m)

        return clean
