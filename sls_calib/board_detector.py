"""
Calibration board circle detector — connected-components approach.

Uses binary threshold + connected components on the mask to find
bright circular blobs on a dark board. More robust than Canny edges
for low-contrast / shallow-color calibration targets.

Flow:
  1. Grayscale → GaussianBlur(3,3)
  2. Fixed threshold(40) → binary mask (white = bright circles, black = bg)
  3. connectedComponentsWithStats → extract white blobs
  4. Filter by area, width, height, aspect ratio
  5. Center = component centroid

This is designed for SLS-type boards where circles are distinctly
brighter than the dark background, even if their edges are soft.
"""
import math, cv2, numpy as np
from typing import List, Tuple, Optional

Marker = Tuple[Tuple[float, float], float]  # ((cx, cy), area)


class BoardDetector:
    """Connected-component based circle detector for calibration boards."""

    def __init__(self,
                 threshold: int = 40,
                 min_area: float = 100,
                 max_area: float = 20000,
                 min_side: float = 8,
                 max_side: float = 220,
                 min_aspect: float = 0.35,
                 max_aspect: float = 2.85):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_side = min_side
        self.max_side = max_side
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def detect(self, image: np.ndarray,
               debug: bool = False,
               debug_dir: str = "output"
               ) -> Tuple[List[Marker], Optional[np.ndarray]]:
        """
        Detect bright circles on a dark calibration board.

        Args:
            image: BGR or grayscale image.
            debug: if True, save mask + detection visualization.
            debug_dir: output directory for debug images.

        Returns:
            (markers, debug_image) where markers is [((cx,cy), area), ...]
            sorted by area descending.
        """
        if image is None or image.size == 0:
            return [], None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape[:2]

        # --- Step 1: Gaussian blur -----------------------------------------
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0.0)

        # --- Step 2: Fixed threshold → binary mask -------------------------
        _, mask = cv2.threshold(smoothed, self.threshold, 255,
                                cv2.THRESH_BINARY)

        # --- Step 3: Connected components ----------------------------------
        n_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=8)

        # --- Step 4: Filter components -------------------------------------
        markers: List[Marker] = []

        for i in range(1, n_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Area filter
            if area < self.min_area or area > self.max_area:
                continue

            # Width / height bounds
            if w < self.min_side or w > self.max_side:
                continue
            if h < self.min_side or h > self.max_side:
                continue

            # Aspect ratio
            if h > 0:
                aspect = w / h
            else:
                continue
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Center = component centroid
            cx = centroids[i, 0]
            cy = centroids[i, 1]

            markers.append(((float(cx), float(cy)), float(area)))

        # --- Step 5: Deduplicate by centroid proximity ---------------------
        markers.sort(key=lambda x: -x[1])  # largest first
        clean: List[Marker] = []
        for m in markers:
            too_close = False
            for c in clean:
                if math.hypot(m[0][0] - c[0][0], m[0][1] - c[0][1]) < 5.0:
                    too_close = True
                    break
            if not too_close:
                clean.append(m)

        # --- Debug visualization -------------------------------------------
        dbg_img = None
        if debug:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/board_mask.png", mask)

            dbg_img = image.copy() if image.ndim == 3 else \
                       cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            for (cx, cy), area in clean:
                # Green ellipse roughly matching the component
                radius = int(math.sqrt(area / math.pi))
                cv2.ellipse(dbg_img,
                           (int(cx), int(cy)),
                           (radius, radius), 0, 0, 360,
                           (0, 255, 0), 1)
                # Red cross at center
                s = 3
                cv2.line(dbg_img, (int(cx)-s, int(cy)), (int(cx)+s, int(cy)),
                        (0, 0, 255), 1)
                cv2.line(dbg_img, (int(cx), int(cy)-s), (int(cx), int(cy)+s),
                        (0, 0, 255), 1)

            cv2.imwrite(f"{debug_dir}/board_detected.png", dbg_img)

        return clean, dbg_img
