"""
Sub-pixel marker point refinement from manual click seeds.

Replaces raw mouse-click positions with ellipse-fitted centers.
Saves both seed and refined positions for traceability.

Flow:
  1. User clicks near marker → seed position
  2. Extract local ROI around seed
  3. Adaptive threshold → find bright circular blob
  4. Fit ellipse → sub-pixel center
  5. Show overlay for user confirmation
  6. Store both seed + refined + confidence metrics
"""
import math, cv2, numpy as np
from typing import Tuple, Optional, Dict


class PointRefiner:
    """Sub-pixel circle center refinement from manual seed points."""

    def __init__(self, roi_size: int = 61, min_circle_area: int = 30,
                 max_circle_area: int = 5000):
        self.roi_size = roi_size          # ROI half-size in pixels
        self.min_circle_area = min_circle_area
        self.max_circle_area = max_circle_area

    def refine(self, image: np.ndarray, seed_x: float, seed_y: float,
               debug: bool = False
               ) -> Optional[Dict]:
        """
        Refine a seed point to sub-pixel center.

        Args:
            image: BGR or grayscale image.
            seed_x, seed_y: Manual click position (pixels).
            debug: If True, return debug overlay image.

        Returns:
            {
                'pixel_x': refined sub-pixel x,
                'pixel_y': refined sub-pixel y,
                'seed_x': original manual click x,
                'seed_y': original manual click y,
                'offset_px': distance between seed and refined,
                'confidence': 0-1 quality score,
                'ellipse_axes': (major, minor) of fitted ellipse,
                'method': 'ellipse_fit',
                'debug_img': ROI with overlay (if debug=True),
            }
            or None if refinement failed.
        """
        if image is None:
            return None

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # ROI bounds
        half = self.roi_size // 2
        x0 = max(0, int(seed_x) - half)
        y0 = max(0, int(seed_y) - half)
        x1 = min(w, int(seed_x) + half)
        y1 = min(h, int(seed_y) + half)

        if x1 - x0 < 10 or y1 - y0 < 10:
            return None

        roi = gray[y0:y1, x0:x1]

        # Adaptive threshold to find bright blob on dark background
        # Otsu or fixed: white circles on dark board
        _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If Otsu fails (too many/few white pixels), try fixed threshold
        white_frac = cv2.countNonZero(mask) / mask.size
        if white_frac < 0.02 or white_frac > 0.5:
            _, mask = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

        # Find connected components
        n_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=8)

        best_comp = None
        best_score = -1
        seed_local_x = seed_x - x0
        seed_local_y = seed_y - y0

        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_circle_area or area > self.max_circle_area:
                continue

            cx, cy = centroids[i]

            # Prefer component closest to seed point
            dist = math.hypot(cx - seed_local_x, cy - seed_local_y)

            # Get contour for circularity check
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = contours[0]
            if len(contour) < 5:
                continue

            area_c = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Circularity
            circularity = 4 * math.pi * area_c / (perimeter * perimeter)
            if circularity < 0.4:
                continue

            # Score: prefer closer to seed, more circular, larger area
            score = (1.0 / (1.0 + dist / 10.0)) * circularity * math.sqrt(area_c)
            if score > best_score:
                best_score = score
                best_comp = (i, cx, cy, area_c, contour, circularity)

        if best_comp is None:
            return None

        _, comp_cx, comp_cy, comp_area, contour, circularity = best_comp

        # Fit ellipse for sub-pixel center
        ellipse = cv2.fitEllipse(contour)
        refined_x = x0 + ellipse[0][0]
        refined_y = y0 + ellipse[0][1]
        axes = (ellipse[1][0], ellipse[1][1])

        # Confidence: combination of circularity and closeness to seed
        offset = math.hypot(refined_x - seed_x, refined_y - seed_y)
        confidence = circularity * min(1.0, 15.0 / max(offset, 0.1))

        result = {
            'pixel_x': float(refined_x),
            'pixel_y': float(refined_y),
            'seed_x': float(seed_x),
            'seed_y': float(seed_y),
            'offset_px': round(float(offset), 2),
            'confidence': round(float(confidence), 3),
            'ellipse_axes': (round(float(axes[0]), 1), round(float(axes[1]), 1)),
            'method': 'ellipse_fit',
        }

        if debug:
            dbg = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            # Draw seed as blue dot
            cv2.circle(dbg, (int(seed_local_x), int(seed_local_y)),
                       3, (255, 0, 0), -1)
            # Draw refined as green cross
            rx, ry = int(refined_x - x0), int(refined_y - y0)
            cv2.line(dbg, (rx-5, ry), (rx+5, ry), (0, 255, 0), 1)
            cv2.line(dbg, (rx, ry-5), (rx, ry+5), (0, 255, 0), 1)
            # Draw fitted ellipse
            ell = ((rx, ry), (int(axes[0]), int(axes[1])), ellipse[2])
            cv2.ellipse(dbg, ell, (0, 255, 0), 1)
            result['debug_img'] = dbg

        return result
