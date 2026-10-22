
import cv2
import numpy as np


class GeometryDetector:
    def detect(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        height, width = gray.shape
        min_line_length = max(30, int(min(width, height) * 0.02))

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=min_line_length,
            maxLineGap=8,
        )

        lines = []
        if raw_lines is not None:
            for segment in raw_lines.reshape(-1, 4):
                x1, y1, x2, y2 = map(int, segment)
                length = float(np.hypot(x2 - x1, y2 - y1))
                lines.append({
                    "start": [x1, y1],
                    "end": [x2, y2],
                    "length_px": round(length, 2),
                })

        # Find circular contours and filter tiny/noisy candidates.
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = []
        min_radius = max(5, int(min(width, height) * 0.003))
        max_radius = int(min(width, height) * 0.25)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)

            if perimeter <= 0 or area <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)

            if not (min_radius <= radius <= max_radius):
                continue
            if circularity < 0.65:
                continue

            circles.append({
                "center_px": [round(float(x), 1), round(float(y), 1)],
                "radius_px": round(radius, 1),
                "circularity": round(float(circularity), 3),
            })

        # Remove near-duplicate detections.
        circles.sort(key=lambda c: c["circularity"], reverse=True)
        filtered = []

        for circle in circles:
            x, y = circle["center_px"]
            r = circle["radius_px"]

            duplicate = any(
                np.hypot(x - fx, y - fy) < max(5, min(r, fr) * 0.25)
                and abs(r - fr) < max(5, min(r, fr) * 0.25)
                for fx, fy, fr in [
                    (c["center_px"][0], c["center_px"][1], c["radius_px"])
                    for c in filtered
                ]
            )

            if not duplicate:
                filtered.append(circle)

            if len(filtered) >= 300:
                break

        return {
            "image_size_px": {"width": int(width), "height": int(height)},
            "line_candidates": lines,
            "circle_candidates": filtered,
            "edges": edges,
            "note": (
                "Candidates are approximate image detections, not confirmed "
                "CAD features. Dimension and view association is still required."
            ),
        }
