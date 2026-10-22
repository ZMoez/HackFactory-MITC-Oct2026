
import cv2
import numpy as np


class ViewDetector:
    """Detect candidate drawing-view regions using image contours."""

    def detect(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Join nearby strokes into candidate drawing regions.
        binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        kernel_size = max(3, int(min(height, width) * 0.008))
        kernel_size += 1 - kernel_size % 2
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        min_area = height * width * 0.002

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < min_area:
                continue

            # Exclude regions occupying almost the entire sheet.
            if w > width * 0.95 and h > height * 0.95:
                continue

            regions.append({
                "bbox_px": [int(x), int(y), int(w), int(h)],
                "area_px": int(area),
                "aspect_ratio": round(w / h, 3) if h else None,
            })

        regions.sort(key=lambda region: region["area_px"], reverse=True)

        return {
            "image_size_px": {"width": int(width), "height": int(height)},
            "candidate_views": regions,
            "note": (
                "These are candidate regions, not verified front, top, "
                "or side views."
            ),
        }
