
import re

import easyocr
import numpy as np


class TextExtractor:
    def __init__(self, languages: list[str] | None = None):
        self.reader = easyocr.Reader(languages or ["en"], gpu=False)

    def extract(self, image: np.ndarray) -> dict:
        results = self.reader.readtext(image)

        items = []
        for bbox, text, confidence in results:
            text = " ".join(text.split())
            if not text:
                continue

            points = [[int(p[0]), int(p[1])] for p in bbox]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            items.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": points,
                "center_px": [
                    round(sum(xs) / len(xs), 1),
                    round(sum(ys) / len(ys), 1),
                ],
            })

        full_text = "\n".join(item["text"] for item in items)

        return {
            "full_text": full_text,
            "items": items,
            "dimension_candidates": self._find_dimension_candidates(items),
        }

    @staticmethod
    def _find_dimension_candidates(items: list[dict]) -> list[dict]:
        # Match complete annotation strings rather than isolated numbers.
        patterns = [
            r"(?i)(?:Ø|⌀|DIA\.?)\s*\d+(?:[.,]\d+)?",
            r"(?i)\b\d+\s*[x×]\s*(?:Ø|⌀)?\s*\d+(?:[.,]\d+)?",
            r"[+-]?\d+(?:[.,]\d+)?\s*(?:\+\s*/\s*-|±)\s*\d+(?:[.,]\d+)?",
            r"[+-]?\d+(?:[.,]\d+)?\s*\+\s*\d+(?:[.,]\d+)?",
            r"(?i)\b\d+(?:[.,]\d+)?\s*(?:THRU|DEEP|PCD|B\.?C\.?)\b",
            r"(?i)\b\d+\s*X\b",
            r"(?i)\b\d+(?:[.,]\d+)?\s*(?:MM|IN)\b",
        ]

        candidates = []
        for item in items:
            text = item["text"]
            matches = []

            for pattern in patterns:
                matches.extend(
                    match.group(0)
                    for match in re.finditer(pattern, text)
                )

            # Keep likely standalone dimensions for later contextual analysis.
            if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
                matches.append(text)

            if matches:
                candidates.append({
                    "source_text": text,
                    "matches": list(dict.fromkeys(matches)),
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                    "center_px": item["center_px"],
                })

        return candidates
