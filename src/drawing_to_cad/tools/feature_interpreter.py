
import re


class FeatureInterpreter:
    """Convert OCR annotations into structured, dimension-related candidates."""

    PATTERNS = {
        "diameter": (
            r"(?:Ø|⌀|DIA\.?)\s*(\d+(?:[.,]\d+)?)"
        ),
        "dimension_with_tolerance": (
            r"(?<!\w)(\d+(?:[.,]\d+)?)\s*"
            r"([+-]\s*\d+(?:[.,]\d+)?)\s+THRU\b"
        ),
        "multiplicity": r"\b(\d+)\s*[X×]\b",
        "bolt_circle": (
            r"(\d+(?:[.,]\d+)?)\s*(?:B\.?C\.?|PCD)\b"
        ),
        "through": r"\bTHRU\b",
        "depth": r"(\d+(?:[.,]\d+)?)\s*DEEP\b",
        "equal_spacing": r"\bEQ\s+SP\b",
    }

    @staticmethod
    def _number(value: str) -> float:
        """Convert a number using either decimal point or decimal comma."""
        return float(value.replace(",", ".").replace(" ", ""))

    def interpret(self, text_result: dict, geometry_result: dict) -> dict:
        annotations = []

        for item in text_result.get("items", []):
            text = item.get("text", "").strip()

            if not text:
                continue

            annotation = {
                "source_text": text,
                "confidence": item.get("confidence"),
                "center_px": item.get("center_px"),
                "properties": {},
            }

            for name, pattern in self.PATTERNS.items():
                match = re.search(pattern, text, re.IGNORECASE)

                if not match:
                    continue

                if name == "diameter":
                    annotation["properties"]["diameter"] = self._number(
                        match.group(1)
                    )

                elif name == "dimension_with_tolerance":
                    annotation["properties"]["nominal_size"] = self._number(
                        match.group(1)
                    )
                    annotation["properties"]["upper_deviation"] = self._number(
                        match.group(2)
                    )

                elif name == "multiplicity":
                    annotation["properties"]["multiplicity"] = int(
                        match.group(1)
                    )

                elif name == "bolt_circle":
                    annotation["properties"]["bolt_circle"] = self._number(
                        match.group(1)
                    )

                elif name == "depth":
                    annotation["properties"]["depth"] = self._number(
                        match.group(1)
                    )

                elif name in ("through", "equal_spacing"):
                    annotation["properties"][name] = True

            if annotation["properties"]:
                annotations.append(annotation)

        return {
            "status": "candidate_interpretation",
            "units": "unknown",
            "annotations": annotations,
            "geometry_summary": {
                "line_candidate_count": len(
                    geometry_result.get("line_candidates", [])
                ),
                "circle_candidate_count": len(
                    geometry_result.get("circle_candidates", [])
                ),
            },
            "limitations": [
                "Dimensions are not yet associated with specific geometric features.",
                "OCR annotations split across multiple text items are not yet merged.",
                "Detected circles may include drawing annotations and duplicate contours.",
                "Units and projection must be established from the drawing.",
            ],
        }
