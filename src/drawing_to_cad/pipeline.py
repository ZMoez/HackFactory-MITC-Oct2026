
import json
from pathlib import Path

import cv2

from src.drawing_to_cad.tools.input_loader import load_drawing
from src.drawing_to_cad.tools.text_extractor import TextExtractor
from src.drawing_to_cad.tools.geometry_detector import GeometryDetector
from src.drawing_to_cad.tools.feature_interpreter import FeatureInterpreter


class DrawingToCADPipeline:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.geometry_detector = GeometryDetector()
        self.feature_interpreter = FeatureInterpreter()

    def run(self, drawing_path: str | Path) -> list[Path]:
        drawing_path = Path(drawing_path).expanduser().resolve()
        images = load_drawing(drawing_path)

        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "result" / drawing_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = []

        for page_number, image in enumerate(images, start=1):
            page_dir = output_dir
            if len(images) > 1:
                page_dir = output_dir / f"page_{page_number:03d}"
                page_dir.mkdir(parents=True, exist_ok=True)

            text_result = self.text_extractor.extract(image)
            geometry_result = self.geometry_detector.detect(image)
            feature_result = self.feature_interpreter.interpret(text_result, geometry_result)


            # Save edge image separately; don't clutter the drawing with overlays.
            cv2.imwrite(str(page_dir / "edges.png"), geometry_result.pop("edges"))

            dimensions_path = page_dir / "detected_dimensions.json"
            geometry_path = page_dir / "detected_geometry.json"
            
            features_path = page_dir / "interpreted_features.json"
            features_path.write_text(
                 json.dumps(feature_result, indent=2, ensure_ascii=False),
                 encoding="utf-8",
            )
            outputs.append(features_path)


            dimensions_path.write_text(
                json.dumps(text_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            geometry_path.write_text(
                 json.dumps(geometry_result, indent=2, ensure_ascii=False),
                 encoding="utf-8",
            )

            outputs.extend([dimensions_path, geometry_path])

            print(f"\nPage {page_number}/{len(images)}")
            print(f"OCR items: {len(text_result['items'])}")
            print(
                "Dimension candidate groups: "
                f"{len(text_result['dimension_candidates'])}"
            )
            print(f"Line candidates: {len(geometry_result['line_candidates'])}")
            print(f"Circle candidates: {len(geometry_result['circle_candidates'])}")
            print(f"Saved to: {page_dir}")

        return outputs
