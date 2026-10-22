
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "result"
SPLIT_DIR = PROJECT_ROOT / "data" / "cadsy" / "splits"
SAMPLES_DIR = PROJECT_ROOT / "data" / "cadsy" / "samples"
FEATURES_DIR = RESULT_DIR / "extracted_features"

IMAGE_SIZE = 256
TARGET_KEYS = ["bbox_x", "bbox_y", "bbox_z"]

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def numeric_tokens(text: str) -> list[float]:
    values = []
    for token in NUMBER_PATTERN.findall(text):
        try:
            values.append(float(token.replace(",", ".")))
        except ValueError:
            continue
    return values


def add_stats(result: dict, prefix: str, values) -> None:
    """Add fixed-length summary statistics, including for empty lists."""
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]

    result[f"{prefix}_count"] = float(len(values))
    result[f"{prefix}_mean"] = float(values.mean()) if len(values) else 0.0
    result[f"{prefix}_std"] = float(values.std()) if len(values) else 0.0
    result[f"{prefix}_min"] = float(values.min()) if len(values) else 0.0
    result[f"{prefix}_median"] = float(np.median(values)) if len(values) else 0.0
    result[f"{prefix}_max"] = float(values.max()) if len(values) else 0.0


def extract_features(part_dir: Path) -> dict:
    dimensions = read_json(part_dir / "detected_dimensions.json")
    geometry = read_json(part_dir / "detected_geometry.json")
    interpreted = read_json(part_dir / "interpreted_features.json")

    result = {}

    # Image dimensions and scale-independent geometric summaries.
    image_size = geometry.get("image_size_px", {})
    width = float(image_size.get("width", 0))
    height = float(image_size.get("height", 0))
    diagonal = max(float(np.hypot(width, height)), 1.0)

    result["image_width_px"] = width
    result["image_height_px"] = height
    result["image_aspect_ratio"] = width / height if height else 0.0

    # OCR dimension candidates.
    candidates = dimensions.get("dimension_candidates", [])
    candidate_texts = [
        str(item.get("source_text", item.get("text", "")))
        for item in candidates
    ]
    candidate_confidences = [
        float(item.get("confidence", 0.0))
        for item in candidates
    ]

    all_numbers = [
        number
        for text in candidate_texts
        for number in numeric_tokens(text)
    ]

    add_stats(result, "dimension_number", all_numbers)
    add_stats(result, "dimension_ocr_confidence", candidate_confidences)

    result["ocr_item_count"] = float(len(dimensions.get("items", [])))
    result["dimension_candidate_count"] = float(len(candidates))
    result["dimension_confidence_ge_08_count"] = float(
        sum(value >= 0.8 for value in candidate_confidences)
    )
    result["dimension_confidence_ge_095_count"] = float(
        sum(value >= 0.95 for value in candidate_confidences)
    )

    # Simple text cues from the dimension annotations.
    combined_text = " ".join(candidate_texts).upper()

    text_patterns = {
        "text_thru_count": r"\bTHRU\b|\bTHROUGH\b",
        "text_depth_count": r"\bDEEP\b|\bDEPTH\b",
        "text_radius_count": r"\bR\s*\d",
        "text_chamfer_count": r"\bC\s*[0-9]",
        "text_multiplicity_count": r"\b\d+\s*X\b",
        "text_diameter_count": r"Ø|⌀|ø|Φ|DIAM",
        "text_bolt_circle_count": r"\b(?:BCD|PCD)\b|BOLT\s*CIRCLE",
        "text_equal_spacing_count": r"EQ\s*SP|EQUALLY\s*SPACED",
        "text_tolerance_count": r"\+\s*0|±|\+/-",
    }

    for feature_name, pattern in text_patterns.items():
        result[feature_name] = float(
            len(re.findall(pattern, combined_text))
        )

    # Interpreted semantic annotations.
    annotations = interpreted.get("annotations", [])
    result["annotation_count"] = float(len(annotations))

    multiplicities = []
    depths = []
    through_count = 0
    diameter_count = 0
    radius_count = 0

    for annotation in annotations:
        properties = annotation.get("properties", {})
        text = str(annotation.get("source_text", "")).upper()

        multiplicity = properties.get("multiplicity")
        if isinstance(multiplicity, (int, float)):
            multiplicities.append(float(multiplicity))

        depth = properties.get("depth")
        if isinstance(depth, (int, float)):
            depths.append(float(depth))

        if properties.get("through") is True:
            through_count += 1

        if (
            "diameter" in properties
            or re.search(r"Ø|⌀|ø|Φ|DIAM", text)
        ):
            diameter_count += 1

        if re.search(r"\bR\s*\d", text):
            radius_count += 1

    add_stats(result, "annotation_multiplicity", multiplicities)
    add_stats(result, "annotation_depth", depths)

    result["annotation_through_count"] = float(through_count)
    result["annotation_diameter_count"] = float(diameter_count)
    result["annotation_radius_count"] = float(radius_count)

    # Geometric candidates: summarize variable-length lists.
    lines = geometry.get("line_candidates", [])
    line_lengths = [
        float(line.get("length_px", 0.0)) / diagonal
        for line in lines
    ]

    circles = geometry.get("circle_candidates", [])
    circle_radii = [
        float(circle.get("radius_px", 0.0)) / diagonal
        for circle in circles
    ]
    circularities = [
        float(circle.get("circularity", 0.0))
        for circle in circles
    ]

    result["line_candidate_count"] = float(len(lines))
    add_stats(result, "line_length_ratio", line_lengths)

    result["line_ratio_ge_005_count"] = float(
        sum(value >= 0.05 for value in line_lengths)
    )
    result["line_ratio_ge_010_count"] = float(
        sum(value >= 0.10 for value in line_lengths)
    )
    result["line_ratio_ge_020_count"] = float(
        sum(value >= 0.20 for value in line_lengths)
    )

    result["circle_candidate_count"] = float(len(circles))
    add_stats(result, "circle_radius_ratio", circle_radii)
    add_stats(result, "circle_circularity", circularities)

    result["circle_circularity_ge_08_count"] = float(
        sum(value >= 0.8 for value in circularities)
    )

    # Keep the same ordered feature columns for every drawing.
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f"Non-finite feature value in {part_dir}")

    return result


def process_split(split_name: str) -> None:
    csv_path = SPLIT_DIR / f"{split_name}.csv"
    targets_path = RESULT_DIR / split_name / f"{split_name}_targets.json"

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    target_records = read_json(targets_path)
    targets_by_id = {
        record["part_id"]: record["targets"]
        for record in target_records
    }

    images = []
    features = []
    labels = []
    part_ids = []

    for row in rows:
        part_id = row["part_id"]
        part_feature_dir = FEATURES_DIR / split_name / part_id
        image_path = SAMPLES_DIR / part_id / "asme.png"

        if part_id not in targets_by_id:
            raise KeyError(f"No STEP target found for {part_id}")

        if not part_feature_dir.is_dir():
            raise FileNotFoundError(
                f"Extracted features not found: {part_feature_dir}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read drawing: {image_path}")

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        image = image.astype(np.float32) / 255.0

        feature_dict = extract_features(part_feature_dir)
        target = targets_by_id[part_id]

        images.append(image)
        features.append(feature_dict)
        labels.append([float(target[key]) for key in TARGET_KEYS])
        part_ids.append(part_id)

        print(f"[{split_name}] Prepared {part_id}")

    feature_names = list(features[0].keys())
    feature_matrix = np.asarray(
        [[item[name] for name in feature_names] for item in features],
        dtype=np.float32,
    )

    label_matrix = np.asarray(labels, dtype=np.float32)
    image_matrix = np.asarray(images, dtype=np.float32)

    output_dir = RESULT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}_feature_dataset.npz"

    np.savez_compressed(
        output_path,
        images=image_matrix,
        features=feature_matrix,
        labels=label_matrix,
        part_ids=np.asarray(part_ids),
        feature_names=np.asarray(feature_names),
    )

    print(f"\n[{split_name}] Samples: {len(part_ids)}")
    print(f"[{split_name}] Images: {image_matrix.shape}")
    print(f"[{split_name}] Features: {feature_matrix.shape}")
    print(f"[{split_name}] Labels: {label_matrix.shape}")
    print(f"[{split_name}] Saved: {output_path}\n")


def main() -> None:
    for split_name in ("train", "validation", "test"):
        process_split(split_name)

    print("All multimodal feature datasets prepared successfully.")


if __name__ == "__main__":
    main()
