
import csv
import json
from pathlib import Path

import cadquery as cq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = PROJECT_ROOT / "data" / "cadsy" / "splits"
RESULT_DIR = PROJECT_ROOT / "result"


def extract_targets(model_path: Path) -> dict:
    """Extract geometric targets from a STEP model."""
    shape = cq.importers.importStep(str(model_path))
    solids = shape.solids().vals()

    if not solids:
        raise ValueError(f"No solids found in STEP model: {model_path}")

    # Bounding box and measurements of the complete imported shape.
    solid_shape = shape.val()
    bbox = solid_shape.BoundingBox()

    return {
        "bbox_x": round(bbox.xlen, 4),
        "bbox_y": round(bbox.ylen, 4),
        "bbox_z": round(bbox.zlen, 4),
        "solid_count": len(solids),
        "volume": round(solid_shape.Volume(), 4),
        "surface_area": round(solid_shape.Area(), 4),
    }


def process_split(split_name: str) -> None:
    """Process one existing dataset split and save its targets separately."""
    csv_path = SPLIT_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    required_columns = {"part_id", "drawing", "model"}
    if not rows or not required_columns.issubset(rows[0].keys()):
        raise ValueError(
            f"{csv_path} must contain columns: "
            f"{', '.join(sorted(required_columns))}"
        )

    results = []

    for row in rows:
        part_id = row["part_id"]
        model_path = PROJECT_ROOT / row["model"]

        if not model_path.is_file():
            raise FileNotFoundError(
                f"STEP model for {part_id} not found: {model_path}"
            )

        targets = extract_targets(model_path)

        results.append({
            "part_id": part_id,
            "drawing": row["drawing"],
            "model": row["model"],
            "targets": targets,
        })

        print(f"[{split_name}] Processed {part_id}")

    output_dir = RESULT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{split_name}_targets.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[{split_name}] Saved {len(results)} targets to {output_path}")


def main() -> None:
    # Use the existing CSV splits; never create a new split here.
    for split_name in ("train", "validation", "test"):
        process_split(split_name)

    print("All target files generated successfully.")


if __name__ == "__main__":
    main()
