
import json
from pathlib import Path

import numpy as np
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "cadsy" / "samples"
RESULT_DIR = PROJECT_ROOT / "result"

IMAGE_SIZE = 256
TARGET_KEYS = ["bbox_x", "bbox_y", "bbox_z"]


def process_split(split_name: str) -> None:
    """Prepare images and dimension labels for one existing split."""
    targets_path = (
        RESULT_DIR / split_name / f"{split_name}_targets.json"
    )

    if not targets_path.is_file():
        raise FileNotFoundError(
            f"Targets file not found: {targets_path}. "
            "Run prepare_targets.py first."
        )

    records = json.loads(targets_path.read_text(encoding="utf-8"))

    images = []
    labels = []
    part_ids = []

    for record in records:
        part_id = record["part_id"]
        image_path = DATA_DIR / part_id / "asme.png"

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Drawing not found for {part_id}: {image_path}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        image = image.astype(np.float32) / 255.0

        target = record["targets"]
        label = [float(target[key]) for key in TARGET_KEYS]

        images.append(image)
        labels.append(label)
        part_ids.append(part_id)

    output_dir = RESULT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{split_name}_image_dataset.npz"

    np.savez_compressed(
        output_path,
        images=np.asarray(images, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        part_ids=np.asarray(part_ids),
    )

    print(f"[{split_name}] Images: {len(images)}")
    print(f"[{split_name}] Image shape: {np.asarray(images).shape}")
    print(f"[{split_name}] Labels shape: {np.asarray(labels).shape}")
    print(f"[{split_name}] Saved to: {output_path}")


def main() -> None:
    for split_name in ("train", "validation", "test"):
        process_split(split_name)

    print("All image datasets prepared successfully.")


if __name__ == "__main__":
    main()
