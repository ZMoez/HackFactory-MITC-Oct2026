
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "result"
MODEL_PATH = RESULT_DIR / "models" / "dimension_baseline.pt"
TEST_PATH = RESULT_DIR / "test" / "test_image_dataset.npz"
OUTPUT_PATH = RESULT_DIR / "test" / "baseline_predictions.csv"

DEVICE = "cpu"
TARGET_NAMES = ["bbox_x", "bbox_y", "bbox_z"]


class DimensionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    # Load the test set separately. Do not use it for training.
    with np.load(TEST_PATH) as data:
        images = data["images"].copy()
        true_labels = data["labels"].copy()
        part_ids = data["part_ids"].tolist()

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )

    model = DimensionRegressor().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = np.asarray(checkpoint["target_mean"])
    target_std = np.asarray(checkpoint["target_std"])

    x_test = torch.tensor(
        images[:, None], dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        normalized_predictions = model(x_test).cpu().numpy()

    predictions = normalized_predictions * target_std + target_mean
    errors = predictions - true_labels
    absolute_errors = np.abs(errors)

    print(f"Test samples: {len(part_ids)}")
    print("\nMean absolute error (mm):")

    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name}: {absolute_errors[:, i].mean():.3f} mm")

    print(
        f"\nOverall MAE across all dimensions: "
        f"{absolute_errors.mean():.3f} mm"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "part_id",
            "true_bbox_x", "pred_bbox_x", "abs_error_x",
            "true_bbox_y", "pred_bbox_y", "abs_error_y",
            "true_bbox_z", "pred_bbox_z", "abs_error_z",
        ])

        for i, part_id in enumerate(part_ids):
            row = [part_id]

            for j in range(3):
                row.extend([
                    round(float(true_labels[i, j]), 3),
                    round(float(predictions[i, j]), 3),
                    round(float(absolute_errors[i, j]), 3),
                ])

            writer.writerow(row)

    print(f"\nPer-part results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
