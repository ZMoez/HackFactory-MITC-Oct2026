
import csv
import random
from pathlib import Path

# Input dataset
source = Path("data/cadsy/dataset_pairs.csv")

# Output directory
output_dir = Path("data/cadsy/splits")
output_dir.mkdir(parents=True, exist_ok=True)

# Read dataset pairs
with source.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 65:
    raise ValueError(
        f"Expected 65 samples, but found {len(rows)}"
    )

# Shuffle reproducibly
random.Random(42).shuffle(rows)

# Split into train, validation, and test
splits = {
    "train.csv": rows[:45],
    "validation.csv": rows[45:55],
    "test.csv": rows[55:65],
}

# Save each split
for filename, samples in splits.items():
    output_path = output_dir / filename

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["part_id", "drawing", "model"],
        )
        writer.writeheader()
        writer.writerows(samples)

    print(f"Created {output_path}: {len(samples)} samples")

print("\nDataset split completed successfully.")
