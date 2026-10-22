# Create three directories in result/ for train, validation, and test splits. For each part_id in the split's CSV file, run the pipeline on the corresponding drawing and save the output to the appropriate directory. 
# Code creagted by: ChatGPT, checked by Z  
import csv
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = PROJECT_ROOT / "data" / "cadsy" / "splits"
SAMPLES_DIR = PROJECT_ROOT / "data" / "cadsy" / "samples"
RESULT_DIR = PROJECT_ROOT / "result"

PIPELINE_OUTPUT_DIR = RESULT_DIR / "asme"
FEATURES_DIR = RESULT_DIR / "extracted_features"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"


def run_split(split_name: str) -> None:
    csv_path = SPLIT_DIR / f"{split_name}.csv"

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for index, row in enumerate(rows, start=1):
        part_id = row["part_id"]
        drawing_path = SAMPLES_DIR / part_id / "asme.png"
        destination = FEATURES_DIR / split_name / part_id

        if not drawing_path.is_file():
            raise FileNotFoundError(
                f"Drawing not found: {drawing_path}"
            )

        # Remove the previous run's output to avoid stale files.
        if PIPELINE_OUTPUT_DIR.exists():
            shutil.rmtree(PIPELINE_OUTPUT_DIR)

        print(
            f"\n[{split_name} {index}/{len(rows)}] "
            f"Processing {part_id}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(MAIN_SCRIPT),
                str(drawing_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        if not PIPELINE_OUTPUT_DIR.is_dir():
            raise FileNotFoundError(
                "Pipeline did not create result/asme/"
            )

        required_files = [
            "detected_dimensions.json",
            "detected_geometry.json",
            "interpreted_features.json",
        ]

        missing = [
            name
            for name in required_files
            if not (PIPELINE_OUTPUT_DIR / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing pipeline outputs for {part_id}: {missing}"
            )

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            PIPELINE_OUTPUT_DIR,
            destination,
            dirs_exist_ok=True,
        )

        print(f"Saved features to: {destination}", flush=True)


def main() -> None:
    if not MAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Pipeline entry point not found: {MAIN_SCRIPT}")

    for split_name in ("train", "validation", "test"):
        run_split(split_name)

    print("\nAll pipeline outputs collected successfully.")


if __name__ == "__main__":
    main()
