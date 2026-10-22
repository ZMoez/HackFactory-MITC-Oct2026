
import argparse
from pathlib import Path

from src.drawing_to_cad.pipeline import DrawingToCADPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and geometric candidates from engineering drawings."
    )
    parser.add_argument(
        "drawing",
        type=Path,
        help="Path to a PNG, JPG, JPEG, or PDF engineering drawing",
    )
    args = parser.parse_args()

    pipeline = DrawingToCADPipeline()
    pipeline.run(args.drawing)


if __name__ == "__main__":
    main()
