
from pathlib import Path
import argparse

import cv2
import ezdxf
import numpy as np


def convert_edges_to_dxf(
    input_path,
    output_path,
    scale=1.0,
    threshold=127,
    min_area=3.0,
    epsilon=1.5,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read edge image as grayscale
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    # Convert bright edges to a binary mask
    _, binary = cv2.threshold(
        image, threshold, 255, cv2.THRESH_BINARY
    )

    # Find contours in the edge image
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.new(
        name="EDGES",
        dxfattribs={"color": 7},
    )

    height, width = image.shape
    saved = 0

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Simplify contours to reduce excessive vertices
        approx = cv2.approxPolyDP(
            contour, epsilon, True
        )

        points = approx.reshape(-1, 2)

        if len(points) < 2:
            continue

        # Convert image coordinates to CAD coordinates.
        # Flip Y so the drawing is upright in CAD.
        cad_points = [
            (float(x) * scale, float(height - y) * scale)
            for x, y in points
        ]

        closed = cv2.arcLength(contour, True) > 0 and (
            np.linalg.norm(points[0] - points[-1]) <= 1
        )

        # OpenCV contours are represented as closed outlines.
        # Save each contour as a closed polyline.
        msp.add_lwpolyline(
            cad_points,
            close=True,
            dxfattribs={"layer": "EDGES"},
        )
        saved += 1

    # Store image dimensions as drawing metadata
    doc.header["$INSUNITS"] = 0  # Unitless; pixel scale for now
    doc.saveas(output_path)

    print(f"Input image: {input_path}")
    print(f"Image size: {width} x {height} pixels")
    print(f"Contours found: {len(contours)}")
    print(f"Polylines saved: {saved}")
    print(f"DXF saved: {output_path}")
    print("Coordinate scale:", scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an edge PNG image to a DXF file."
    )
    parser.add_argument(
        "--input",
        default="result/asme/edges.png",
    )
    parser.add_argument(
        "--output",
        default="result/asme/edges.dxf",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="CAD units per pixel; default is 1.",
    )
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--min-area", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=1.5)

    args = parser.parse_args()

    convert_edges_to_dxf(
        args.input,
        args.output,
        scale=args.scale,
        threshold=args.threshold,
        min_area=args.min_area,
        epsilon=args.epsilon,
    )
