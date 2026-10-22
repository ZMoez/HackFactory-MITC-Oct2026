
from pathlib import Path
import argparse
import json

import cv2
import ezdxf
import numpy as np


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def find_ocr_boxes(image):
    """Return text boxes if pytesseract and Tesseract are available."""
    try:
        import pytesseract

        data = pytesseract.image_to_data(
            image,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
        boxes = []
        texts = []

        for i, raw in enumerate(data["text"]):
            text = raw.strip()
            if not text:
                continue

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = -1

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            boxes.append((x, y, w, h))
            texts.append({
                "text": text,
                "confidence": confidence,
                "box": [x, y, w, h],
            })

        return boxes, texts, None

    except Exception as exc:
        return [], [], str(exc)


def remove_text_regions(binary, boxes, padding=3):
    """Mask OCR text boxes; this can accidentally mask nearby geometry."""
    cleaned = binary.copy()
    h, w = cleaned.shape

    for x, y, bw, bh in boxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        cleaned[y1:y2, x1:x2] = 0

    return cleaned


def remove_long_dimension_lines(binary, length=100):
    """
    Optional heuristic for long horizontal/vertical lines.
    Use cautiously: real part edges can also be long.
    """
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (length, 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, length)
    )

    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, vertical_kernel
    )

    lines = cv2.bitwise_or(horizontal, vertical)
    cleaned = cv2.subtract(binary, lines)
    return cleaned, lines


def contours_to_dxf(binary, output_path, epsilon=1.5, min_area=8):
    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = binary.shape
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 0  # Unitless: pixel coordinates

    doc.layers.new(
        name="CONTOURS",
        dxfattribs={"color": 7},
    )

    msp = doc.modelspace()
    saved = 0

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < min_area:
            continue

        approx = cv2.approxPolyDP(
            contour, epsilon, True
        )
        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue

        # Convert image coordinates to CAD coordinates.
        # Image y increases downward; CAD y increases upward.
        cad_points = [
            (float(x), float(h - y))
            for x, y in points
        ]

        msp.add_lwpolyline(
            cad_points,
            close=True,
            dxfattribs={"layer": "CONTOURS"},
        )
        saved += 1

    doc.saveas(output_path)

    return {
        "detected_contours": len(contours),
        "saved_polylines": saved,
        "image_width_px": w,
        "image_height_px": h,
        "units": "unitless pixels",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="result/extracted_features/test/part_0044/edges.png",
    )
    parser.add_argument(
        "--drawing",
        default=None,
        help="Optional original drawing image, if available.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/extracted_features/test/part_0044/cad_output",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=0.03,
        help="Fraction cropped from each image edge; default 0.03.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
    )
    parser.add_argument(
        "--remove-text",
        action="store_true",
        help="Remove OCR text boxes from the edge mask.",
    )
    parser.add_argument(
        "--remove-long-lines",
        action="store_true",
        help="Experimental: remove long horizontal/vertical lines.",
    )
    parser.add_argument("--line-length", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--min-area", type=float, default=8.0)
    args = parser.parse_args()

    source = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    edge_image = read_gray(source)

    # OCR should preferably use the original drawing, not edge-only data.
    ocr_image = (
        read_gray(args.drawing)
        if args.drawing
        else edge_image
    )

    if ocr_image.shape != edge_image.shape:
        raise ValueError(
            "The original drawing and edges image must have "
            "matching dimensions for OCR masking."
        )

    h, w = edge_image.shape
    crop = max(0.0, min(args.crop, 0.20))
    x0, x1 = int(w * crop), int(w * (1 - crop))
    y0, y1 = int(h * crop), int(h * (1 - crop))

    cropped = edge_image[y0:y1, x0:x1]
    cv2.imwrite(str(outdir / "01_cropped_edges.png"), cropped)

    _, binary = cv2.threshold(
        cropped, args.threshold, 255, cv2.THRESH_BINARY
    )

    ocr_boxes, ocr_text, ocr_error = find_ocr_boxes(ocr_image)

    # Shift OCR coordinates into cropped-image coordinates.
    cropped_boxes = []
    for x, y, bw, bh in ocr_boxes:
        x -= x0
        y -= y0

        if x + bw <= 0 or y + bh <= 0:
            continue
        if x >= binary.shape[1] or y >= binary.shape[0]:
            continue

        cropped_boxes.append((x, y, bw, bh))

    processed = binary.copy()

    if args.remove_text:
        processed = remove_text_regions(processed, cropped_boxes)

    removed_lines = np.zeros_like(processed)
    if args.remove_long_lines:
        processed, removed_lines = remove_long_dimension_lines(
            processed, args.line_length
        )
        cv2.imwrite(
            str(outdir / "02_removed_long_lines.png"),
            removed_lines,
        )

    cv2.imwrite(str(outdir / "03_processed_mask.png"), processed)

    # Draw OCR boxes for inspection, not as CAD geometry.
    preview = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    for x, y, bw, bh in cropped_boxes:
        cv2.rectangle(
            preview, (x, y), (x + bw, y + bh),
            (0, 0, 255), 2
        )
    cv2.imwrite(str(outdir / "04_ocr_preview.png"), preview)

    dxf_path = outdir / "part_0044.dxf"
    geometry_report = contours_to_dxf(
        processed,
        dxf_path,
        epsilon=args.epsilon,
        min_area=args.min_area,
    )

    # a machine readable 3D mode
    # save a 2D preview of the contours for visual inspection in STEP 
    #  



    # This is a 2D edge/contour visualization, not a solid model.
    contours, _ = cv2.findContours(
        processed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    model_preview = np.full(processed.shape, 255, dtype=np.uint8)
    cv2.drawContours(model_preview, contours, -1, 0, 1)
    cv2.imwrite(
        str(outdir / "05_2d_model_preview.png"),
        model_preview,
    )

    report = {
        "input": str(source),
        "original_drawing": args.drawing,
        "crop_fraction_each_edge": crop,
        "ocr_text": ocr_text,
        "ocr_error": ocr_error,
        "text_boxes_masked": len(cropped_boxes) if args.remove_text else 0,
        "long_line_removal_enabled": args.remove_long_lines,
        "geometry": geometry_report,
        "limitations": [
            "OCR detections are not guaranteed to be correct.",
            "Text boxes may overlap real part geometry.",
            "Long dimension lines may resemble real part edges.",
            "Dimensions are not yet associated with specific features.",
            "DXF coordinates are pixels, not calibrated millimetres.",
            "This is a 2D contour model, not a 3D printable solid.",
        ],
    }

    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(outdir / "ocr_report.txt", "w") as f:
        for item in ocr_text:
            f.write(
                f"{item['text']}\tconfidence={item['confidence']}"
                f"\tbox={item['box']}\n"
            )
        if ocr_error:
            f.write(f"\nOCR unavailable: {ocr_error}\n")

    print("\nPipeline finished.")
    print("Output folder:", outdir)
    print("DXF:", dxf_path)
    print("2D preview:", outdir / "05_2d_model_preview.png")
    print("OCR report:", outdir / "ocr_report.txt")
    print("Report:", outdir / "report.json")
    print("Geometry:", geometry_report)


if __name__ == "__main__":
    main()
