
import cv2
import numpy as np
import ezdxf
import cadquery as cq

from pathlib import Path

# Replace this with the path to your ORIGINAL drawing image

# INPUT = Path("YOUR_ORIGINAL_DRAWING.png")
INPUT = Path("data/cadsy/samples/part_0043/asme.png")
OUT = Path("result/drawing_to_3d")
OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Drawing not found: {INPUT}\n"
        "Set INPUT to the path of your original PNG/JPG image."
    )

# 1. Read and preprocess the drawing
image = cv2.imread(str(INPUT), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Could not read the drawing image.")

blur = cv2.GaussianBlur(image, (3, 3), 0)
binary = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 8
)

cv2.imwrite(str(OUT / "01_binary.png"), binary)

# 2. Find contours as 2D geometry candidates
contours, _ = cv2.findContours(
    binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
)

# Ignore tiny contours, which are often noise or text fragments
height, width = image.shape
min_area = max(20, (height * width) * 0.000005)

doc = ezdxf.new("R2010")
msp = doc.modelspace()

closed_profiles = []
preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
        continue

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour, max(1.0, 0.002 * perimeter), True
    )

    points_px = approx.reshape(-1, 2)
    if len(points_px) < 3:
        continue

    # Convert image coordinates to drawing coordinates.
    # These are PIXELS, not millimetres.
    points = [(float(x), float(height - y)) for x, y in points_px]

    msp.add_lwpolyline(points, close=True)
    cv2.drawContours(preview, [approx], -1, (0, 180, 0), 1)

    # Keep only plausible larger closed profiles for 3D testing
    x, y, w, h = cv2.boundingRect(contour)
    if area > image.shape[0] * image.shape[1] * 0.002:
        closed_profiles.append((area, points_px, (x, y, w, h)))

doc.saveas(str(OUT / "02_candidates.dxf"))
cv2.imwrite(str(OUT / "03_contours_overlay.png"), preview)

print(f"Image size: {width} x {height} pixels")
print(f"Contours found: {len(contours)}")
print(f"Large profile candidates: {len(closed_profiles)}")
print(f"DXF saved: {OUT / '02_candidates.dxf'}")

# 3. Attempt a cautious 3D demonstration using the largest contour.
# This is only a geometric experiment: the largest contour might be
# a drawing frame, view outline, or annotation—not the actual part.
if not closed_profiles:
    print("No large closed profile found; STEP was not generated.")
    raise SystemExit(0)

closed_profiles.sort(key=lambda item: item[0], reverse=True)
_, profile_px, bbox = closed_profiles[0]
x, y, w, h = bbox

# Shift the candidate profile to the origin.
# Pixel units are NOT physical dimensions.
points = [
    (float(px - x), float((y + h) - py))
    for px, py in profile_px
]

try:
    solid = (
        cq.Workplane("XY")
        .polyline(points)
        .close()
        .extrude(10.0)  # arbitrary 10-pixel-depth equivalent
    )

    cq.exporters.export(solid, str(OUT / "04_experimental_model.step"))
    cq.exporters.export(solid, str(OUT / "04_experimental_model.stl"))
    print("Experimental STEP/STL exported.")
    print("WARNING: not dimensionally accurate and may not represent the part.")
except Exception as exc:
    print("Could not create a solid from the selected contour.")
    print(f"Reason: {exc}")
    print("Inspect the overlay and DXF before proceeding.")
