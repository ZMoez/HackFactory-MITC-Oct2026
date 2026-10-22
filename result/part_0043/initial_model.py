
from pathlib import Path
import json
import cadquery as cq

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "cadsy" / "samples" / "part_0043"
OUT = ROOT / "result" / "part_0043"
OUT.mkdir(parents=True, exist_ok=True)

# Read the reference STEP for validation only.
step_path = SAMPLE / "model.step"
if step_path.exists():
    reference = cq.importers.importStep(str(step_path))
    bb = reference.val().BoundingBox()
    reference_info = {
        "x_mm": round(bb.xlen, 3),
        "y_mm": round(bb.ylen, 3),
        "z_mm": round(bb.zlen, 3),
        "solids": len(reference.solids().vals()),
        "volume_mm3": round(reference.val().Volume(), 3),
    }
    print("Reference STEP properties:")
    print(json.dumps(reference_info, indent=2))
    (OUT / "reference_properties.json").write_text(
        json.dumps(reference_info, indent=2)
    )
else:
    print("Reference STEP not found:", step_path)

# Initial independent approximation.
# Dimensions are provisional and must be refined from the drawing.
outer_diameter = 90.0
height = 30.0
center_hole_diameter = 40.0

model = (
    cq.Workplane("XY")
    .circle(outer_diameter / 2)
    .extrude(height)
    .faces(">Z")
    .workplane()
    .hole(center_hole_diameter)
)

cq.exporters.export(model, str(OUT / "initial_model.step"))
(OUT / "initial_model.py").write_text(
    Path(__file__).read_text()
)

bb = model.val().BoundingBox()
summary = {
    "status": "initial approximation",
    "x_mm": round(bb.xlen, 3),
    "y_mm": round(bb.ylen, 3),
    "z_mm": round(bb.zlen, 3),
    "volume_mm3": round(model.val().Volume(), 3),
    "note": "Central hole diameter is provisional; inspect drawing before finalizing.",
}
(OUT / "initial_model_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\nInitial model exported to:", OUT / "initial_model.step")
print(json.dumps(summary, indent=2))
