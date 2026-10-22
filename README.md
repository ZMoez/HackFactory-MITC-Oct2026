# 2D Engineering Drawing to 3D CAD

**HackFactory 2026 – Alfa Laval Challenge**

**Team name – CADventures*

This project explores automated conversion of 2D engineering drawings into 3D CAD models using image processing, geometric feature extraction, and machine learning.

## Approach

1. **Input Processing:** Load engineering drawings (PNG, JPG, PDF).
2. **Geometry Detection:** Identify lines, circles, and edges.
3. **Feature Interpretation:** Extract dimensions and geometric features.
4. **CAD Generation:** Experiment with converting detected edges into 2D DXF files.
5. **Machine Learning:** Train a baseline model using extracted features and dataset information.

## Project Structure (partial)

```text
src/drawing_to_cad/
├── pipeline.py
└── tools/
    ├── input_loader.py
    ├── geometry_detector.py
    ├── feature_interpreter.py
    ├── edges_to_dxf.py
    └── train_baseline.py
```

## Current Results

- Implemented an initial drawing analysis pipeline.
- Extracted geometric elements, dimensions, and features into structured JSON files.
- Generated preliminary 2D DXF outputs.
- Trained a baseline machine learning model, although prediction accuracy remains limited.

## Next Steps

Improve geometric interpretation, associate dimensions with detected features, and explore more accurate 2D and 3D CAD reconstruction.

**Status:** Experimental prototype developed during HackFactory 2026.