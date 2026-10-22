
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}


def load_drawing(path: str | Path) -> list[np.ndarray]:
    """Load a drawing as one or more OpenCV images (BGR)."""
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Drawing not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {extension}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires PyMuPDF. Install it with: "
                "pip install pymupdf"
            ) from exc

        images = []
        with pymupdf.open(path) as document:
            for page in document:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )
                image = np.frombuffer(
                    pixmap.samples, dtype=np.uint8
                ).reshape(pixmap.height, pixmap.width, pixmap.n)

                images.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if not images:
            raise ValueError(f"The PDF contains no pages: {path}")

        return images

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read drawing: {path}")

    return [image]
