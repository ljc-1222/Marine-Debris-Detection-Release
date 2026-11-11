"""Create an unlabeled image index with the fixed 34-class category mapping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.detector_geometry import encoded_image_info

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_index(images: Path, output: Path) -> dict:
    root = images.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(output)
    files = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS),
                   key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError("The image directory contains no supported images")
    rows = []
    for identifier, path in enumerate(files, 1):
        width, height, _ = encoded_image_info(path)
        rows.append({"id": identifier, "file_name": path.relative_to(root).as_posix(),
                     "width": width, "height": height})
    categories = json.loads((ROOT / "configs/categories.json").read_text(encoding="utf-8"))["categories"]
    document = {"info": {"description": "Unlabeled marine-debris inference images"},
                "images": rows, "categories": categories, "annotations": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_index(args.images, args.output)
    print(json.dumps({"images": len(document["images"]), "categories": len(document["categories"])}))


if __name__ == "__main__":
    main()
