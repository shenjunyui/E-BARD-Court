"""Validate court-grounding dataset paths and ground-truth JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from .infer import get_ground_truth, resolve_image
    from .schema import parse_prediction, validate_prediction
except ImportError:
    from infer import get_ground_truth, resolve_image
    from schema import parse_prediction, validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--allow-empty-ground-truth", action="store_true")
    args = parser.parse_args()

    dataset_path = args.data.resolve()
    with dataset_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    errors = []
    for index, sample in enumerate(samples):
        try:
            image_path = resolve_image(sample["image"], dataset_path)
            with Image.open(image_path) as image:
                width, height = image.size
            ground_truth = get_ground_truth(sample)
            if ground_truth is None:
                if not args.allow_empty_ground_truth:
                    raise ValueError("ground truth is empty")
            elif isinstance(ground_truth, str):
                parse_prediction(ground_truth, width=width, height=height)
            else:
                validate_prediction(ground_truth, width=width, height=height)
        except Exception as error:
            errors.append(f"sample {index}: {error}")

    for error in errors:
        print(error)
    if errors:
        raise SystemExit(f"Dataset validation failed: {len(errors)}/{len(samples)} invalid samples")
    print(f"Dataset is valid: {len(samples)} samples")


if __name__ == "__main__":
    main()

