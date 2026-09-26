"""Draw court-grounding predictions over their source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "sideline": "#00ff66",
    "baseline": "#ff3b30",
    "half_court_line": "#00c7ff",
    "free_throw_line": "#ffd60a",
    "paint_boundary": "#bf5af2",
    "three_point_line": "#ff9f0a",
    "center_circle": "#64d2ff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--line-width", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.predictions.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)

    written = 0
    for index, result in enumerate(results):
        prediction = result.get("prediction")
        image_value = result.get("resolved_image") or result.get("image")
        if not prediction or not image_value or not Path(image_value).exists():
            continue
        with Image.open(image_value) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in prediction.get("court_lines", []):
            points = [tuple(point) for point in line["points"]]
            color = COLORS.get(line["label"], "white")
            draw.line(points, fill=color, width=args.line_width, joint="curve")
            x, y = points[0]
            draw.text((x + 4, y + 4), line["label"], fill=color, font=font, stroke_width=2, stroke_fill="black")
        source_name = Path(image_value).stem
        image.save(args.output_dir / f"{index:05d}_{source_name}.jpg", quality=95)
        written += 1
    print(f"Wrote {written} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()

