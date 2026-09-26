"""Run two-stage court grounding with DeepSeek Vision and visualize it."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

try:
    from .infer import get_ground_truth, load_dataset, resolve_image
    from .schema import COURT_LABELS, extract_json, parse_prediction_lenient
    from .visualize import visualize_results
except ImportError:  # Supports direct script execution.
    from infer import get_ground_truth, load_dataset, resolve_image
    from schema import COURT_LABELS, extract_json, parse_prediction_lenient
    from visualize import visualize_results


LABEL_ORDER = [
    "sideline",
    "baseline",
    "half_court_line",
    "free_throw_line",
    "paint_boundary",
    "three_point_line",
    "center_circle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Infer one image directly")
    source.add_argument("--data", type=Path, help="Infer a conversation JSON dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("court_grounding/outputs/predictions_deepseek.json"),
    )
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        help="Visualization directory; defaults to <output parent>/visualized_deepseek",
    )
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--detail", choices=["low", "high", "original", "auto"], default="original"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError(f"image exceeds the 32 MiB inline-image limit: {path}")
    with Image.open(path) as image:
        image_format = (image.format or "").upper()
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }
    if image_format not in mime_types:
        raise ValueError(f"unsupported image format: {image_format or 'unknown'}")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_types[image_format]};base64,{encoded}"


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    parts = []
    for item in output or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content", [])
        for part in content or []:
            part_type = getattr(part, "type", None)
            part_text = getattr(part, "text", None)
            if isinstance(part, dict):
                part_type, part_text = part.get("type"), part.get("text")
            if part_type in {"output_text", "text"} and isinstance(part_text, str):
                parts.append(part_text)
    if parts:
        return "\n".join(parts)
    raise ValueError("DeepSeek response did not contain output text")


def call_vision(client: Any, model: str, prompt: str, data_url: str, detail: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
    )
    return response_text(response)


def parse_visible_labels(text: str) -> list[str]:
    value = extract_json(text)
    if not isinstance(value, dict) or not isinstance(value.get("visible_labels"), list):
        raise ValueError('presence response must be {"visible_labels": [...]}')
    labels = []
    for label in value["visible_labels"]:
        if label not in COURT_LABELS:
            raise ValueError(f"invalid visible label: {label!r}")
        if label not in labels:
            labels.append(label)
    return [label for label in LABEL_ORDER if label in labels]


def presence_prompt(width: int, height: int) -> str:
    labels = ", ".join(LABEL_ORDER)
    return f"""Examine this basketball image and classify only court markings that are
unambiguously visible as painted regulation lines. Ignore players, uniforms,
shadows, floor-board seams, logos, logo edges, advertisements, and decorations.
Do not infer lines outside the frame or behind objects. It is correct to return
an empty list. Candidate labels: {labels}.

The image is {width} pixels wide and {height} pixels high. This step is presence
classification only; do not return coordinates. Return JSON only:
{{"visible_labels":["half_court_line"]}}"""


def localization_prompt(label: str, width: int, height: int) -> str:
    curved = label in {"three_point_line", "center_circle"}
    geometry_type = "polyline" if curved else "line"
    geometry_instruction = (
        "Return ordered points along the visible curve."
        if curved
        else "Return exactly two visible endpoints."
    )
    return f"""Locate only the visible painted basketball marking labeled {label!r}.
Do not locate any other marking. Ignore logos, logo edges, advertisements,
players, shadows, and floor-board seams. {geometry_instruction}

Use original-image pixel coordinates. The image is exactly {width} pixels wide
and {height} pixels high: x must be 0..{width - 1} and y must be 0..{height - 1}.
If it cannot be located unambiguously, return an empty court_lines list.
Return JSON only in this shape:
{{"court_lines":[{{"label":"{label}","type":"{geometry_type}","points":[[x1,y1],[x2,y2]]}}]}}"""


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key: export {args.api_key_env}='...'")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url, max_retries=args.max_retries)
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")
        dataset_path = Path.cwd() / "direct_image_input.json"
        dataset = [{"image": str(image_path)}]
    else:
        dataset_path = args.data.resolve()
        dataset = load_dataset(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    results = []
    for item in tqdm(dataset, desc="DeepSeek court grounding"):
        image_value = item.get("image")
        result: dict[str, Any] = {
            "image": image_value,
            "ground_truth": get_ground_truth(item),
            "model": args.model,
        }
        try:
            if not isinstance(image_value, str):
                raise ValueError("sample.image must be a path string")
            image_path = resolve_image(image_value, dataset_path)
            with Image.open(image_path) as image:
                width, height = image.size
            data_url = image_data_url(image_path)

            presence_raw = call_vision(
                client, args.model, presence_prompt(width, height), data_url, args.detail
            )
            visible_labels = parse_visible_labels(presence_raw)
            result.update(
                {
                    "resolved_image": str(image_path),
                    "image_size": [width, height],
                    "presence_raw": presence_raw,
                    "visible_labels": visible_labels,
                }
            )

            court_lines = []
            localization_raw: dict[str, str] = {}
            validation_warnings = []
            for label in visible_labels:
                raw = call_vision(
                    client,
                    args.model,
                    localization_prompt(label, width, height),
                    data_url,
                    args.detail,
                )
                localization_raw[label] = raw
                parsed, warnings = parse_prediction_lenient(raw, width=width, height=height)
                validation_warnings.extend(f"{label}: {warning}" for warning in warnings)
                matching = [line for line in parsed["court_lines"] if line["label"] == label]
                if len(matching) > 1:
                    validation_warnings.append(f"{label}: returned multiple instances")
                court_lines.extend(matching)

            result.update(
                {
                    "localization_raw": localization_raw,
                    "prediction": {"court_lines": court_lines},
                    "validation_warnings": validation_warnings,
                    "parse_error": None,
                }
            )
        except Exception as error:
            result.update({"prediction": None, "error": str(error)})
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    visualize_dir = args.visualize_dir or args.output.parent / "visualized_deepseek"
    written = visualize_results(args.output, visualize_dir)
    valid = sum(result.get("prediction") is not None for result in results)
    print(f"Saved {len(results)} predictions to {args.output} ({valid} completed)")
    print(f"Saved {written} visualizations to {visualize_dir}")


if __name__ == "__main__":
    main()
