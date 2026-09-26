"""Run EBQwen/Qwen2.5-VL court-marking grounding inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

try:
    from .schema import parse_prediction
except ImportError:  # Supports: python court_grounding/src/infer.py
    from schema import parse_prediction


DEFAULT_PROMPT = """Identify all visible basketball court markings in this image.
Use only these labels: sideline, baseline, half_court_line, free_throw_line,
paint_boundary, three_point_line, center_circle.
Represent each straight marking as type \"line\" with exactly two visible pixel
endpoints. Represent each curved marking as type \"polyline\" with ordered pixel
points along the visible curve. Coordinates must use the original image pixel
coordinate system with origin [0, 0] at the top-left.
Ignore players, balls, hoops, advertisements, logos, floor decorations, shadows,
and all non-court lines. Do not infer invisible geometry.
Return JSON only in this exact shape:
{"court_lines":[{"label":"sideline","type":"line","points":[[x1,y1],[x2,y2]]}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Fine-tuned EBQwen/Qwen model directory or Hub ID")
    parser.add_argument("--processor", help="Processor directory or Hub ID; defaults to --model")
    parser.add_argument("--data", required=True, type=Path, help="Input conversation JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output predictions JSON")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1200 * 28 * 28)
    parser.add_argument("--limit", type=int, help="Only process the first N samples")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise ValueError("dataset root must be a JSON list")
    return dataset


def get_prompt(item: dict[str, Any]) -> str:
    if isinstance(item.get("prompt"), str) and item["prompt"].strip():
        return item["prompt"].strip()
    for turn in item.get("conversations", []):
        if turn.get("from") == "human":
            prompt = str(turn.get("value", "")).replace("<image>\n", "", 1).strip()
            return prompt or DEFAULT_PROMPT
    return DEFAULT_PROMPT


def get_ground_truth(item: dict[str, Any]) -> str | dict | None:
    if "ground_truth" in item:
        return item["ground_truth"]
    for turn in item.get("conversations", []):
        if turn.get("from") in {"gpt", "assistant"}:
            return turn.get("value") or None
    return None


def resolve_image(image_value: str, dataset_path: Path) -> Path:
    image_path = Path(image_value).expanduser()
    candidates = [image_path]
    if not image_path.is_absolute():
        candidates = [dataset_path.parent / image_path, Path.cwd() / image_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"image not found: {image_value}")


def main() -> None:
    args = parse_args()

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this inference entry point")

    processor_source = args.processor or args.model
    print(f"Loading model: {args.model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=args.trust_remote_code,
    )

    dataset_path = args.data.resolve()
    dataset = load_dataset(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    results = []
    for item in tqdm(dataset, desc="Court grounding"):
        image_value = item.get("image")
        result: dict[str, Any] = {"image": image_value, "ground_truth": get_ground_truth(item)}
        try:
            if not isinstance(image_value, str):
                raise ValueError("sample.image must be a path string")
            image_path = resolve_image(image_value, dataset_path)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            prompt = get_prompt(item)
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            input_device = next(model.parameters()).device
            inputs = inputs.to(input_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
            raw = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            result.update({
                "resolved_image": str(image_path),
                "image_size": [width, height],
                "prompt": prompt,
                "prediction_raw": raw,
            })
            try:
                result["prediction"] = parse_prediction(raw, width=width, height=height)
                result["parse_error"] = None
            except ValueError as error:
                result["prediction"] = None
                result["parse_error"] = str(error)
        except Exception as error:  # Preserve per-sample failures in batch jobs.
            result.update({"prediction": None, "prediction_raw": None, "error": str(error)})
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    valid = sum(result.get("prediction") is not None for result in results)
    print(f"Saved {len(results)} predictions to {args.output} ({valid} schema-valid)")


if __name__ == "__main__":
    main()

