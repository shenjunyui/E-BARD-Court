"""Parsing and validation for court-grounding model output."""

from __future__ import annotations

import json
import re
from typing import Any


COURT_LABELS = {
    "sideline",
    "baseline",
    "half_court_line",
    "free_throw_line",
    "paint_boundary",
    "three_point_line",
    "center_circle",
}

GEOMETRY_TYPES = {"line", "polyline"}


def extract_json(text: str) -> Any:
    """Extract the first valid JSON object/array from a model response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("model response is empty")

    candidates = [text.strip()]
    candidates.extend(
        block.strip()
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    )

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value
            except json.JSONDecodeError:
                continue

    raise ValueError("no valid JSON object or array found in model response")


def validate_prediction(value: Any, width: int | None = None, height: int | None = None) -> dict:
    """Validate and normalize the public court-grounding schema.

    Returns a clean ``{"court_lines": [...]}`` dictionary. Coordinates are
    preserved as floats and must be pixel coordinates inside the image when
    width/height are supplied.
    """
    if isinstance(value, list):
        value = {"court_lines": value}
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")

    lines = value.get("court_lines")
    if not isinstance(lines, list):
        raise ValueError("court_lines must be a list")

    normalized = []
    for index, item in enumerate(lines):
        prefix = f"court_lines[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")

        label = item.get("label")
        geometry_type = item.get("type")
        points = item.get("points")
        if label not in COURT_LABELS:
            raise ValueError(f"{prefix}.label is invalid: {label!r}")
        if geometry_type not in GEOMETRY_TYPES:
            raise ValueError(f"{prefix}.type must be line or polyline")
        if not isinstance(points, list):
            raise ValueError(f"{prefix}.points must be a list")
        if geometry_type == "line" and len(points) != 2:
            raise ValueError(f"{prefix}.points must contain exactly 2 points for a line")
        if geometry_type == "polyline" and len(points) < 2:
            raise ValueError(f"{prefix}.points must contain at least 2 points for a polyline")

        clean_points = []
        for point_index, point in enumerate(points):
            point_prefix = f"{prefix}.points[{point_index}]"
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{point_prefix} must be [x, y]")
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"{point_prefix} coordinates must be numbers")
            x, y = float(x), float(y)
            if width is not None and not 0 <= x < width:
                raise ValueError(f"{point_prefix} x={x} is outside image width {width}")
            if height is not None and not 0 <= y < height:
                raise ValueError(f"{point_prefix} y={y} is outside image height {height}")
            clean_points.append([x, y])

        normalized.append({"label": label, "type": geometry_type, "points": clean_points})

    return {"court_lines": normalized}


def parse_prediction(text: str, width: int | None = None, height: int | None = None) -> dict:
    """Extract JSON from text, then validate it."""
    return validate_prediction(extract_json(text), width=width, height=height)


def parse_prediction_lenient(
    text: str,
    width: int,
    height: int,
    boundary_tolerance: float = 0.1,
) -> tuple[dict, list[str]]:
    """Parse predictions for visualization without discarding a whole image.

    Broadcast-image models sometimes emit coordinates for their internally
    resized image. Coordinates no more than ``boundary_tolerance`` beyond an
    edge are clipped to the original image. Structurally invalid individual
    markings are skipped and reported as warnings. Ground-truth validation
    remains strict through :func:`parse_prediction`.
    """
    value = extract_json(text)
    if isinstance(value, list):
        value = {"court_lines": value}
    if not isinstance(value, dict) or not isinstance(value.get("court_lines"), list):
        raise ValueError("court_lines must be a list")

    clean_lines = []
    warnings = []
    x_margin = max(1.0, width * boundary_tolerance)
    y_margin = max(1.0, height * boundary_tolerance)

    for index, original in enumerate(value["court_lines"]):
        if not isinstance(original, dict):
            warnings.append(f"court_lines[{index}] skipped: marking must be an object")
            continue
        item = dict(original)
        points = item.get("points")
        if isinstance(points, list):
            clipped_points = []
            can_clip = True
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    can_clip = False
                    break
                x, y = point
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    can_clip = False
                    break
                if not -x_margin <= x <= (width - 1) + x_margin:
                    can_clip = False
                    break
                if not -y_margin <= y <= (height - 1) + y_margin:
                    can_clip = False
                    break
                clipped_points.append([
                    min(max(float(x), 0.0), float(width - 1)),
                    min(max(float(y), 0.0), float(height - 1)),
                ])
            if can_clip:
                if clipped_points != points:
                    warnings.append(f"court_lines[{index}] coordinates clipped to image bounds")
                item["points"] = clipped_points

        try:
            validated = validate_prediction(
                {"court_lines": [item]}, width=width, height=height
            )
            clean_lines.append(validated["court_lines"][0])
        except ValueError as error:
            warnings.append(f"court_lines[{index}] skipped: {error}")

    return {"court_lines": clean_lines}, warnings
