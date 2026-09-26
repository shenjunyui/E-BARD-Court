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

