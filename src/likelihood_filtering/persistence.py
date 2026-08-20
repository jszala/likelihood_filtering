from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(data), indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def write_metrics(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write replicate metrics in CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("metrics must contain at least one row")
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: str | Path) -> list[dict[str, str]]:
    """Read replicate metrics from CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
