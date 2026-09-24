"""Serialization and deserialization utilities for canonical CircuitData."""

import json
from pathlib import Path
from typing import Union

from qco.circuits.representation import CircuitData


def serialize_to_json(circuit: CircuitData, indent: int = 2) -> str:
    """Serialize CircuitData to JSON string."""
    return json.dumps(circuit.to_dict(), indent=indent)


def deserialize_from_json(json_str: str) -> CircuitData:
    """Deserialize CircuitData from JSON string."""
    data = json.loads(json_str)
    return CircuitData.from_dict(data)


def save_circuit(circuit: CircuitData, file_path: Union[str, Path]) -> None:
    """Save CircuitData to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(circuit.to_dict(), f, indent=2)


def load_circuit(file_path: Union[str, Path]) -> CircuitData:
    """Load CircuitData from JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Circuit file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CircuitData.from_dict(data)
