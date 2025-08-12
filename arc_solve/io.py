from __future__ import annotations

import json
from typing import Dict, List, Tuple

from .dsl import Grid


def load_task(json_path: str, task_id: str) -> Dict:
    """Load a single ARC task by id from a JSON file.

    Returns a dict with keys: 'train' (list of dicts with 'input' and 'output' grids)
    and 'test' (list of input grids).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    task = data[task_id]
    # Ensure deep copy of lists (json already loads fresh lists)
    return task


def to_train_pairs(task: Dict) -> List[Tuple[Grid, Grid]]:
    return [(ex["input"], ex["output"]) for ex in task.get("train", [])]


def to_test_inputs(task: Dict) -> List[Grid]:
    return [ex for ex in task.get("test", [])]


