from __future__ import annotations

import argparse
import json
from typing import Any

from .io import load_task, to_train_pairs, to_test_inputs
from .search import BeamSearchSolver


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal ARC solver")
    parser.add_argument("--json", required=True, help="Path to ARC tasks JSON file")
    parser.add_argument("--task", required=True, help="Task id in the JSON file")
    parser.add_argument("--max_depth", type=int, default=2, help="Max program length")
    parser.add_argument("--beam", type=int, default=64, help="Beam width")
    args = parser.parse_args(argv)

    task = load_task(args.json, args.task)
    solver = BeamSearchSolver(max_depth=args.max_depth, beam_width=args.beam)

    train_pairs = to_train_pairs(task)
    program = solver.fit(train_pairs)

    tests = to_test_inputs(task)
    preds = [program.apply(t) for t in tests]
    print(json.dumps({"program": program.signature(), "predictions": preds}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


