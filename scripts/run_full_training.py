from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

from arc_solve.dsl import Grid
from arc_solve.search import BeamSearchSolver
from arc_solve.learn import build_imitation_dataset
from arc_solve.learn_torch import TorchLogisticPolicy  # optional, used if torch is present
from arc_solve.search import PolicyGuidance
from arc_solve.learn import basic_features
from arc_solve.kg import KnowledgeGraph, BehaviorLogger, KGScorer, KGGuidance


def run_full_training(
    challenges_path: str,
    solutions_path: str,
    max_depth: int = 2,
    beam: int = 64,
    save_kg_path: str | None = None,
    use_policy: bool = False,
    policy_backend: str = "numpy",
    device: str | None = None,
    policy_epochs: int = 5,
    policy_lr: float = 0.5,
) -> Dict[str, Any]:
    with open(challenges_path, "r") as f:
        challenges = json.load(f)
    with open(solutions_path, "r") as f:
        solutions = json.load(f)

    task_ids = list(set(challenges.keys()) & set(solutions.keys()))

    kg = KnowledgeGraph()
    logger = BehaviorLogger(kg)
    kg_guidance = KGGuidance(KGScorer(kg))

    # Optional global policy training on all training pairs for extra guidance
    policy_info: Dict[str, Any] = {"used": False}
    composite_guidance = kg_guidance
    if use_policy:
        all_pairs: List[Tuple[Grid, Grid]] = []
        for tid in task_ids:
            task = challenges[tid]
            all_pairs.extend([(ex["input"], ex["output"]) for ex in task.get("train", [])])
        X, y, op_sigs = build_imitation_dataset(all_pairs, max_rollout_steps=1)
        guidance = None
        if policy_backend == "torch":
            try:
                import torch  # type: ignore

                torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                torch_device = "cpu"
            policy = TorchLogisticPolicy.init(dim=X.shape[1], op_signatures=op_sigs, device=torch_device)
            policy.fit(X, y, lr=policy_lr, epochs=policy_epochs)

            def score_fn(inp: Grid, target: Grid, current: Grid, op) -> float:
                import numpy as np

                x = basic_features(inp, target, current)[None, :]
                logp = policy.predict_log_proba(x)[0]
                idx = op_sigs.index(op.signature())
                return float(logp[idx])

            guidance = PolicyGuidance(score_fn)
            policy_info = {"used": True, "backend": "torch", "device": torch_device}
        else:
            # numpy backend is already available in scripts.run_sample; keep full training minimal
            guidance = None
        if guidance is not None:
            class _Composite:
                def __init__(self, g1, g2):
                    self.g1 = g1
                    self.g2 = g2

                def rank_operations(self, base, train_pairs, ops):
                    ranked1 = self.g1.rank_operations(base, train_pairs, ops)
                    ranked2 = self.g2.rank_operations(base, train_pairs, ops)
                    idx1 = {o.signature(): i for i, o in enumerate(ranked1)}
                    idx2 = {o.signature(): i for i, o in enumerate(ranked2)}
                    def score(o):
                        r1 = idx1.get(o.signature(), len(ops))
                        r2 = idx2.get(o.signature(), len(ops))
                        return r1 + r2
                    return sorted(ops, key=score)

            composite_guidance = _Composite(kg_guidance, guidance)

    total_tests = 0
    matched_tests = 0
    tasks_solved = 0
    per_task: List[Dict[str, Any]] = []

    for tid in task_ids:
        task = challenges[tid]
        solver = BeamSearchSolver(max_depth=max_depth, beam_width=beam, guidance=composite_guidance, behavior_logger=logger)
        train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        program = solver.fit(train_pairs)
        test_inputs = [ex["input"] if isinstance(ex, dict) and "input" in ex else ex for ex in task.get("test", [])]
        preds = [program.apply(g) for g in test_inputs]
        gold = solutions.get(tid, [])

        num_tests = len(gold)
        correct = 0
        for p, g in zip(preds, gold):
            # Equality check inline to avoid importing grids_equal here
            same = (len(p) == len(g)) and all(pr == gr for pr, gr in zip(p, g))
            if same:
                correct += 1
        total_tests += num_tests
        matched_tests += correct
        solved = correct == num_tests and num_tests > 0
        if solved:
            tasks_solved += 1
        per_task.append({
            "task": tid,
            "program": program.signature(),
            "num_tests": num_tests,
            "num_correct": correct,
            "solved": solved,
        })

    if save_kg_path:
        kg.save(save_kg_path)

    summary = {
        "num_tasks": len(task_ids),
        "tasks_solved": tasks_solved,
        "total_tests": total_tests,
        "matched_tests": matched_tests,
        "test_accuracy": (matched_tests / total_tests) if total_tests > 0 else 0.0,
        "task_solve_rate": (tasks_solved / len(task_ids)) if task_ids else 0.0,
        "details": per_task,
        "kg_edges": kg.num_edges(),
        "kg_saved": bool(save_kg_path),
        "kg_path": save_kg_path,
        "policy": policy_info,
    }
    return summary


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run full training across ARC training set with KG logging/guidance")
    parser.add_argument("--challenges", default="arc-prize-2025/arc-agi_training_challenges.json")
    parser.add_argument("--solutions", default="arc-prize-2025/arc-agi_training_solutions.json")
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--save_kg", type=str, default=None, help="Path to save the learned KG JSON")
    parser.add_argument("--use_policy", action="store_true")
    parser.add_argument("--policy_backend", type=str, default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--policy_epochs", type=int, default=5)
    parser.add_argument("--policy_lr", type=float, default=0.5)
    args = parser.parse_args(argv)

    summary = run_full_training(
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        max_depth=args.max_depth,
        beam=args.beam,
        save_kg_path=args.save_kg,
        use_policy=args.use_policy,
        policy_backend=args.policy_backend,
        device=args.device,
        policy_epochs=args.policy_epochs,
        policy_lr=args.policy_lr,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


