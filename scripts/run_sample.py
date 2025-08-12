from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

from arc_solve.dsl import Grid, grids_equal
from arc_solve.io import load_task
from arc_solve.learn import LogisticPolicy, basic_features, build_imitation_dataset
from arc_solve.search import BeamSearchSolver, PolicyGuidance
from arc_solve.kg import KnowledgeGraph, BehaviorLogger, KGScorer, KGGuidance


def _score_fn_from_policy(policy):
    op_index = {sig: i for i, sig in enumerate(policy.op_signatures)}

    def score_fn(inp: Grid, target: Grid, current: Grid, op) -> float:
        import numpy as np

        x = basic_features(inp, target, current)[None, :]
        logp = policy.predict_log_proba(x)[0]
        return float(logp[op_index[op.signature()]])

    return score_fn


def run_sample(
    challenges_path: str,
    solutions_path: str,
    sample_size: int = 20,
    seed: int = 0,
    max_depth: int = 2,
    beam: int = 64,
    use_policy: bool = False,
    policy_rollout_steps: int = 1,
    policy_epochs: int = 10,
    policy_lr: float = 0.5,
    policy_path: str | None = None,
    load_policy: bool = False,
    save_policy: bool = False,
    policy_backend: str = "numpy",  # "numpy" or "torch"
    device: str | None = None,  # "cpu" or "cuda"
) -> Dict[str, Any]:
    with open(challenges_path, "r") as f:
        challenges = json.load(f)
    with open(solutions_path, "r") as f:
        solutions = json.load(f)

    task_ids = list(set(challenges.keys()) & set(solutions.keys()))
    rnd = random.Random(seed)
    rnd.shuffle(task_ids)
    sample = task_ids[: max(1, min(sample_size, len(task_ids)))]

    # Optional: train a global policy on sampled tasks
    guidance = None
    policy_info: Dict[str, Any] = {"used": False, "loaded": False, "saved": False, "path": policy_path}
    if load_policy and policy_path and policy_backend == "numpy":
        try:
            policy = LogisticPolicy.load(policy_path)
            guidance = PolicyGuidance(_score_fn_from_policy(policy))
            policy_info.update({"used": True, "loaded": True})
        except FileNotFoundError:
            # If requested to load but file not found, fall back to training when use_policy=True
            pass
    elif load_policy and policy_backend == "torch":
        # Persistence not implemented for torch backend
        pass
    if guidance is None and use_policy:
        all_pairs: List[Tuple[Grid, Grid]] = []
        for tid in sample:
            task = challenges[tid]
            all_pairs.extend([(ex["input"], ex["output"]) for ex in task.get("train", [])])
        X, y, op_sigs = build_imitation_dataset(all_pairs, max_rollout_steps=policy_rollout_steps)
        if policy_backend == "torch":
            # Lazy import to avoid hard dependency
            from arc_solve.learn_torch import TorchLogisticPolicy
            torch_device = "cpu"
            if device is not None:
                torch_device = device
            else:
                try:
                    import torch  # type: ignore

                    torch_device = "cuda" #if torch.cuda.is_available() else "cpu"
                except Exception:
                    torch_device = "cpu"
            policy = TorchLogisticPolicy.init(dim=X.shape[1], op_signatures=op_sigs, device=torch_device)
            policy.fit(X, y, lr=policy_lr, epochs=policy_epochs)
            # No save/load for torch backend in this minimal scaffold
            guidance = PolicyGuidance(_score_fn_from_policy(policy))
            policy_info.update({"used": True, "backend": "torch", "device": torch_device})
        else:
            policy = LogisticPolicy.init(dim=X.shape[1], op_signatures=op_sigs)
            policy.fit(X, y, lr=policy_lr, epochs=policy_epochs)
            if save_policy and policy_path:
                policy.save(policy_path)
                policy_info.update({"saved": True})
            guidance = PolicyGuidance(_score_fn_from_policy(policy))
            policy_info.update({"used": True, "backend": "numpy"})

    total_tests = 0
    matched_tests = 0
    tasks_solved = 0
    per_task: List[Dict[str, Any]] = []

    # Initialize global KG and behavior logger
    kg = KnowledgeGraph()
    behavior_logger = BehaviorLogger(graph=kg)

    for tid in sample:
        task = challenges[tid]
        # Compose KG-based guidance with policy guidance if any
        kg_guidance = KGGuidance(KGScorer(kg))

        class _CompositeGuidance:
            def __init__(self, guidances):
                self.guidances = [g for g in guidances if g is not None]

            def rank_operations(self, base, train_pairs, ops):
                # Average normalized ranks from all guidances
                if not self.guidances:
                    return ops
                scored_lists = []
                for g in self.guidances:
                    ranked = g.rank_operations(base, train_pairs, ops)
                    index = {op.signature(): i for i, op in enumerate(ranked)}
                    # Lower rank index is better → convert to score by inverse rank
                    scores = {op.signature(): (len(ranked) - index.get(op.signature(), len(ranked))) for op in ops}
                    scored_lists.append(scores)
                # aggregate
                agg_scores = {op.signature(): 0.0 for op in ops}
                for scores in scored_lists:
                    for sig, sc in scores.items():
                        agg_scores[sig] += sc
                ops_sorted = sorted(ops, key=lambda o: (-agg_scores[o.signature()], o.signature()))
                return ops_sorted

        composite = _CompositeGuidance([kg_guidance, guidance])

        solver = BeamSearchSolver(max_depth=max_depth, beam_width=beam, guidance=composite, behavior_logger=behavior_logger)
        train_pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        program = solver.fit(train_pairs)
        test_inputs = [ex["input"] if isinstance(ex, dict) and "input" in ex else ex for ex in task.get("test", [])]
        preds = [program.apply(g) for g in test_inputs]
        gold = solutions[tid]

        num_tests = len(gold)
        correct = 0
        for p, g in zip(preds, gold):
            if grids_equal(p, g):
                correct += 1
        total_tests += num_tests
        matched_tests += correct
        solved = correct == num_tests and num_tests > 0
        if solved:
            tasks_solved += 1
        per_task.append(
            {
                "task": tid,
                "program": program.signature(),
                "num_tests": num_tests,
                "num_correct": correct,
                "solved": solved,
            }
        )

    summary = {
        "num_tasks": len(sample),
        "tasks_solved": tasks_solved,
        "total_tests": total_tests,
        "matched_tests": matched_tests,
        "test_accuracy": (matched_tests / total_tests) if total_tests > 0 else 0.0,
        "task_solve_rate": (tasks_solved / len(sample)) if sample else 0.0,
        "details": per_task,
        "policy": policy_info,
    }
    return summary


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run a sample of ARC training tasks with the solver")
    parser.add_argument("--challenges", default="arc-prize-2025/arc-agi_training_challenges.json")
    parser.add_argument("--solutions", default="arc-prize-2025/arc-agi_training_solutions.json")
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--use_policy", action="store_true")
    parser.add_argument("--policy_rollout_steps", type=int, default=1)
    parser.add_argument("--policy_epochs", type=int, default=10)
    parser.add_argument("--policy_lr", type=float, default=0.5)
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--load_policy", action="store_true")
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--policy_backend", type=str, default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args(argv)

    summary = run_sample(
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        sample_size=args.sample_size,
        seed=args.seed,
        max_depth=args.max_depth,
        beam=args.beam,
        use_policy=args.use_policy,
        policy_rollout_steps=args.policy_rollout_steps,
        policy_epochs=args.policy_epochs,
        policy_lr=args.policy_lr,
        policy_path=args.policy_path,
        load_policy=args.load_policy,
        save_policy=args.save_policy,
        policy_backend=args.policy_backend,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


