## ARC-AGI Minimal Neuro-Symbolic Solver

This repository contains a compact, test-covered scaffold for solving ARC (Abstraction and Reasoning Corpus) tasks using a neuro-symbolic approach:

- A tiny DSL of grid operations and a `Program` to compose them
- A beam search solver over programs
- Optional learned guidance (simple logistic policy) and a lightweight Knowledge Graph (KG) for behavior-guided ranking
- Utilities and scripts to run on ARC training data

### Repository layout

- `arc_solve/`
  - `dsl.py`: grid `Grid` type, primitive operations, and `Program`
  - `search.py`: `BeamSearchSolver`, operation enumeration, scoring/guidance adapter
  - `kg.py`: predicate extraction, behavior logging, `KnowledgeGraph`, and KG-based guidance
  - `io.py`: load ARC tasks JSON and helpers to extract train/test data
  - `learn.py`: numpy-based features and `LogisticPolicy` (save/load supported)
  - `learn_torch.py`: optional PyTorch variant of logistic policy
  - `cli.py`: single-task command line interface
- `scripts/`
  - `run_sample.py`: sample a subset of ARC training tasks and solve with optional policy+KG guidance
  - `run_full_training.py`: iterate across all training tasks, optional torch policy, save KG
- `arc-prize-2025/`: ARC training/test JSONs used by the scripts
- `tests/`: unit tests covering DSL ops, KG, search, and scripts
- `requirements.txt`: minimal runtime/test deps (numpy, pytest)

### Core concepts

- **Grid (`Grid`)**: a list of rows, each a list of ints (colors 0..9). Zero denotes background in many ops.
- **Operations**: pure functions on grids, e.g.:
  - `Identity`, `CropBoundingBox`, `LargestComponent(treat_zero_as_background=True)`
  - `KeepColor(c)`, `RemoveColor(c)`, `ReplaceColor(a, b)`
  - Symmetries: `ReflectH`, `ReflectV`, `TransposeOp`, `Rotate90`
- **Program**: an ordered list of operations (`steps`) applied sequentially.

### Search: `BeamSearchSolver`

- Enumerates a fixed operation set (`enumerate_operations`) and performs beam search over programs up to `max_depth`.
- Scoring uses `pixel_mismatch` between predicted and target grids with heavy penalties for shape mismatch.
- Maintains a signature cache to avoid duplicate expansions; supports early termination on perfect match.
- Accepts an optional `guidance` with method `rank_operations(base_program, train_pairs, ops)` to reorder exploration.
- Can log behavior via an optional `behavior_logger` with methods `reset_episode()` and `log_step(current, op_sig, post)`.

### Knowledge Graph guidance (`kg.py`)

- Extracts simple, interpretable predicates from grids:
  - `size:HxW`, `bbox:HxW` (non-zero bounding box)
  - `has_color:C` for C in 1..9, and `num_colors:K`
- Computes predicate deltas between successive states: `gained_color:*`, `lost_color:*`, `bbox_shrunk/grew`, `size_changed`, `num_colors_inc/dec/same`.
- Logs behavior during search via `BehaviorLogger`:
  - `enables`: predicate → op
  - `achieves`: op → delta
  - `follows`: op → op (temporal)
- `KnowledgeGraph` stores edge counts and can be saved/loaded as JSON.
- `KGScorer` scores ops for current/target using KG statistics; `KGGuidance` adapts it to the solver guidance API.

### Learned policy guidance (`learn.py`, `learn_torch.py`)

- `basic_features(inp, target, current)`: concatenation of color histograms, sizes, and nonzero bounding boxes.
- `build_imitation_dataset(tasks, max_rollout_steps)`: labels each state with the best single-step op (argmin pixel mismatch) to create (X, y).
- `LogisticPolicy` (numpy): simple multinomial logistic regression with save/load (`.json`).
- `TorchLogisticPolicy` (optional): PyTorch version with GPU support; persistence omitted in this scaffold.
- Scripts can combine policy-guided ranking with KG-guided ranking by averaging normalized ranks.

### Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Optional: install PyTorch for torch-backed policy (see pytorch.org for the right wheel)
```

### Data

The repository includes ARC JSONs under `arc-prize-2025/` used by the scripts:

- Training: `arc-agi_training_challenges.json`, `arc-agi_training_solutions.json`
- Evaluation/test subsets and samples for quick runs

### Usage

#### Solve a single task

```bash
python -m arc_solve.cli --json arc-prize-2025/arc-agi_training_challenges.json \
  --task ff805c23 --max_depth 2 --beam 64
```

Prints a JSON with the learned program signature and predictions for test inputs.

#### Run a sampled benchmark with optional policy and KG guidance

```bash
python scripts/run_sample.py --sample_size 20 --seed 0 --max_depth 2 --beam 64 \
  --use_policy --policy_rollout_steps 1 --policy_epochs 10 --policy_lr 0.5 \
  --save_policy --policy_path /tmp/policy.json
```

Key options:

- `--use_policy`: train a lightweight policy on sampled train pairs
- `--policy_backend {numpy,torch}` and `--device {cpu,cuda}`
- `--load_policy/--save_policy` with `--policy_path` (numpy backend only)

Outputs a summary JSON with task solve rate, test accuracy, per-task details, and policy info.

#### Run across all training tasks and optionally save the KG

```bash
python scripts/run_full_training.py --max_depth 2 --beam 64 --save_kg ./kg.json
```

Options:

- `--use_policy --policy_backend torch` to enable torch policy guidance (numpy path intentionally omitted here for minimalism)
- `--save_kg`: persist the KG as JSON with edge counts

### API highlights

- `arc_solve.dsl`
  - `Program(steps: List[Operation])`
  - Operations: `Identity`, `CropBoundingBox`, `LargestComponent`, `KeepColor`, `RemoveColor`, `ReplaceColor`, `ReflectH`, `ReflectV`, `TransposeOp`, `Rotate90`
- `arc_solve.search`
  - `BeamSearchSolver(max_depth: int, beam_width: int, guidance=None, behavior_logger=None)`
  - `PolicyGuidance(score_fn)` where `score_fn(inp, target, current, op) -> float`
- `arc_solve.kg`
  - `KnowledgeGraph.save(path) / KnowledgeGraph.load(path)`
  - `BehaviorLogger(graph).log_step(current, op_sig, post)`
  - `KGGuidance(KGScorer(kg))`
- `arc_solve.learn`
  - `build_imitation_dataset(tasks, max_rollout_steps)` → `(X, y, op_signatures)`
  - `LogisticPolicy.init(dim, op_signatures)`; `.fit(X, y, ...)`; `.predict_log_proba(X)`; `.save/.load`

### Testing

Run the unit test suite:

```bash
pytest -q
```

Coverage includes:

- DSL ops and helpers (`tests/test_dsl.py`)
- Knowledge Graph logging and guidance (`tests/test_kg.py`)
- Policy training and solver integration (`tests/test_policy.py`, `tests/test_policy_persist.py`)
- Search behavior on small tasks (`tests/test_search_small.py`)
- Script entrypoints smoke tests (`tests/test_run_sample.py`, `tests/test_run_full_training.py`)

### Extending the solver

- **Add a new operation**
  - Implement an `Operation` subclass in `arc_solve/dsl.py` and add it to `enumerate_operations()` in `arc_solve/search.py`.
  - Add/adjust tests to cover semantics of the new op.
- **Change the loss or program prior**
  - Modify `pixel_mismatch` or the beam ordering in `search.py`.
- **Add new features/guidance**
  - Extend `basic_features` or implement a new policy; create a `rank_operations` adapter similar to `PolicyGuidance`.
  - Enhance the KG by adding new predicates/deltas in `kg.py`.

### Notes

- Torch support is optional and only used for the `TorchLogisticPolicy` in scripts. If PyTorch is not installed, the numpy backend remains available.
- JSON inputs/outputs for scripts are kept minimal to encourage easy experimentation and integration.


