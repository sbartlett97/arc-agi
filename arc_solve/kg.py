from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json

from .dsl import Grid


# ------------------------------
# Predicate extraction utilities
# ------------------------------


def _grid_size(grid: Grid) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def _nonzero_bbox(grid: Grid) -> Tuple[int, int]:
    h, w = _grid_size(grid)
    if h == 0 or w == 0:
        return 0, 0
    min_r, min_c = h, w
    max_r, max_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                if r < min_r:
                    min_r = r
                if c < min_c:
                    min_c = c
                if r > max_r:
                    max_r = r
                if c > max_c:
                    max_c = c
    if max_r == -1:
        return 0, 0
    return (max_r - min_r + 1, max_c - min_c + 1)


def extract_predicates(grid: Grid) -> Set[str]:
    """Extract simple symbolic predicates from a grid.

    Predicates are strings such as:
      - size:HxW
      - bbox:HxW (non-zero bounding box)
      - has_color:C for C in 1..9 present in the grid
      - num_colors:K where K counts colors in 1..9 present
    """
    h, w = _grid_size(grid)
    preds: Set[str] = set()
    preds.add(f"size:{h}x{w}")
    bb_h, bb_w = _nonzero_bbox(grid)
    preds.add(f"bbox:{bb_h}x{bb_w}")
    present: Set[int] = set()
    for r in range(h):
        row = grid[r]
        for v in row:
            if 1 <= v <= 9:
                present.add(v)
    for c in sorted(present):
        preds.add(f"has_color:{c}")
    preds.add(f"num_colors:{len(present)}")
    return preds


def predicate_deltas(pre: Set[str], post: Set[str]) -> Set[str]:
    """Compute coarse deltas from pre to post predicate sets.

    - gained_color:C and lost_color:C
    - bbox_shrunk / bbox_grew
    - size_changed
    - num_colors_inc / num_colors_dec / num_colors_same
    """
    deltas: Set[str] = set()
    # colors
    pre_colors = {p.split(":")[1] for p in pre if p.startswith("has_color:")}
    post_colors = {p.split(":")[1] for p in post if p.startswith("has_color:")}
    for c in sorted(post_colors - pre_colors):
        deltas.add(f"gained_color:{c}")
    for c in sorted(pre_colors - post_colors):
        deltas.add(f"lost_color:{c}")

    # bbox
    def _bbox_hw(preds: Set[str]) -> Tuple[int, int]:
        for p in preds:
            if p.startswith("bbox:"):
                hw = p.split(":")[1]
                h_str, w_str = hw.split("x")
                return int(h_str), int(w_str)
        return (0, 0)

    pre_bb = _bbox_hw(pre)
    post_bb = _bbox_hw(post)
    if pre_bb != post_bb:
        if post_bb[0] * post_bb[1] < pre_bb[0] * pre_bb[1]:
            deltas.add("bbox_shrunk")
        elif post_bb[0] * post_bb[1] > pre_bb[0] * pre_bb[1]:
            deltas.add("bbox_grew")

    # size
    def _size_hw(preds: Set[str]) -> Tuple[int, int]:
        for p in preds:
            if p.startswith("size:"):
                hw = p.split(":")[1]
                h_str, w_str = hw.split("x")
                return int(h_str), int(w_str)
        return (0, 0)

    if _size_hw(pre) != _size_hw(post):
        deltas.add("size_changed")

    # num_colors
    def _num_colors(preds: Set[str]) -> Optional[int]:
        for p in preds:
            if p.startswith("num_colors:"):
                return int(p.split(":")[1])
        return None

    pre_n = _num_colors(pre)
    post_n = _num_colors(post)
    if pre_n is not None and post_n is not None:
        if post_n > pre_n:
            deltas.add("num_colors_inc")
        elif post_n < pre_n:
            deltas.add("num_colors_dec")
        else:
            deltas.add("num_colors_same")

    return deltas


def desired_deltas(current: Grid, target: Grid) -> Set[str]:
    """Desired predicate changes to move from current toward target."""
    return predicate_deltas(extract_predicates(current), extract_predicates(target))


# ------------------------------
# Knowledge Graph data structure
# ------------------------------


@dataclass
class EdgeStats:
    count: int = 0


@dataclass
class KnowledgeGraph:
    """A lightweight property graph for behaviors and predicates.

    Nodes are represented implicitly by their string identifiers. Edge types:
      - enables: predicate -> operator
      - achieves: operator -> delta_predicate
      - follows: operator -> operator
    Each edge stores a simple count (frequency).
    """

    enables: Dict[Tuple[str, str], EdgeStats] = field(default_factory=dict)
    achieves: Dict[Tuple[str, str], EdgeStats] = field(default_factory=dict)
    follows: Dict[Tuple[str, str], EdgeStats] = field(default_factory=dict)

    def inc(self, table: Dict[Tuple[str, str], EdgeStats], src: str, dst: str, by: int = 1) -> None:
        key = (src, dst)
        if key not in table:
            table[key] = EdgeStats(0)
        table[key].count += by

    # Public API to record events
    def record_enables(self, predicate: str, op_sig: str, by: int = 1) -> None:
        self.inc(self.enables, predicate, op_sig, by)

    def record_achieves(self, op_sig: str, delta: str, by: int = 1) -> None:
        self.inc(self.achieves, op_sig, delta, by)

    def record_follows(self, prev_op_sig: str, op_sig: str, by: int = 1) -> None:
        self.inc(self.follows, prev_op_sig, op_sig, by)

    # Query helpers
    def get_enables(self, predicate: str) -> Dict[str, int]:
        return {op: stats.count for (pred, op), stats in self.enables.items() if pred == predicate}

    def get_achieves(self, op_sig: str) -> Dict[str, int]:
        return {delta: stats.count for (op, delta), stats in self.achieves.items() if op == op_sig}

    def get_follows(self, op_sig: str) -> Dict[str, int]:
        return {op2: stats.count for (op1, op2), stats in self.follows.items() if op1 == op_sig}

    # Serialization helpers
    def to_dict(self) -> Dict[str, List[Tuple[str, str, int]]]:
        def dump_table(table: Dict[Tuple[str, str], EdgeStats]) -> List[Tuple[str, str, int]]:
            return [(a, b, s.count) for (a, b), s in table.items()]

        return {
            "enables": dump_table(self.enables),
            "achieves": dump_table(self.achieves),
            "follows": dump_table(self.follows),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[Tuple[str, str, int]]]) -> "KnowledgeGraph":
        def load_table(items: List[Tuple[str, str, int]]) -> Dict[Tuple[str, str], EdgeStats]:
            table: Dict[Tuple[str, str], EdgeStats] = {}
            for a, b, c in items:
                table[(a, b)] = EdgeStats(int(c))
            return table

        kg = cls()
        kg.enables = load_table(data.get("enables", []))
        kg.achieves = load_table(data.get("achieves", []))
        kg.follows = load_table(data.get("follows", []))
        return kg

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # Stats
    def num_edges(self) -> Dict[str, int]:
        return {
            "enables": len(self.enables),
            "achieves": len(self.achieves),
            "follows": len(self.follows),
        }


# ------------------------------
# Behavior logging and guidance
# ------------------------------


@dataclass
class BehaviorLogger:
    graph: KnowledgeGraph
    last_op_sig: Optional[str] = None

    def log_step(self, current: Grid, op_sig: str, post: Grid) -> None:
        pre_preds = extract_predicates(current)
        post_preds = extract_predicates(post)
        deltas = predicate_deltas(pre_preds, post_preds)
        # enables: each pre predicate enables using op
        for p in pre_preds:
            self.graph.record_enables(p, op_sig, 1)
        # achieves: op leads to these deltas
        for d in deltas:
            self.graph.record_achieves(op_sig, d, 1)
        # follows: temporal transition
        if self.last_op_sig is not None:
            self.graph.record_follows(self.last_op_sig, op_sig, 1)
        self.last_op_sig = op_sig

    def reset_episode(self) -> None:
        self.last_op_sig = None


class KGScorer:
    """Scores operations based on KG statistics and a current/target state."""

    def __init__(self, kg: KnowledgeGraph, alpha_enables: float = 1.0, beta_achieves: float = 1.0):
        self.kg = kg
        self.alpha_enables = alpha_enables
        self.beta_achieves = beta_achieves

    def score(self, current: Grid, target: Optional[Grid], op_sig: str) -> float:
        pre = extract_predicates(current)
        enables_score = 0.0
        for p in pre:
            enables_counts = self.kg.get_enables(p)
            enables_score += float(enables_counts.get(op_sig, 0))

        achieves_score = 0.0
        if target is not None:
            wanted = desired_deltas(current, target)
            achieves_counts = self.kg.get_achieves(op_sig)
            for d in wanted:
                achieves_score += float(achieves_counts.get(d, 0))

        return self.alpha_enables * enables_score + self.beta_achieves * achieves_score


class KGGuidance:
    """Adapter to plug KG scoring into the solver's guidance interface."""

    def __init__(self, kg_scorer: KGScorer):
        self.kg_scorer = kg_scorer

    def rank_operations(self, base_program, train_pairs: Sequence[Tuple[Grid, Grid]], ops: List) -> List:
        scored: List[Tuple[float, object]] = []
        for op in ops:
            total = 0.0
            for inp, target in train_pairs:
                current = base_program.apply(inp)
                total += self.kg_scorer.score(current=current, target=target, op_sig=op.signature())
            avg = total / max(1, len(train_pairs))
            scored.append((avg, op))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [op for _, op in scored]



