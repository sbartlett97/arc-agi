from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .dsl import (
    CropBoundingBox,
    Grid,
    Identity,
    KeepColor,
    LargestComponent,
    Program,
    ReflectH,
    ReflectV,
    ReplaceColor,
    Rotate90,
    TransposeOp,
    grids_equal,
)


TrainPair = Tuple[Grid, Grid]


def pixel_mismatch(a: Grid, b: Grid) -> int:
    if len(a) == 0 and len(b) == 0:
        return 0
    if len(a) != len(b) or (a and b and len(a[0]) != len(b[0])):
        # penalize size mismatch heavily
        ah, aw = (len(a), len(a[0]) if a else 0)
        bh, bw = (len(b), len(b[0]) if b else 0)
        return 10_000 + abs(ah - bh) * 100 + abs(aw - bw) * 100
    mismatches = 0
    for r in range(len(a)):
        row_a = a[r]
        row_b = b[r]
        for c in range(len(row_a)):
            if row_a[c] != row_b[c]:
                mismatches += 1
    return mismatches


def enumerate_operations() -> List:
    ops = [
        Identity(),
        CropBoundingBox(),
        LargestComponent(True),
        ReflectH(),
        ReflectV(),
        TransposeOp(),
        Rotate90(),
    ]
    # KeepColor for colors 0..9
    for color in range(10):
        ops.append(KeepColor(color))
    # ReplaceColor limited for tractability: only map 0..5 to 0..5 initially
    for fr in range(6):
        for to in range(6):
            if fr == to:
                continue
            ops.append(ReplaceColor(fr, to))
    return ops


@dataclass
class ScoredProgram:
    loss: int
    length: int
    idx: int
    program: Program

    def __lt__(self, other: "ScoredProgram") -> bool:
        # heap ordered by loss, then by shorter length, then insertion order
        if self.loss != other.loss:
            return self.loss < other.loss
        if self.length != other.length:
            return self.length < other.length
        return self.idx < other.idx


class BeamSearchSolver:
    def __init__(self, max_depth: int = 2, beam_width: int = 64, guidance: Optional["PolicyGuidance"] = None, behavior_logger: Optional[object] = None):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.ops_cache = enumerate_operations()
        self.guidance = guidance
        # Optional behavior logger with interface: reset_episode(), log_step(current, op_sig, post)
        self.behavior_logger = behavior_logger

    def _score_program(self, program: Program, train_pairs: Sequence[TrainPair]) -> int:
        total = 0
        for inp, out in train_pairs:
            pred = program.apply(inp)
            total += pixel_mismatch(pred, out)
            if total > 1_000_000:
                # early stop bad candidates
                return total
        return total

    def fit(self, train_pairs: Sequence[TrainPair]) -> Program:
        # Exact match early exit: identity
        identity = Program([])
        if all(grids_equal(inp, out) for inp, out in train_pairs):
            return identity

        beam: List[ScoredProgram] = []
        seen_signatures = set()
        next_idx = 0

        # seed beam with length-0
        loss0 = self._score_program(identity, train_pairs)
        heapq.heappush(beam, ScoredProgram(loss0, 0, next_idx, identity))
        seen_signatures.add(identity.signature())
        next_idx += 1

        best = beam[0]

        for depth in range(1, self.max_depth + 1):
            # expand current beam
            candidates: List[ScoredProgram] = []
            base_programs = [sp.program for sp in beam]
            for base in base_programs:
                ops_to_try = self.ops_cache
                if self.guidance is not None:
                    ops_to_try = self.guidance.rank_operations(base, train_pairs, self.ops_cache)
                for op in ops_to_try:
                    new_prog = Program(base.steps + [op])
                    sig = new_prog.signature()
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)
                    loss = self._score_program(new_prog, train_pairs)
                    sp = ScoredProgram(loss, len(new_prog), next_idx, new_prog)
                    next_idx += 1
                    candidates.append(sp)

                    # Log behavior for each train pair using the incremental step just applied
                    if self.behavior_logger is not None:
                        # Reset episode when extending from length-0 to length-1
                        if len(base) == 0:
                            try:
                                self.behavior_logger.reset_episode()
                            except Exception:
                                pass
                        try:
                            for inp, _target in train_pairs:
                                current = base.apply(inp)
                                post = op.apply(current)
                                self.behavior_logger.log_step(current, op.signature(), post)
                        except Exception:
                            # Never fail search due to logging
                            pass

            if not candidates:
                break

            # keep top beam_width
            candidates.sort()
            beam = candidates[: self.beam_width]

            # check perfect match
            if beam and beam[0].loss == 0:
                return beam[0].program

            # track best seen
            if beam and beam[0] < best:
                best = beam[0]

        return best.program


class PolicyGuidance:
    """Ranks operations based on a learned policy over simple features.

    The policy is expected to implement `score_ops(inp, target, current, ops)`
    and return a list of operations sorted by descending score.
    """

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def rank_operations(self, base: Program, train_pairs: Sequence[TrainPair], ops: List) -> List:
        # Score each operation by avg policy log-prob across train pairs given current state
        scored: List[Tuple[float, object]] = []
        for op in ops:
            total = 0.0
            for inp, target in train_pairs:
                current = base.apply(inp)
                total += float(self.score_fn(inp, target, current, op))
            avg = total / max(1, len(train_pairs))
            scored.append((avg, op))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [op for _, op in scored]


