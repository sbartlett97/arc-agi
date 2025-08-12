from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# Grid represented as list of rows, each row a list of ints (colors 0..9)
Grid = List[List[int]]


def grid_size(grid: Grid) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def grids_equal(a: Grid, b: Grid) -> bool:
    ah, aw = grid_size(a)
    bh, bw = grid_size(b)
    if ah != bh or aw != bw:
        return False
    for r in range(ah):
        if a[r] != b[r]:
            return False
    return True


def crop_to_bbox(grid: Grid) -> Grid:
    h, w = grid_size(grid)
    if h == 0 or w == 0:
        return []
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
        # all zeros → return empty grid with size 0x0 to signal no content
        return []
    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]


def transpose(grid: Grid) -> Grid:
    h, w = grid_size(grid)
    if h == 0 or w == 0:
        return []
    return [[grid[r][c] for r in range(h)] for c in range(w)]


def reflect_horizontal(grid: Grid) -> Grid:
    # Flip left-right
    return [list(reversed(row)) for row in grid]


def reflect_vertical(grid: Grid) -> Grid:
    # Flip top-bottom
    return list(reversed(grid))


def rotate90(grid: Grid) -> Grid:
    # Rotate 90 degrees clockwise
    t = transpose(grid)
    return reflect_horizontal(t)


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def largest_component(grid: Grid, treat_zero_as_background: bool = True) -> Grid:
    """Keep only the largest 4-connected component according to a binary mask.

    By default, zero is background: mask cell is True if value != 0.
    The kept cells retain their original color; all others become 0.
    """
    h, w = grid_size(grid)
    if h == 0 or w == 0:
        return []
    visited = [[False] * w for _ in range(h)]

    def in_bounds(rr: int, cc: int) -> bool:
        return 0 <= rr < h and 0 <= cc < w

    def is_foreground(rr: int, cc: int) -> bool:
        v = grid[rr][cc]
        if treat_zero_as_background:
            return v != 0
        return True

    components: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r][c] or not is_foreground(r, c):
                continue
            stack = [(r, c)]
            visited[r][c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if in_bounds(nr, nc) and not visited[nr][nc] and is_foreground(nr, nc):
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            components.append(comp)

    if not components:
        return [[0 for _ in range(w)] for _ in range(h)]

    largest = max(components, key=len)
    out = [[0 for _ in range(w)] for _ in range(h)]
    for rr, cc in largest:
        out[rr][cc] = grid[rr][cc]
    return out


def keep_color(grid: Grid, color: int) -> Grid:
    return [[cell if cell == color else 0 for cell in row] for row in grid]


def remove_color(grid: Grid, color: int) -> Grid:
    return [[0 if cell == color else cell for cell in row] for row in grid]


def replace_color(grid: Grid, from_color: int, to_color: int) -> Grid:
    if from_color == to_color:
        return copy_grid(grid)
    return [[(to_color if cell == from_color else cell) for cell in row] for row in grid]


class Operation:
    """Abstract operation on a grid."""

    def apply(self, grid: Grid) -> Grid:  # pragma: no cover - interface
        raise NotImplementedError

    def signature(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class Identity(Operation):
    def apply(self, grid: Grid) -> Grid:
        return copy_grid(grid)


@dataclass(frozen=True)
class CropBoundingBox(Operation):
    def apply(self, grid: Grid) -> Grid:
        return crop_to_bbox(grid)


@dataclass(frozen=True)
class LargestComponent(Operation):
    treat_zero_as_background: bool = True

    def apply(self, grid: Grid) -> Grid:
        return largest_component(grid, self.treat_zero_as_background)


@dataclass(frozen=True)
class KeepColor(Operation):
    color: int

    def apply(self, grid: Grid) -> Grid:
        return keep_color(grid, self.color)


@dataclass(frozen=True)
class RemoveColor(Operation):
    color: int

    def apply(self, grid: Grid) -> Grid:
        return remove_color(grid, self.color)


@dataclass(frozen=True)
class ReplaceColor(Operation):
    from_color: int
    to_color: int

    def apply(self, grid: Grid) -> Grid:
        return replace_color(grid, self.from_color, self.to_color)


@dataclass(frozen=True)
class ReflectH(Operation):
    def apply(self, grid: Grid) -> Grid:
        return reflect_horizontal(grid)


@dataclass(frozen=True)
class ReflectV(Operation):
    def apply(self, grid: Grid) -> Grid:
        return reflect_vertical(grid)


@dataclass(frozen=True)
class TransposeOp(Operation):
    def apply(self, grid: Grid) -> Grid:
        return transpose(grid)


@dataclass(frozen=True)
class Rotate90(Operation):
    def apply(self, grid: Grid) -> Grid:
        return rotate90(grid)


@dataclass
class Program:
    steps: List[Operation]

    def apply(self, grid: Grid) -> Grid:
        out = grid
        for op in self.steps:
            out = op.apply(out)
        return out

    def signature(self) -> str:
        return "|".join(op.signature() for op in self.steps)

    def __len__(self) -> int:
        return len(self.steps)


