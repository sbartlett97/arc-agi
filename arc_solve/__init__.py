"""Minimal neuro-symbolic ARC solver scaffold.

Modules:
- dsl: grid operations and program representation
- search: simple beam search over small DSL
- io: ARC dataset loader utilities
- cli: command-line entrypoint
"""

from .dsl import Grid, Operation, Program
from .kg import KnowledgeGraph, BehaviorLogger
from .search import BeamSearchSolver
from .io import load_task

__all__ = [
    "Grid",
    "Operation",
    "Program",
    "BeamSearchSolver",
    "load_task",
    "KnowledgeGraph",
    "BehaviorLogger",
]


