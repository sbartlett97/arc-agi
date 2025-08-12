import unittest

from arc_solve.dsl import (
    CropBoundingBox,
    Program,
    TransposeOp,
    crop_to_bbox,
    grids_equal,
)
from arc_solve.search import BeamSearchSolver


class TestSearchSmall(unittest.TestCase):
    def test_find_crop_bbox(self):
        # Train pairs describe cropping non-zero bbox
        train = [
            (
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [1, 1],
                    [1, 1],
                ],
            ),
            (
                [
                    [0, 0, 2],
                    [0, 2, 2],
                    [0, 0, 0],
                ],
                [
                    [0, 2],
                    [2, 2],
                ],
            ),
        ]

        solver = BeamSearchSolver(max_depth=2, beam_width=32)
        prog = solver.fit(train)
        self.assertTrue(isinstance(prog, Program))
        # apply to a test example
        test_inp = [
            [0, 0, 3, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 0],
        ]
        # Accept either crop-bbox or crop-bbox+transpose (training pairs
        # cannot disambiguate orientation)
        pred = prog.apply(test_inp)
        cropped = crop_to_bbox(test_inp)
        transposed = TransposeOp().apply(cropped)
        self.assertTrue(
            grids_equal(pred, cropped) or grids_equal(pred, transposed),
            f"Unexpected output: {pred}",
        )

    def test_identity(self):
        train = [
            (
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [1, 2],
                    [3, 4],
                ],
            )
        ]
        solver = BeamSearchSolver(max_depth=1, beam_width=8)
        prog = solver.fit(train)
        self.assertTrue(grids_equal(prog.apply([[9]]), [[9]]))


if __name__ == "__main__":
    unittest.main()


