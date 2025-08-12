import unittest

from arc_solve.dsl import (
    CropBoundingBox,
    Identity,
    KeepColor,
    LargestComponent,
    Program,
    ReflectH,
    ReflectV,
    TransposeOp,
    Rotate90,
    crop_to_bbox,
    grids_equal,
)


class TestDSL(unittest.TestCase):
    def test_crop_bbox(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        expected = [
            [1, 1],
            [1, 1],
        ]
        self.assertTrue(grids_equal(crop_to_bbox(grid), expected))

    def test_largest_component(self):
        grid = [
            [0, 2, 2, 0],
            [0, 2, 0, 0],
            [3, 0, 0, 3],
            [3, 0, 0, 3],
        ]
        op = LargestComponent(True)
        out = op.apply(grid)
        # largest is the 2-cluster of size 3
        expected = [
            [0, 2, 2, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        self.assertTrue(grids_equal(out, expected))

    def test_reflections_transpose_rotate(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        self.assertEqual(ReflectH().apply(grid), [[2, 1], [4, 3]])
        self.assertEqual(ReflectV().apply(grid), [[3, 4], [1, 2]])
        self.assertEqual(TransposeOp().apply(grid), [[1, 3], [2, 4]])
        self.assertEqual(Rotate90().apply(grid), [[3, 1], [4, 2]])

    def test_keep_color(self):
        grid = [
            [1, 2, 0],
            [2, 1, 2],
        ]
        kept = KeepColor(2).apply(grid)
        self.assertEqual(kept, [[0, 2, 0], [2, 0, 2]])


if __name__ == "__main__":
    unittest.main()


