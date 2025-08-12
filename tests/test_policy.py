import unittest

import numpy as np

from arc_solve.dsl import Program, crop_to_bbox, grids_equal
from arc_solve.learn import LogisticPolicy, basic_features, build_imitation_dataset
from arc_solve.search import BeamSearchSolver, PolicyGuidance


class TestPolicyGuidance(unittest.TestCase):
    def test_guided_crop(self):
        # Simple tasks that are solved by crop_to_bbox
        tasks = [
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

        X, y, op_sigs = build_imitation_dataset(tasks, max_rollout_steps=1)
        policy = LogisticPolicy.init(dim=X.shape[1], op_signatures=op_sigs)
        policy.fit(X, y, lr=0.5, epochs=10)

        def score_fn(inp, target, current, op):
            x = basic_features(inp, target, current)[None, :]
            logp = policy.predict_log_proba(x)[0]
            # return log-prob of this op
            idx = op_sigs.index(op.signature())
            return logp[idx]

        solver = BeamSearchSolver(max_depth=2, beam_width=16, guidance=PolicyGuidance(score_fn))
        prog = solver.fit(tasks)
        out = prog.apply([
            [0, 0, 3, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 0],
        ])
        self.assertTrue(grids_equal(out, crop_to_bbox([
            [0, 0, 3, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 0],
        ])))


if __name__ == "__main__":
    unittest.main()


