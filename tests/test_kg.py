import unittest

from arc_solve.dsl import KeepColor, Program
from arc_solve.kg import KnowledgeGraph, BehaviorLogger, KGScorer, KGGuidance
from arc_solve.search import BeamSearchSolver


class TestKnowledgeGraph(unittest.TestCase):
    def test_logging_and_queries(self):
        kg = KnowledgeGraph()
        logger = BehaviorLogger(kg)
        inp = [
            [1, 2, 0],
            [2, 1, 2],
        ]
        current = inp
        op = KeepColor(2)
        post = op.apply(current)
        logger.reset_episode()
        logger.log_step(current, op.signature(), post)

        # enables should include size and has_color predicates → op signature present
        enables = kg.get_enables("size:2x3")
        self.assertIn(op.signature(), enables)
        # achieves should include lost_color:1 if color 1 removed from post
        achieves = kg.get_achieves(op.signature())
        self.assertTrue(any(k.startswith("lost_color:") for k in achieves))

    def test_guidance_improves_ordering(self):
        # Train pairs where keeping a specific color is helpful
        train = [
            (
                [
                    [1, 2, 0],
                    [2, 1, 2],
                ],
                [
                    [0, 2, 0],
                    [2, 0, 2],
                ],
            )
        ]
        kg = KnowledgeGraph()
        logger = BehaviorLogger(kg)
        solver = BeamSearchSolver(max_depth=1, beam_width=8, behavior_logger=logger)
        # First run populates KG via logging during search
        _ = solver.fit(train)

        # Now use KGGuidance to rank operations; KeepColor(2) should be prioritized
        kg_guidance = KGGuidance(KGScorer(kg))
        ranked = kg_guidance.rank_operations(Program([]), train, solver.ops_cache)
        sigs = [op.signature() for op in ranked[:10]]
        self.assertIn(KeepColor(2).signature(), sigs)


if __name__ == "__main__":
    unittest.main()


