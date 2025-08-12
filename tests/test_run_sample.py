import json
import unittest

from scripts.run_sample import run_sample


class TestRunSample(unittest.TestCase):
    def test_small_sample(self):
        # Use a tiny sample to keep CI fast
        summary = run_sample(
            challenges_path="arc-prize-2025/arc-agi_training_challenges.json",
            solutions_path="arc-prize-2025/arc-agi_training_solutions.json",
            sample_size=3,
            seed=1,
            max_depth=1,
            beam=8,
            use_policy=False,
        )
        self.assertIn("num_tasks", summary)
        self.assertEqual(summary["num_tasks"], 3)
        self.assertIn("test_accuracy", summary)


if __name__ == "__main__":
    unittest.main()


