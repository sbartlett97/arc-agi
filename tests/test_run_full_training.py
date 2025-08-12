import json
import os
import tempfile
import unittest

from scripts.run_full_training import run_full_training


class TestRunFullTraining(unittest.TestCase):
    def test_small_full_training_and_save_kg(self):
        with tempfile.TemporaryDirectory() as td:
            kg_path = os.path.join(td, "kg.json")
            summary = run_full_training(
                challenges_path="arc-prize-2025/arc-agi_training_challenges.json",
                solutions_path="arc-prize-2025/arc-agi_training_solutions.json",
                max_depth=1,
                beam=8,
                save_kg_path=kg_path,
            )
            self.assertIn("kg_edges", summary)
            self.assertTrue(os.path.exists(kg_path))
            data = json.load(open(kg_path, "r"))
            # Expect keys in KG dump
            self.assertTrue(all(k in data for k in ["enables", "achieves", "follows"]))


if __name__ == "__main__":
    unittest.main()


