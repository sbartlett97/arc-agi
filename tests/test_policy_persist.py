import os
import tempfile
import unittest

from scripts.run_sample import run_sample


class TestPolicyPersistence(unittest.TestCase):
    def test_save_and_load_policy(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "policy.json")
            summary_train = run_sample(
                challenges_path="arc-prize-2025/arc-agi_training_challenges.json",
                solutions_path="arc-prize-2025/arc-agi_training_solutions.json",
                sample_size=5,
                seed=2,
                max_depth=1,
                beam=8,
                use_policy=True,
                policy_rollout_steps=1,
                policy_epochs=1,
                policy_lr=0.5,
                policy_path=path,
                save_policy=True,
            )
            self.assertTrue(summary_train["policy"]["used"])
            self.assertTrue(summary_train["policy"]["saved"])
            # Now load
            summary_load = run_sample(
                challenges_path="arc-prize-2025/arc-agi_training_challenges.json",
                solutions_path="arc-prize-2025/arc-agi_training_solutions.json",
                sample_size=3,
                seed=3,
                max_depth=1,
                beam=8,
                use_policy=True,
                load_policy=True,
                policy_path=path,
            )
            self.assertTrue(summary_load["policy"]["used"])
            self.assertTrue(summary_load["policy"]["loaded"])


if __name__ == "__main__":
    unittest.main()


