import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import os


class ARCDataset(Dataset):
    """Dataset loader for ARC-AGI challenges"""
    
    def __init__(self, challenges_file: str, solutions_file: str, max_grid_size: int = 30):
        """
        Initialize ARC dataset
        
        Args:
            challenges_file: Path to challenges JSON file
            solutions_file: Path to solutions JSON file  
            max_grid_size: Maximum grid size for padding
        """
        self.max_grid_size = max_grid_size
        
        # Load challenges and solutions
        with open(challenges_file, 'r') as f:
            self.challenges = json.load(f)
        
        with open(solutions_file, 'r') as f:
            self.solutions = json.load(f)
        
        # Create task IDs list
        self.task_ids = list(self.challenges.keys())
        
    def __len__(self):
        return len(self.task_ids)
    
    def pad_grid(self, grid: List[List[int]], target_size: int = 30) -> np.ndarray:
        """Pad grid to target size with zeros"""
        if not grid:
            return np.zeros((target_size, target_size), dtype=np.int64)
        
        rows, cols = len(grid), len(grid[0]) if grid else 0
        padded = np.zeros((target_size, target_size), dtype=np.int64)
        
        # Copy original grid to padded array
        if rows > 0 and cols > 0:
            padded[:rows, :cols] = grid
            
        return padded
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Get a single task
        
        Returns:
            (test_sample, support_samples, test_solution)
        """
        task_id = self.task_ids[idx]
        challenge = self.challenges[task_id]
        solution = self.solutions[task_id]
        
        # Get test sample (first example)
        test_sample = self.pad_grid(challenge['test'][0]['input'])
        
        # Get support samples (training examples)
        support_samples = []
        for example in challenge['train']:
            support_samples.append({
                'input': self.pad_grid(example['input']),
                'output': self.pad_grid(example['output'])
            })
        
        # Get test solution - solutions is an array, first element corresponds to first test case
        test_solution = self.pad_grid(solution[0])
        
        return test_sample, support_samples, test_solution


class ARCDataLoader:
    """Utility class for loading ARC data in the required format"""
    
    def __init__(self, data_dir: str = "arc-prize-2025"):
        self.data_dir = data_dir
        
    def load_training_data(self) -> ARCDataset:
        """Load training dataset"""
        challenges_file = os.path.join(self.data_dir, "arc-agi_training_challenges.json")
        solutions_file = os.path.join(self.data_dir, "arc-agi_training_solutions.json")
        return ARCDataset(challenges_file, solutions_file)
    
    def load_test_data(self) -> ARCDataset:
        """Load test dataset"""
        challenges_file = os.path.join(self.data_dir, "arc-agi_test_challenges.json")
        solutions_file = os.path.join(self.data_dir, "arc-agi_evaluation_solutions.json")
        return ARCDataset(challenges_file, solutions_file)
    
    def load_evaluation_data(self) -> ARCDataset:
        """Load evaluation dataset"""
        challenges_file = os.path.join(self.data_dir, "arc-agi_evaluation_challenges.json")
        solutions_file = os.path.join(self.data_dir, "arc-agi_evaluation_solutions.json")
        return ARCDataset(challenges_file, solutions_file)


def create_sample_batch() -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """Create a sample batch for testing"""
    # Test sample: 2x2 grid
    test_sample = np.array([[3, 2], [7, 8]], dtype=np.int64)
    
    # Support samples
    support_samples = [
        {
            'input': np.array([[7, 9], [4, 3]], dtype=np.int64),
            'output': np.array([
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3],
                [9, 7, 9, 7, 9, 7],
                [3, 4, 3, 4, 3, 4],
                [7, 9, 7, 9, 7, 9],
                [4, 3, 4, 3, 4, 3]
            ], dtype=np.int64)
        }
    ]
    
    # Test solution: 6x6 grid
    test_solution = np.array([
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
        [2, 3, 2, 3, 2, 3],
        [8, 7, 8, 7, 8, 7],
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8]
    ], dtype=np.int64)
    
    # Pad all grids to 30x30
    test_sample = np.pad(test_sample, ((0, 28), (0, 28)), mode='constant')
    
    # Pad support samples
    for sample in support_samples:
        sample['input'] = np.pad(sample['input'], ((0, 28), (0, 28)), mode='constant')
        sample['output'] = np.pad(sample['output'], ((0, 24), (0, 24)), mode='constant')
    
    test_solution = np.pad(test_solution, ((0, 24), (0, 24)), mode='constant')
    
    return test_sample, support_samples, test_solution


if __name__ == "__main__":
    # Test the dataset loader
    loader = ARCDataLoader()
    
    try:
        train_dataset = loader.load_training_data()
        print(f"Training dataset loaded with {len(train_dataset)} tasks")
        
        # Test a sample
        test_sample, support_samples, test_solution = train_dataset[0]
        print(f"Test sample shape: {test_sample.shape}")
        print(f"Number of support samples: {len(support_samples)}")
        print(f"Test solution shape: {test_solution.shape}")
        
    except FileNotFoundError:
        print("Training data not found, testing with sample data")
        test_sample, support_samples, test_solution = create_sample_batch()
        print(f"Sample test shape: {test_sample.shape}")
        print(f"Sample solution shape: {test_solution.shape}")
