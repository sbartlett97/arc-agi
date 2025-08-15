#!/usr/bin/env python3
"""
Main script for ARC-AGI expert-based solution
"""

import torch
import numpy as np
import argparse
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import time

from arc_dataset import ARCDataLoader, ARCDataset
from arc_expert_system import ExpertManager, Expert
from arc_vit import create_arc_model


class ARCAGISolver:
    """Main solver for ARC-AGI challenges"""
    
    def __init__(self, 
                 experts_dir: str = "experts",
                 data_dir: str = "arc-prize-2025",
                 device: str = "auto"):
        self.experts_dir = experts_dir
        self.data_dir = data_dir
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.data_loader = ARCDataLoader(data_dir)
        self.expert_manager = ExpertManager(experts_dir=experts_dir, device=self.device)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'expert_reuse': 0,
            'new_experts_created': 0,
            'average_accuracy': 0.0,
            'total_time': 0.0
        }
    
    def train_on_dataset(self, 
                        dataset: ARCDataset,
                        max_tasks: int = None,
                        save_interval: int = 10) -> Dict:
        """Train experts on a dataset"""
        print(f"Training on dataset with {len(dataset)} tasks")
        
        if max_tasks:
            dataset_size = min(max_tasks, len(dataset))
        else:
            dataset_size = len(dataset)
        
        start_time = time.time()
        accuracies = []
        
        for i in tqdm(range(dataset_size), desc="Training experts"):
            try:
                # Get task
                test_sample, support_samples, test_solution = dataset[i]
                
                # Solve task
                solution, expert = self.expert_manager.solve_task(
                    test_sample, support_samples, f"task_{i}"
                )
                
                # Calculate accuracy
                if solution.shape == test_solution.shape:
                    accuracy = (solution == test_solution).mean()
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0.0)
                
                # Update statistics
                if hasattr(expert, 'id') and expert.id.startswith('expert_'):
                    self.stats['new_experts_created'] += 1
                else:
                    self.stats['expert_reuse'] += 1
                
                # Save experts periodically
                if (i + 1) % save_interval == 0:
                    self.expert_manager.merge_similar_experts()
                    print(f"Processed {i + 1}/{dataset_size} tasks, "
                          f"Accuracy: {np.mean(accuracies[-save_interval:]):.3f}")
                
            except Exception as e:
                print(f"Error processing task {i}: {e}")
                accuracies.append(0.0)
        
        # Final statistics
        self.stats['total_tasks'] = dataset_size
        self.stats['average_accuracy'] = np.mean(accuracies)
        self.stats['total_time'] = time.time() - start_time
        
        print(f"\nTraining completed!")
        print(f"Total tasks: {self.stats['total_tasks']}")
        print(f"New experts created: {self.stats['new_experts_created']}")
        print(f"Expert reuse: {self.stats['expert_reuse']}")
        print(f"Average accuracy: {self.stats['average_accuracy']:.3f}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        
        return self.stats
    
    def evaluate_on_dataset(self, 
                           dataset: ARCDataset,
                           max_tasks: int = None) -> Dict:
        """Evaluate performance on a dataset"""
        print(f"Evaluating on dataset with {len(dataset)} tasks")
        
        if max_tasks:
            dataset_size = min(max_tasks, len(dataset))
        else:
            dataset_size = len(dataset)
        
        accuracies = []
        expert_usage = {}
        
        for i in tqdm(range(dataset_size), desc="Evaluating"):
            try:
                # Get task
                test_sample, support_samples, test_solution = dataset[i]
                
                # Solve task
                solution, expert = self.expert_manager.solve_task(
                    test_sample, support_samples, f"eval_task_{i}"
                )
                
                # Track expert usage
                expert_id = expert.id
                if expert_id not in expert_usage:
                    expert_usage[expert_id] = 0
                expert_usage[expert_id] += 1
                
                # Calculate accuracy
                if solution.shape == test_solution.shape:
                    accuracy = (solution == test_solution).mean()
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0.0)
                
            except Exception as e:
                print(f"Error evaluating task {i}: {e}")
                accuracies.append(0.0)
        
        # Evaluation statistics
        eval_stats = {
            'total_tasks': dataset_size,
            'average_accuracy': np.mean(accuracies),
            'expert_usage': expert_usage,
            'num_experts_used': len(expert_usage)
        }
        
        print(f"\nEvaluation completed!")
        print(f"Total tasks: {eval_stats['total_tasks']}")
        print(f"Average accuracy: {eval_stats['average_accuracy']:.3f}")
        print(f"Number of experts used: {eval_stats['num_experts_used']}")
        
        return eval_stats
    
    def solve_single_task(self, 
                          test_sample: np.ndarray,
                          support_samples: List[Dict]) -> Tuple[np.ndarray, Expert]:
        """Solve a single ARC task"""
        return self.expert_manager.solve_task(
            test_sample, support_samples, "single_task"
        )
    
    def save_experts(self):
        """Save all experts to disk"""
        print("Saving experts...")
        for expert in self.expert_manager.experts:
            self.expert_manager.save_expert(expert)
        print(f"Saved {len(self.expert_manager.experts)} experts")
    
    def load_experts(self):
        """Load experts from disk"""
        print("Loading experts...")
        self.expert_manager.load_experts()
        print(f"Loaded {len(self.expert_manager.experts)} experts")
    
    def get_expert_summary(self) -> Dict:
        """Get summary of all experts"""
        summary = {
            'total_experts': len(self.expert_manager.experts),
            'experts': []
        }
        
        for expert in self.expert_manager.experts:
            expert_info = {
                'id': expert.id,
                'performance_score': expert.performance_score,
                'num_training_samples': expert.num_training_samples,
                'task_signatures': expert.task_signatures,
                'created_at': expert.created_at
            }
            summary['experts'].append(expert_info)
        
        return summary
    
    def check_device_placement(self):
        """Check if all models are on the correct device"""
        print(f"Checking device placement for {len(self.expert_manager.experts)} experts...")
        for i, expert in enumerate(self.expert_manager.experts):
            model_device = next(expert.model.parameters()).device
            print(f"Expert {i}: {expert.id} -> Device: {model_device}")
            if model_device != self.device:
                print(f"  WARNING: Expert {expert.id} is on {model_device}, expected {self.device}")
                # Move to correct device
                expert.model.to(self.device)
                print(f"  Moved to {self.device}")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Expert-Based Solver")
    parser.add_argument("--mode", choices=["train", "evaluate", "solve"], 
                       default="train", help="Operation mode")
    parser.add_argument("--data_dir", default="arc-prize-2025", 
                       help="Directory containing ARC data")
    parser.add_argument("--experts_dir", default="experts", 
                       help="Directory to store/load experts")
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="Maximum number of tasks to process")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save experts every N tasks during training")
    
    args = parser.parse_args()
    
    # Show device information
    print("=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Selected device: {args.device}")
    print()
    
    # Initialize solver
    solver = ARCAGISolver(
        experts_dir=args.experts_dir,
        data_dir=args.data_dir,
        device=args.device
    )
    
    if args.mode == "train":
        # Load training data
        try:
            train_dataset = solver.data_loader.load_training_data()
            print(f"Training dataset loaded: {len(train_dataset)} tasks")
            
            # Train experts
            stats = solver.train_on_dataset(
                train_dataset, 
                max_tasks=args.max_tasks,
                save_interval=args.save_interval
            )
            
            # Save final experts
            solver.save_experts()
            
            # Check device placement
            solver.check_device_placement()
            
            # Save training statistics
            with open("training_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
                
        except FileNotFoundError:
            print("Training data not found. Please check the data directory.")
            return
    
    elif args.mode == "evaluate":
        # Load test/evaluation data
        try:
            test_dataset = solver.data_loader.load_evaluation_data()
            print(f"Test dataset loaded: {len(test_dataset)} tasks")
            
            # Evaluate
            eval_stats = solver.evaluate_on_dataset(test_dataset, max_tasks=args.max_tasks)
            
            # Check device placement
            solver.check_device_placement()
            
            # Save evaluation statistics
            with open("evaluation_stats.json", "w") as f:
                json.dump(eval_stats, f, indent=2)
                
        except FileNotFoundError:
            print("Test data not found. Please check the data directory.")
            return
    
    elif args.mode == "solve":
        # Interactive mode for solving single tasks
        print("Interactive ARC task solver")
        print("Enter grid values (space-separated, empty line to finish):")
        
        # Get test sample
        print("\nEnter test sample grid:")
        test_sample = []
        while True:
            line = input().strip()
            if not line:
                break
            row = [int(x) for x in line.split()]
            test_sample.append(row)
        
        if not test_sample:
            print("No test sample provided")
            return
        
        # Convert to numpy array and pad
        test_sample = np.array(test_sample)
        if test_sample.shape[0] > 30 or test_sample.shape[1] > 30:
            print("Grid too large (max 30x30)")
            return
        
        # Pad to 30x30
        padded_test = np.zeros((30, 30), dtype=np.int64)
        padded_test[:test_sample.shape[0], :test_sample.shape[1]] = test_sample
        
        # Get support samples
        support_samples = []
        print("\nEnter support samples (input then output, empty line between samples):")
        
        while True:
            print("\nSupport sample input (empty line to finish):")
            input_grid = []
            while True:
                line = input().strip()
                if not line:
                    break
                row = [int(x) for x in line.split()]
                input_grid.append(row)
            
            if not input_grid:
                break
            
            print("Support sample output:")
            output_grid = []
            while True:
                line = input().strip()
                if not line:
                    break
                row = [int(x) for x in line.split()]
                output_grid.append(row)
            
            if input_grid and output_grid:
                # Pad grids
                input_padded = np.zeros((30, 30), dtype=np.int64)
                output_padded = np.zeros((30, 30), dtype=np.int64)
                
                input_padded[:len(input_grid), :len(input_grid[0])] = input_grid
                output_padded[:len(output_grid), :len(output_grid[0])] = output_grid
                
                support_samples.append({
                    'input': input_padded,
                    'output': output_padded
                })
        
        if not support_samples:
            print("No support samples provided")
            return
        
        # Solve task
        print("\nSolving task...")
        solution, expert = solver.solve_single_task(padded_test, support_samples)
        
        # Check device placement
        solver.check_device_placement()
        
        print(f"\nSolution found using expert: {expert.id}")
        print("Solution grid:")
        
        # Find actual grid size (remove padding)
        non_zero_rows = np.any(solution != 0, axis=1)
        non_zero_cols = np.any(solution != 0, axis=0)
        
        if np.any(non_zero_rows) and np.any(non_zero_cols):
            max_row = np.max(np.where(non_zero_rows)[0]) + 1
            max_col = np.max(np.where(non_zero_cols)[0]) + 1
            
            for i in range(max_row):
                row_str = " ".join(str(int(x)) for x in solution[i, :max_col])
                print(row_str)
        else:
            print("No solution found")


if __name__ == "__main__":
    main()
