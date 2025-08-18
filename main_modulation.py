#!/usr/bin/env python3
"""
Main script for ARC-AGI modulation-based solution
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
from arc_modulation_system import ModulationSystem


class ARCModulationSolver:
    """Main solver for ARC-AGI challenges using modulation approach"""
    
    def __init__(self, 
                 model_dir: str = "modulation_models",
                 data_dir: str = "arc-prize-2025",
                 device: str = "auto"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.data_loader = ARCDataLoader(data_dir)
        self.modulation_system = ModulationSystem(model_dir=model_dir, device=self.device)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'tasks_solved_without_training': 0,
            'tasks_requiring_training': 0,
            'average_accuracy': 0.0,
            'total_time': 0.0,
            'training_time': 0.0
        }
    
    def train_on_dataset(self, 
                        dataset: ARCDataset,
                        max_tasks: int = None,
                        save_interval: int = 10,
                        train_threshold: float = 0.7,
                        use_merged_training: bool = True,
                        batch_size: int = 2,
                        gradient_accumulation: int = 4,
                        max_support_per_batch: int = 8) -> Dict:
        """Train the modulation model on a dataset"""
        print(f"Training on dataset with {len(dataset)} tasks")
        
        if max_tasks:
            dataset_size = min(max_tasks, len(dataset))
        else:
            dataset_size = len(dataset)
        
        start_time = time.time()
        training_start_time = time.time()
        accuracies = []
        
        if use_merged_training:
            print("Using merged training approach: Phase 1 (support samples) + Phase 2 (test data)")
            
            # Phase 1: Collect all support samples and train on them
            print("\nPhase 1: Collecting and training on all support samples...")
            all_support_samples = []
            test_inputs = []
            test_targets = []
            
            for i in range(dataset_size):
                try:
                    test_sample, support_samples, test_solution = dataset[i]
                    
                    # Add support samples to the collection
                    all_support_samples.extend(support_samples)
                    
                    # Store test data for phase 2
                    test_inputs.append(test_sample)
                    test_targets.append(test_solution)
                    
                except Exception as e:
                    print(f"Error collecting data from task {i}: {e}")
            
            print(f"Collected {len(all_support_samples)} support samples from {dataset_size} tasks")
            
            # Train on merged support samples first
            support_training_stats = self.modulation_system.train_with_merged_support_samples(
                all_support_samples=all_support_samples,
                test_inputs=test_inputs,
                test_targets=test_targets,
                support_epochs=5,  # Single epoch on support samples
                test_epochs=100,    # Multiple epochs on test data
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation,
                max_support_samples_per_batch=max_support_per_batch
            )
            
            # Evaluate on all tasks
            print("\nEvaluating on all tasks after merged training...")
            for i in tqdm(range(dataset_size), desc="Evaluating after training"):
                try:
                    test_sample, support_samples, test_solution = dataset[i]
                    
                    # Solve task (no additional training needed)
                    solution, accuracy = self.modulation_system.solve_and_train(
                        test_sample, support_samples, f"task_{i}", train_if_needed=False
                    )
                    
                    accuracies.append(accuracy)
                    
                    # Update statistics
                    if accuracy >= train_threshold:
                        self.stats['tasks_solved_without_training'] += 1
                    else:
                        self.stats['tasks_requiring_training'] += 1
                    
                except Exception as e:
                    print(f"Error evaluating task {i}: {e}")
                    accuracies.append(0.0)
            
        else:
            # Original approach: train on each task individually
            print("Using individual task training approach")
            
            for i in tqdm(range(dataset_size), desc="Training modulation model"):
                try:
                    # Get task
                    test_sample, support_samples, test_solution = dataset[i]
                    
                    # Solve task with training if needed
                    solution, accuracy = self.modulation_system.solve_and_train(
                        test_sample, support_samples, f"task_{i}", train_if_needed=True
                    )
                    
                    accuracies.append(accuracy)
                    
                    # Update statistics
                    if accuracy >= train_threshold:
                        self.stats['tasks_solved_without_training'] += 1
                    else:
                        self.stats['tasks_requiring_training'] += 1
                    
                    # Save model periodically
                    if (i + 1) % save_interval == 0:
                        self.modulation_system.save_model()
                        
                        # Get current performance stats
                        perf_stats = self.modulation_system.get_performance_stats()
                        current_avg = np.mean(accuracies[-save_interval:])
                        
                        print(f"Processed {i + 1}/{dataset_size} tasks")
                        print(f"  Recent accuracy: {current_avg:.3f}")
                        print(f"  Overall accuracy: {perf_stats['average_accuracy']:.3f}")
                        print(f"  Tasks above threshold: {perf_stats['tasks_above_threshold']}/{perf_stats['total_tasks']}")
                    
                except Exception as e:
                    print(f"Error processing task {i}: {e}")
                    accuracies.append(0.0)
        
        # Final statistics
        self.stats['total_tasks'] = dataset_size
        self.stats['average_accuracy'] = np.mean(accuracies)
        self.stats['total_time'] = time.time() - start_time
        self.stats['training_time'] = time.time() - training_start_time
        
        print(f"\nTraining completed!")
        print(f"Total tasks: {self.stats['total_tasks']}")
        print(f"Tasks solved without training: {self.stats['tasks_solved_without_training']}")
        print(f"Tasks requiring training: {self.stats['tasks_requiring_training']}")
        print(f"Average accuracy: {self.stats['average_accuracy']:.3f}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Training time: {self.stats['training_time']:.2f}s")
        
        return self.stats
    
    def evaluate_on_dataset(self, 
                           dataset: ARCDataset,
                           max_tasks: int = None,
                           train_threshold: float = 0.7) -> Dict:
        """Evaluate performance on a dataset"""
        print(f"Evaluating on dataset with {len(dataset)} tasks")
        
        if max_tasks:
            dataset_size = min(max_tasks, len(dataset))
        else:
            dataset_size = len(dataset)
        
        accuracies = []
        training_required = []
        
        for i in tqdm(range(dataset_size), desc="Evaluating"):
            try:
                # Get task
                test_sample, support_samples, test_solution = dataset[i]
                
                # Solve task (with training if needed)
                solution, accuracy = self.modulation_system.solve_and_train(
                    test_sample, support_samples, f"eval_task_{i}", train_if_needed=True
                )
                
                accuracies.append(accuracy)
                training_required.append(accuracy < train_threshold)
                
            except Exception as e:
                print(f"Error evaluating task {i}: {e}")
                accuracies.append(0.0)
                training_required.append(True)
        
        # Evaluation statistics
        eval_stats = {
            'total_tasks': dataset_size,
            'average_accuracy': np.mean(accuracies),
            'tasks_requiring_training': sum(training_required),
            'tasks_solved_without_training': sum(not req for req in training_required),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies)
        }
        
        print(f"\nEvaluation completed!")
        print(f"Total tasks: {eval_stats['total_tasks']}")
        print(f"Average accuracy: {eval_stats['average_accuracy']:.3f}")
        print(f"Tasks requiring training: {eval_stats['tasks_requiring_training']}")
        print(f"Tasks solved without training: {eval_stats['tasks_solved_without_training']}")
        
        return eval_stats
    
    def solve_single_task(self, 
                          test_sample: np.ndarray,
                          support_samples: List[Dict],
                          task_id: str = "single_task") -> Tuple[np.ndarray, float]:
        """Solve a single ARC task"""
        solution, accuracy = self.modulation_system.solve_and_train(
            test_sample, support_samples, task_id, train_if_needed=True
        )
        return solution, accuracy
    
    def save_model(self):
        """Save the modulation model"""
        print("Saving modulation model...")
        self.modulation_system.save_model()
        print("Model saved successfully")
    
    def load_model(self):
        """Load the modulation model"""
        print("Loading modulation model...")
        self.modulation_system.load_model()
        print("Model loaded successfully")
    
    def get_performance_summary(self) -> Dict:
        """Get summary of model performance"""
        perf_stats = self.modulation_system.get_performance_stats()
        
        summary = {
            'modulation_system_stats': perf_stats,
            'training_stats': self.stats,
            'model_parameters': sum(p.numel() for p in self.modulation_system.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.modulation_system.model.parameters() if p.requires_grad)
        }
        
        return summary
    
    def check_device_placement(self):
        """Check if the model is on the correct device"""
        print(f"Checking device placement...")
        model_device = next(self.modulation_system.model.parameters()).device
        print(f"Model device: {model_device}")
        if model_device != self.device:
            print(f"  WARNING: Model is on {model_device}, expected {self.device}")
            # Move to correct device
            self.modulation_system.model.to(self.device)
            print(f"  Moved to {self.device}")
        else:
            print(f"  Model is correctly placed on {self.device}")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Modulation-Based Solver")
    parser.add_argument("--mode", choices=["train", "evaluate", "solve"], 
                       default="train", help="Operation mode")
    parser.add_argument("--data_dir", default="arc-prize-2025", 
                       help="Directory containing ARC data")
    parser.add_argument("--model_dir", default="modulation_models", 
                       help="Directory to store/load modulation model")
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="Maximum number of tasks to process")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save model every N tasks during training")
    parser.add_argument("--train_threshold", type=float, default=0.7,
                       help="Accuracy threshold for determining if training is needed")
    parser.add_argument("--use_merged_training", action="store_true", default=True,
                       help="Use merged training approach (Phase 1: support samples, Phase 2: test data)")
    parser.add_argument("--no_merged_training", action="store_true", default=False,
                       help="Use individual task training approach")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training (smaller = more memory efficient)")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--max_support_per_batch", type=int, default=8,
                       help="Maximum support samples to use per batch")
    
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
    solver = ARCModulationSolver(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        device=args.device
    )
    
    if args.mode == "train":
        # Load training data
        try:
            train_dataset = solver.data_loader.load_training_data()
            print(f"Training dataset loaded: {len(train_dataset)} tasks")
            
            # Train the modulation model
            use_merged = args.use_merged_training and not args.no_merged_training
            stats = solver.train_on_dataset(
                train_dataset, 
                max_tasks=args.max_tasks,
                save_interval=args.save_interval,
                use_merged_training=use_merged,
                batch_size=args.batch_size,
                gradient_accumulation=args.gradient_accumulation,
                max_support_per_batch=args.max_support_per_batch
            )
            
            # Save final model
            solver.save_model()
            
            # Check device placement
            solver.check_device_placement()
            
            # Save training statistics
            with open("modulation_training_stats.json", "w") as f:
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
            eval_stats = solver.evaluate_on_dataset(
                test_dataset, 
                max_tasks=args.max_tasks,
                train_threshold=args.train_threshold
            )
            
            # Check device placement
            solver.check_device_placement()
            
            # Save evaluation statistics
            with open("modulation_evaluation_stats.json", "w") as f:
                json.dump(eval_stats, f, indent=2)
                
        except FileNotFoundError:
            print("Test data not found. Please check the data directory.")
            return
    
    elif args.mode == "solve":
        # Interactive mode for solving single tasks
        print("Interactive ARC task solver using modulation")
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
        print("\nSolving task using modulation...")
        solution, accuracy = solver.solve_single_task(padded_test, support_samples)
        
        # Check device placement
        solver.check_device_placement()
        
        print(f"\nSolution found with accuracy: {accuracy:.3f}")
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
    
    # Show final performance summary
    print("\n=== Final Performance Summary ===")
    summary = solver.get_performance_summary()
    print(f"Model parameters: {summary['model_parameters']:,}")
    print(f"Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"Total tasks processed: {summary['training_stats']['total_tasks']}")
    print(f"Average accuracy: {summary['training_stats']['average_accuracy']:.3f}")


if __name__ == "__main__":
    main()
