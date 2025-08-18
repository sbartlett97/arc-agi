#!/usr/bin/env python3
"""
Comparison script for expert-based vs modulation-based approaches
"""

import torch
import numpy as np
import time
from typing import List, Dict, Tuple

from arc_expert_system import create_expert_manager
from arc_modulation_system import create_modulation_system


def create_sample_task():
    """Create a sample ARC task for testing"""
    # Create a simple pattern: input grid with some colored cells
    input_grid = np.zeros((10, 10), dtype=np.int64)
    input_grid[2:4, 2:4] = 1  # 2x2 red square
    input_grid[6:8, 6:8] = 2  # 2x2 blue square
    
    # Output: move the squares down by 2 positions
    output_grid = np.zeros((10, 10), dtype=np.int64)
    output_grid[4:6, 2:4] = 1  # moved red square
    output_grid[8:10, 6:8] = 2  # moved blue square
    
    # Support samples
    support_samples = [
        {
            'input': input_grid.copy(),
            'output': output_grid.copy()
        }
    ]
    
    # Test sample: similar pattern but different size
    test_input = np.zeros((8, 8), dtype=np.int64)
    test_input[1:3, 1:3] = 1  # smaller red square
    test_input[4:6, 4:6] = 2  # smaller blue square
    
    test_output = np.zeros((8, 8), dtype=np.int64)
    test_output[3:5, 1:3] = 1  # moved red square
    test_output[6:8, 4:6] = 2  # moved blue square
    
    return test_input, support_samples, test_output


def test_expert_approach():
    """Test the expert-based approach"""
    print("=== Testing Expert-Based Approach ===")
    
    # Create expert manager
    expert_manager = create_expert_manager(experts_dir="test_experts")
    
    # Create sample task
    test_input, support_samples, test_output = create_sample_task()
    
    # Time the expert approach
    start_time = time.time()
    
    try:
        # Solve task
        solution, expert = expert_manager.solve_task(test_input, support_samples, "test_task")
        
        # Calculate accuracy
        if solution.shape == test_output.shape:
            accuracy = (solution == test_output).mean()
        else:
            accuracy = 0.0
        
        expert_time = time.time() - start_time
        
        print(f"Expert approach:")
        print(f"  Solution shape: {solution.shape}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Time: {expert_time:.3f}s")
        print(f"  Expert ID: {expert.id}")
        print(f"  Number of experts: {len(expert_manager.experts)}")
        
        return accuracy, expert_time, len(expert_manager.experts)
        
    except Exception as e:
        print(f"Expert approach failed: {e}")
        return 0.0, time.time() - start_time, 0


def test_modulation_approach():
    """Test the modulation-based approach"""
    print("\n=== Testing Modulation-Based Approach ===")
    
    # Create modulation system
    modulation_system = create_modulation_system(model_dir="test_modulation_models")
    
    # Create sample task
    test_input, support_samples, test_output = create_sample_task()
    
    # Time the modulation approach
    start_time = time.time()
    
    try:
        # Solve task
        solution, accuracy = modulation_system.solve_and_train(
            test_input, support_samples, "test_task", train_if_needed=True
        )
        
        modulation_time = time.time() - start_time
        
        print(f"Modulation approach:")
        print(f"  Solution shape: {solution.shape}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Time: {modulation_time:.3f}s")
        print(f"  Model parameters: {sum(p.numel() for p in modulation_system.model.parameters()):,}")
        
        return accuracy, modulation_time, sum(p.numel() for p in modulation_system.model.parameters())
        
    except Exception as e:
        print(f"Modulation approach failed: {e}")
        return 0.0, time.time() - start_time, 0


def compare_approaches():
    """Compare both approaches"""
    print("ARC-AGI Approach Comparison")
    print("=" * 50)
    
    # Test expert approach
    expert_acc, expert_time, num_experts = test_expert_approach()
    
    # Test modulation approach
    modulation_acc, modulation_time, num_params = test_modulation_approach()
    
    # Comparison summary
    print("\n=== Comparison Summary ===")
    print(f"{'Metric':<20} {'Expert':<15} {'Modulation':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {expert_acc:<15.3f} {modulation_acc:<15.3f}")
    print(f"{'Time (s)':<20} {expert_time:<15.3f} {modulation_time:<15.3f}")
    print(f"{'Complexity':<20} {f'{num_experts} experts':<15} {f'{num_params:,} params':<15}")
    
    # Advantages/disadvantages
    print("\n=== Key Differences ===")
    print("Expert-Based Approach:")
    print("  + Task-specific optimization")
    print("  + Can reuse learned patterns")
    print("  - Requires storing multiple models")
    print("  - Limited generalization across tasks")
    print("  - Memory grows with number of tasks")
    
    print("\nModulation-Based Approach:")
    print("  + Single model for all tasks")
    print("  + Learns to adapt via modulation")
    print("  + Better generalization potential")
    print("  - Requires training on support samples")
    print("  - May need more training data initially")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if expert_acc > modulation_acc:
        print("Expert approach performed better on this specific task")
    elif modulation_acc > expert_acc:
        print("Modulation approach performed better on this specific task")
    else:
        print("Both approaches performed similarly on this task")
    
    if expert_time < modulation_time:
        print("Expert approach was faster")
    else:
        print("Modulation approach was faster")
    
    print("\nFor production use:")
    print("- Use expert approach if you have many similar tasks and want fast inference")
    print("- Use modulation approach if you want a single model that can adapt to new tasks")


if __name__ == "__main__":
    compare_approaches()

