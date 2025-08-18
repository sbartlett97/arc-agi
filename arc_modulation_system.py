import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
from arc_modulation_vit import create_arc_modulation_model, ARCModulationModel
from arc_dataset import ARCDataset


@dataclass
class ModulationTask:
    """Represents a task with its support samples and performance"""
    id: str
    support_samples: List[Dict]
    performance_score: float
    created_at: float
    num_support_samples: int


class ModulationSystem:
    """System that uses delta-embedding and modulation instead of task-specific experts"""
    
    def __init__(self, 
                 model_dir: str = "modulation_models",
                 embed_dim: int = 256,
                 device: str = "auto"):
        self.model_dir = model_dir
        self.embed_dim = embed_dim
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Modulation system using device: {self.device}")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize the modulation model
        self.model = create_arc_modulation_model(embed_dim=embed_dim)
        self.model.to(self.device)
        
        # Task history for analysis
        self.task_history: List[ModulationTask] = []
        
        # Load existing model if available
        self.load_model()
    
    def load_model(self):
        """Load existing model from disk"""
        model_path = os.path.join(self.model_dir, "model.pth")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                print(f"Loaded existing model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Starting with fresh model")
        else:
            print("No existing model found, starting fresh")
    
    def save_model(self):
        """Save model to disk"""
        model_path = os.path.join(self.model_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save task history
        history_path = os.path.join(self.model_dir, "task_history.json")
        history_data = []
        for task in self.task_history:
            history_data.append({
                'id': task.id,
                'performance_score': task.performance_score,
                'created_at': task.created_at,
                'num_support_samples': task.num_support_samples
            })
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def solve_task(self, 
                   test_sample: np.ndarray,
                   support_samples: List[Dict],
                   task_id: str) -> np.ndarray:
        """
        Solve an ARC task using modulation
        
        Args:
            test_sample: Test input grid
            support_samples: List of support sample dicts with 'input' and 'output' keys
            task_id: Identifier for the task
            
        Returns:
            Predicted output grid
        """
        # Convert to tensors
        test_tensor = torch.tensor(test_sample, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Get prediction with modulation
        with torch.no_grad():
            self.model.eval()
            solution = self.model(test_tensor, support_samples)
        
        # Convert back to numpy
        solution_np = solution.squeeze(0).cpu().numpy()
        
        # Record task
        task = ModulationTask(
            id=task_id,
            support_samples=support_samples,
            performance_score=1.0,  # Will be updated during training
            created_at=float(time.time()),
            num_support_samples=len(support_samples)
        )
        self.task_history.append(task)
        
        return solution_np
    
    def train_on_support_samples(self, 
                                support_samples: List[Dict],
                                learning_rate: float = 1e-4,
                                num_epochs: int = 50,
                                batch_size: int = 4) -> float:
        """
        Train the model on support samples
        
        Args:
            support_samples: List of support sample dicts
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Final training loss
        """
        self.model.train()
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare training data
        inputs = []
        targets = []
        for sample in support_samples:
            inputs.append(torch.tensor(sample['input'], dtype=torch.long, device=self.device))
            targets.append(torch.tensor(sample['output'], dtype=torch.long, device=self.device))
        
        # Training loop
        total_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            # Process in batches
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                
                # Pad batch to full grid size (30x30)
                padded_inputs = []
                padded_targets = []
                for inp, tgt in zip(batch_inputs, batch_targets):
                    # Pad input to 30x30
                    padded_inp = torch.zeros(30, 30, dtype=inp.dtype, device=self.device)
                    padded_inp[:inp.shape[0], :inp.shape[1]] = inp
                    padded_inputs.append(padded_inp)
                    
                    # Pad target to 30x30
                    padded_tgt = torch.zeros(30, 30, dtype=tgt.dtype, device=self.device)
                    padded_tgt[:tgt.shape[0], :tgt.shape[1]] = tgt
                    padded_targets.append(padded_tgt)
                
                # Stack into batch
                batch_input = torch.stack(padded_inputs)
                batch_target = torch.stack(padded_targets)
                
                optimizer.zero_grad()
                
                # Forward pass with modulation
                logits = self.model.vit(batch_input, support_samples)
                
                # Reshape for loss calculation
                actual_batch_size, grid_size, _, num_classes = logits.shape
                logits = logits.view(actual_batch_size * grid_size * grid_size, num_classes)
                targets_flat = batch_target.view(-1)
                
                # Calculate loss
                loss = criterion(logits, targets_flat)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
            
            # Early stopping if loss is very low
            if epoch_loss < 0.01:
                break
        
        avg_loss = total_loss / num_epochs
        print(f"Training completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate_on_support_samples(self, 
                                   support_samples: List[Dict],
                                   threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Validate model performance on support samples
        
        Args:
            support_samples: List of support sample dicts
            threshold: Accuracy threshold for validation
            
        Returns:
            Tuple of (is_valid, accuracy)
        """
        self.model.eval()
        
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for sample in support_samples:
                input_grid = torch.tensor(sample['input'], dtype=torch.long, device=self.device).unsqueeze(0)
                expected_output = torch.tensor(sample['output'], dtype=torch.long, device=self.device)
                
                # Get prediction
                predicted_output = self.model(input_grid, support_samples)
                
                # Calculate accuracy
                if predicted_output.shape == expected_output.shape:
                    accuracy = (predicted_output == expected_output).float().mean()
                    correct_predictions += accuracy.item()
                    total_predictions += 1
        
        if total_predictions == 0:
            return False, 0.0
        
        avg_accuracy = correct_predictions / total_predictions
        return avg_accuracy >= threshold, avg_accuracy
    
    def solve_and_train(self, 
                        test_sample: np.ndarray,
                        support_samples: List[Dict],
                        task_id: str,
                        train_if_needed: bool = True) -> Tuple[np.ndarray, float]:
        """
        Solve a task and optionally train if performance is poor
        
        Args:
            test_sample: Test input grid
            support_samples: List of support sample dicts
            task_id: Identifier for the task
            train_if_needed: Whether to train if validation fails
            
        Returns:
            Tuple of (solution, accuracy)
        """
        # First, try to solve without training
        solution = self.solve_task(test_sample, support_samples, task_id)
        
        # Validate performance
        is_valid, accuracy = self.validate_on_support_samples(support_samples)
        
        # Update task performance
        for task in self.task_history:
            if task.id == task_id:
                task.performance_score = accuracy
                break
        
        # Train if needed and requested
        if not is_valid and train_if_needed:
            print(f"Performance below threshold ({accuracy:.3f}), training on support samples...")
            loss = self.train_on_support_samples(support_samples)
            
            # Re-solve after training
            solution = self.solve_task(test_sample, support_samples, task_id)
            
            # Re-validate
            is_valid, accuracy = self.validate_on_support_samples(support_samples)
            
            # Update task performance
            for task in self.task_history:
                if task.id == task_id:
                    task.performance_score = accuracy
                    break
            
            print(f"After training - Accuracy: {accuracy:.3f}")
        
        return solution, accuracy
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.task_history:
            return {
                'total_tasks': 0,
                'average_accuracy': 0.0,
                'tasks_above_threshold': 0
            }
        
        accuracies = [task.performance_score for task in self.task_history]
        avg_accuracy = np.mean(accuracies)
        tasks_above_threshold = sum(1 for acc in accuracies if acc >= 0.7)
        
        return {
            'total_tasks': len(self.task_history),
            'average_accuracy': avg_accuracy,
            'tasks_above_threshold': tasks_above_threshold,
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies)
        }
    
    def clear_task_history(self):
        """Clear task history"""
        self.task_history.clear()
        print("Task history cleared")


def create_modulation_system(model_dir: str = "modulation_models", device: str = "auto") -> ModulationSystem:
    """Create a modulation system with default settings"""
    return ModulationSystem(model_dir=model_dir, device=device)


if __name__ == "__main__":
    # Test the modulation system
    system = create_modulation_system()
    
    # Create sample data
    test_sample = np.random.randint(0, 10, (15, 15))
    support_samples = [
        {
            'input': np.random.randint(0, 10, (10, 10)),
            'output': np.random.randint(0, 10, (10, 10))
        },
        {
            'input': np.random.randint(0, 10, (12, 12)),
            'output': np.random.randint(0, 10, (12, 12))
        }
    ]
    
    print("Testing modulation system...")
    
    # Solve task
    solution, accuracy = system.solve_and_train(
        test_sample, support_samples, "test_task_1", train_if_needed=True
    )
    
    print(f"Solution shape: {solution.shape}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Get performance stats
    stats = system.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Save model
    system.save_model()
    
    print("Modulation system test completed successfully!")
