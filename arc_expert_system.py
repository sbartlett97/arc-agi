import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
from tqdm import tqdm
import time
from arc_vit import create_arc_model, ARCModel
from arc_dataset import ARCDataset


@dataclass
class Expert:
    """Represents a trained expert model"""
    id: str
    model: ARCModel
    delta_embeddings: List[np.ndarray]  # List of delta embeddings from training
    task_signatures: List[str]  # List of task IDs this expert can solve
    performance_score: float  # Average performance on validation tasks
    created_at: float  # Timestamp when expert was created
    num_training_samples: int  # Number of samples used for training


class ExpertManager:
    """Manages expert models for ARC tasks"""
    
    def __init__(self, 
                 experts_dir: str = "experts",
                 embed_dim: int = 256,
                 similarity_threshold: float = 0.8,
                 max_experts: int = 100):
        self.experts_dir = experts_dir
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        self.max_experts = max_experts
        
        # Create experts directory
        os.makedirs(experts_dir, exist_ok=True)
        
        # Load existing experts
        self.experts: List[Expert] = []
        self.expert_embeddings: List[np.ndarray] = []
        self.load_experts()
        
        # Initialize FAISS index for kNN search
        self.faiss_index = None
        self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if not self.expert_embeddings:
            return
        
        # Convert to numpy array
        embeddings_array = np.array(self.expert_embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.faiss_index.add(embeddings_array)
    
    def load_experts(self):
        """Load existing experts from disk"""
        if not os.path.exists(self.experts_dir):
            return
        
        for expert_dir in os.listdir(self.experts_dir):
            expert_path = os.path.join(self.experts_dir, expert_dir)
            if os.path.isdir(expert_path):
                try:
                    expert = self._load_expert(expert_path)
                    if expert:
                        self.experts.append(expert)
                        # Add average delta embedding to search index
                        avg_delta = np.mean(expert.delta_embeddings, axis=0)
                        self.expert_embeddings.append(avg_delta)
                except Exception as e:
                    print(f"Failed to load expert {expert_dir}: {e}")
    
    def _load_expert(self, expert_path: str) -> Optional[Expert]:
        """Load a single expert from disk"""
        # Load model
        model_path = os.path.join(expert_path, "model.pth")
        if not os.path.exists(model_path):
            return None
        
        model = create_arc_model()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Load metadata
        metadata_path = os.path.join(expert_path, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load delta embeddings
        embeddings_path = os.path.join(expert_path, "embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                delta_embeddings = pickle.load(f)
        else:
            delta_embeddings = []
        
        return Expert(
            id=metadata['id'],
            model=model,
            delta_embeddings=delta_embeddings,
            task_signatures=metadata['task_signatures'],
            performance_score=metadata['performance_score'],
            created_at=metadata['created_at'],
            num_training_samples=metadata['num_training_samples']
        )
    
    def save_expert(self, expert: Expert):
        """Save expert to disk"""
        expert_dir = os.path.join(self.experts_dir, expert.id)
        os.makedirs(expert_dir, exist_ok=True)
        
        # Save model
        torch.save(expert.model.state_dict(), os.path.join(expert_dir, "model.pth"))
        
        # Save metadata
        metadata = {
            'id': expert.id,
            'task_signatures': expert.task_signatures,
            'performance_score': expert.performance_score,
            'created_at': expert.created_at,
            'num_training_samples': expert.num_training_samples
        }
        with open(os.path.join(expert_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        # Save delta embeddings
        with open(os.path.join(expert_dir, "embeddings.pkl"), 'wb') as f:
            pickle.dump(expert.delta_embeddings, f)
    
    def calculate_delta_embeddings(self, 
                                 support_samples: List[Dict],
                                 model: ARCModel) -> List[np.ndarray]:
        """Calculate delta embeddings between input/output pairs"""
        deltas = []
        
        with torch.no_grad():
            for sample in support_samples:
                input_grid = torch.tensor(sample['input'], dtype=torch.long).unsqueeze(0)
                output_grid = torch.tensor(sample['output'], dtype=torch.long).unsqueeze(0)
                
                # Get embeddings
                input_emb = model.get_embeddings(input_grid)  # (1, num_patches, embed_dim)
                output_emb = model.get_embeddings(output_grid)  # (1, num_patches, embed_dim)
                
                # Calculate delta
                delta = output_emb - input_emb  # (1, num_patches, embed_dim)
                
                # Average across patches
                delta_avg = torch.mean(delta, dim=1).squeeze(0)  # (embed_dim,)
                deltas.append(delta_avg.numpy())
        
        return deltas
    
    def find_best_expert(self, 
                         delta_embeddings: List[np.ndarray],
                         k: int = 5) -> Optional[Tuple[Expert, float]]:
        """Find the best expert using kNN search on delta embeddings"""
        if not self.faiss_index or not self.experts:
            return None
        
        # Calculate average delta embedding
        avg_delta = np.mean(delta_embeddings, axis=0).astype(np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(avg_delta)
        
        # Search for similar experts
        similarities, indices = self.faiss_index.search(avg_delta, min(k, len(self.experts)))
        
        # Find best expert above threshold
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= self.similarity_threshold:
                return self.experts[idx], float(similarity)
        
        return None
    
    def validate_expert(self, 
                       expert: Expert,
                       support_samples: List[Dict],
                       threshold: float = 0.7) -> Tuple[bool, float]:
        """Validate if expert performs well on support samples"""
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for sample in support_samples:
                input_grid = torch.tensor(sample['input'], dtype=torch.long).unsqueeze(0)
                expected_output = torch.tensor(sample['output'], dtype=torch.long)
                
                # Get prediction
                predicted_output = expert.model(input_grid)
                
                # Calculate accuracy
                if predicted_output.shape == expected_output.shape:
                    accuracy = (predicted_output == expected_output).float().mean()
                    correct_predictions += accuracy.item()
                    total_predictions += 1
        
        if total_predictions == 0:
            return False, 0.0
        
        avg_accuracy = correct_predictions / total_predictions
        return avg_accuracy >= threshold, avg_accuracy
    
    def train_new_expert(self, 
                         support_samples: List[Dict],
                         task_id: str,
                         learning_rate: float = 1e-4,
                         num_epochs: int = 100) -> Expert:
        """Train a new expert on support samples"""
        # Create new model
        model = create_arc_model()
        model.train()
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare training data
        inputs = []
        targets = []
        for sample in support_samples:
            inputs.append(torch.tensor(sample['input'], dtype=torch.long))
            targets.append(torch.tensor(sample['output'], dtype=torch.long))
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            
            for input_grid, target_grid in zip(inputs, targets):
                optimizer.zero_grad()
                
                # Forward pass
                input_grid = input_grid.unsqueeze(0)
                target_grid = target_grid.unsqueeze(0)
                
                # Get logits
                input_grid_norm = model.input_norm(input_grid.float())
                input_grid_norm = input_grid_norm.unsqueeze(1)
                logits = model.vit(input_grid_norm)
                
                # Reshape for loss calculation
                batch_size, grid_size, _, num_classes = logits.shape
                logits = logits.view(batch_size * grid_size * grid_size, num_classes)
                targets_flat = target_grid.view(-1)
                
                # Calculate loss
                loss = criterion(logits, targets_flat)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Early stopping if loss is very low
            if total_loss < 0.01:
                break
        
        # Calculate delta embeddings
        model.eval()
        delta_embeddings = self.calculate_delta_embeddings(support_samples, model)
        
        # Create expert
        expert = Expert(
            id=f"expert_{len(self.experts)}_{task_id}",
            model=model,
            delta_embeddings=delta_embeddings,
            task_signatures=[task_id],
            performance_score=1.0,  # Will be updated during validation
            created_at=float(time.time()),
            num_training_samples=len(support_samples)
        )
        
        # Save expert
        self.save_expert(expert)
        
        # Add to active experts
        self.experts.append(expert)
        avg_delta = np.mean(delta_embeddings, axis=0)
        self.expert_embeddings.append(avg_delta)
        
        # Rebuild FAISS index
        self._build_faiss_index()
        
        return expert
    
    def merge_similar_experts(self, similarity_threshold: float = 0.95):
        """Merge experts that are very similar"""
        if len(self.experts) < 2:
            return
        
        # Calculate pairwise similarities
        embeddings_array = np.array(self.expert_embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        similarities = np.dot(embeddings_array, embeddings_array.T)
        
        # Find pairs to merge
        to_merge = []
        for i in range(len(self.experts)):
            for j in range(i + 1, len(self.experts)):
                if similarities[i, j] >= similarity_threshold:
                    to_merge.append((i, j))
        
        # Merge experts (keep the one with better performance)
        for i, j in to_merge:
            if self.experts[i].performance_score >= self.experts[j].performance_score:
                keep_idx, remove_idx = i, j
            else:
                keep_idx, remove_idx = j, i
            
            # Merge task signatures
            self.experts[keep_idx].task_signatures.extend(self.experts[remove_idx].task_signatures)
            self.experts[keep_idx].task_signatures = list(set(self.experts[keep_idx].task_signatures))
            
            # Remove the worse expert
            expert_to_remove = self.experts.pop(remove_idx)
            self.expert_embeddings.pop(remove_idx)
            
            # Remove from disk
            expert_dir = os.path.join(self.experts_dir, expert_to_remove.id)
            if os.path.exists(expert_dir):
                import shutil
                shutil.rmtree(expert_dir)
        
        # Rebuild FAISS index
        self._build_faiss_index()
    
    def solve_task(self, 
                   test_sample: np.ndarray,
                   support_samples: List[Dict],
                   task_id: str) -> Tuple[np.ndarray, Expert]:
        """Solve an ARC task using the expert system"""
        # Create a temporary model to calculate delta embeddings
        temp_model = create_arc_model()
        temp_model.eval()
        
        # Calculate delta embeddings
        delta_embeddings = self.calculate_delta_embeddings(support_samples, temp_model)
        
        # Find best expert
        expert_result = self.find_best_expert(delta_embeddings)
        
        if expert_result:
            expert, similarity = expert_result
            
            # Validate expert
            is_valid, accuracy = self.validate_expert(expert, support_samples)
            
            if is_valid:
                # Use existing expert
                with torch.no_grad():
                    test_tensor = torch.tensor(test_sample, dtype=torch.long).unsqueeze(0)
                    solution = expert.model(test_tensor)
                    return solution.squeeze(0).numpy(), expert
        
        # No suitable expert found, train new one
        print(f"Training new expert for task {task_id}")
        new_expert = self.train_new_expert(support_samples, task_id)
        
        # Use new expert
        with torch.no_grad():
            test_tensor = torch.tensor(test_sample, dtype=torch.long).unsqueeze(0)
            solution = new_expert.model(test_tensor)
            return solution.squeeze(0).numpy(), new_expert


def create_expert_manager(experts_dir: str = "experts") -> ExpertManager:
    """Create an expert manager with default settings"""
    return ExpertManager(experts_dir=experts_dir)


if __name__ == "__main__":
    # Test the expert system
    manager = create_expert_manager()
    
    print(f"Loaded {len(manager.experts)} experts")
    
    # Test with sample data
    from arc_dataset import create_sample_batch
    
    test_sample, support_samples, test_solution = create_sample_batch()
    
    # Solve task
    solution, expert = manager.solve_task(test_sample, support_samples, "test_task")
    
    print(f"Solution shape: {solution.shape}")
    print(f"Expert ID: {expert.id}")
    print(f"Expert performance: {expert.performance_score}")
    
    # Test merging
    manager.merge_similar_experts()
    print(f"After merging: {len(manager.experts)} experts")
