import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 900):  # 30x30 = 900
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class DeltaEmbeddingNetwork(nn.Module):
    """Network that calculates delta embeddings between input/output pairs"""
    
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Network to process delta embeddings
        self.delta_processor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Layer normalization for delta embeddings
        self.delta_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, input_embeddings: torch.Tensor, output_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate and process delta embeddings
        
        Args:
            input_embeddings: (batch_size, num_patches, embed_dim)
            output_embeddings: (batch_size, num_patches, embed_dim)
            
        Returns:
            Processed delta embeddings: (batch_size, embed_dim)
        """
        # Calculate delta
        delta = output_embeddings - input_embeddings  # (batch_size, num_patches, embed_dim)
        
        # Average across patches
        delta_avg = torch.mean(delta, dim=1)  # (batch_size, embed_dim)
        
        # Process through network
        delta_processed = self.delta_processor(delta_avg)  # (batch_size, embed_dim)
        
        # Normalize
        delta_processed = self.delta_norm(delta_processed)
        
        return delta_processed


class SupportNetwork(nn.Module):
    """Network that learns modulation parameters from delta embeddings"""
    
    def __init__(self, 
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 num_modulation_params: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modulation_params = num_modulation_params
        
        # Network to generate modulation parameters
        self.modulation_generator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_modulation_params)
        )
        
        # Layer normalization
        self.modulation_norm = nn.LayerNorm(num_modulation_params)
    
    def forward(self, delta_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate modulation parameters from delta embedding
        
        Args:
            delta_embedding: (batch_size, embed_dim)
            
        Returns:
            Modulation parameters: (batch_size, num_modulation_params)
        """
        modulation_params = self.modulation_generator(delta_embedding)
        modulation_params = self.modulation_norm(modulation_params)
        return modulation_params


class ModulatedTransformerDecoder(nn.Module):
    """Transformer decoder that can be modulated by external parameters"""
    
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Create transformer decoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Modulation layers for each transformer layer
        self.modulation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, d_model * 2),  # 1024 is num_modulation_params
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor,
                modulation_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with modulation
        
        Args:
            tgt: Target sequence (batch_size, seq_len, d_model)
            memory: Memory sequence from encoder (batch_size, seq_len, d_model)
            modulation_params: Modulation parameters (batch_size, num_modulation_params)
            
        Returns:
            Modulated output (batch_size, seq_len, d_model)
        """
        output = tgt
        
        for i, (layer, mod_layer) in enumerate(zip(self.layers, self.modulation_layers)):
            # Apply transformer layer
            output = layer(output, memory)
            
            # Apply modulation
            modulation = mod_layer(modulation_params)  # (batch_size, d_model)
            modulation = modulation.unsqueeze(1)  # (batch_size, 1, d_model)
            
            # Add modulation to output
            output = output + modulation
        
        # Final normalization
        output = self.norm(output)
        return output


class ARCModulationViT(nn.Module):
    """Vision Transformer with delta-embedding and modulation for ARC tasks"""
    
    def __init__(self, 
                 grid_size: int = 30,
                 patch_size: int = 1,
                 in_channels: int = 1,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 mlp_ratio: float = 4.0,
                 num_classes: int = 10,  # Maximum color value + 1
                 dropout: float = 0.1):
        super().__init__()
        
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (grid_size // patch_size) ** 2
        
        # Patch embedding: convert each 1x1 patch to embedding
        self.patch_embed = nn.Linear(in_channels, embed_dim)
        
        # Positional encoding
        self.pos_embed = PositionalEncoding(embed_dim, self.num_patches)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Delta embedding network
        self.delta_network = DeltaEmbeddingNetwork(embed_dim=embed_dim)
        
        # Support network for modulation parameters
        self.support_network = SupportNetwork(embed_dim=embed_dim)
        
        # Modulated transformer decoder
        self.decoder = ModulatedTransformerDecoder(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embeddings"""
        batch_size = x.shape[0]
        
        # Handle different input shapes
        if len(x.shape) == 4:  # (batch_size, channels, height, width)
            # Remove channel dimension and ensure correct size
            if x.shape[1] == 1:  # Single channel
                x = x.squeeze(1)  # (batch_size, height, width)
            else:
                raise ValueError(f"Expected 1 channel, got {x.shape[1]}")
        
        # Convert to float
        x = x.float()
        
        # Ensure input is padded to grid_size if needed
        if x.shape[1] != self.grid_size or x.shape[2] != self.grid_size:
            padded_x = torch.zeros(batch_size, self.grid_size, self.grid_size, dtype=x.dtype, device=x.device)
            padded_x[:, :x.shape[1], :x.shape[2]] = x
            x = padded_x
        
        # Reshape to patches: (batch_size, num_patches, patch_size^2 * in_channels)
        x = x.view(batch_size, self.num_patches, self.patch_size ** 2 * self.in_channels)
        
        # Project to embeddings
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (num_patches, batch_size, embed_dim)
        x = self.pos_embed(x)
        x = x.transpose(0, 1)  # (batch_size, num_patches, embed_dim)
        
        # Apply transformer encoder
        encoded = self.encoder(x)  # (batch_size, num_patches, embed_dim)
        
        return encoded
    
    def forward(self, 
                x: torch.Tensor,
                support_samples: List[Dict] = None) -> torch.Tensor:
        """
        Forward pass with support sample modulation
        
        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)
            support_samples: List of support sample dicts with 'input' and 'output' keys
            
        Returns:
            Output tensor of shape (batch_size, grid_size, grid_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Encode input
        encoded = self.encode(x)  # (batch_size, num_patches, embed_dim)
        
        if support_samples is not None and len(support_samples) > 0:
            # Process support samples to get modulation parameters
            modulation_params = self._get_modulation_params(support_samples)
        else:
            # Use zero modulation if no support samples
            modulation_params = torch.zeros(batch_size, 1024, device=x.device)
        
        # Use encoded input as both target and memory for decoder
        # This allows the modulation to influence the decoding process
        decoded = self.decoder(encoded, encoded, modulation_params)
        
        # Project to output classes
        output = self.output_proj(decoded)  # (batch_size, num_patches, num_classes)
        
        # Reshape back to grid
        output = output.view(batch_size, self.grid_size, self.grid_size, -1)
        
        return output
    
    def _get_modulation_params(self, support_samples: List[Dict]) -> torch.Tensor:
        """Get modulation parameters from support samples"""
        device = next(self.parameters()).device
        
        # Process each support sample
        delta_embeddings = []
        for sample in support_samples:
            input_grid = torch.tensor(sample['input'], dtype=torch.long, device=device).unsqueeze(0)
            output_grid = torch.tensor(sample['output'], dtype=torch.long, device=device).unsqueeze(0)
            
            # Encode input and output
            input_encoded = self.encode(input_grid)
            output_encoded = self.encode(output_grid)
            
            # Get delta embedding
            delta_emb = self.delta_network(input_encoded, output_encoded)
            delta_embeddings.append(delta_emb)
        
        # Average delta embeddings across support samples
        if delta_embeddings:
            avg_delta = torch.mean(torch.stack(delta_embeddings), dim=0)
        else:
            avg_delta = torch.zeros(1, self.embed_dim, device=device)
        
        # Generate modulation parameters
        modulation_params = self.support_network(avg_delta)  # (1, num_modulation_params)
        
        return modulation_params
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for kNN search (compatibility method)"""
        return self.encode(x)


class ARCModulationModel(nn.Module):
    """Complete ARC model with modulation capabilities"""
    
    def __init__(self, 
                 grid_size: int = 30,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 10):
        super().__init__()
        
        self.grid_size = grid_size
        self.vit = ARCModulationViT(
            grid_size=grid_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Input normalization
        self.input_norm = nn.LayerNorm([grid_size, grid_size])
        
        # Output mask for variable grid sizes
        self.output_mask = nn.Parameter(torch.ones(grid_size, grid_size))
    
    def forward(self, 
                x: torch.Tensor, 
                support_samples: List[Dict] = None,
                target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass with support sample modulation
        
        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)
            support_samples: List of support sample dicts
            target_size: Optional target output size (rows, cols)
            
        Returns:
            Output tensor of shape (batch_size, target_rows, target_cols) or (batch_size, grid_size, grid_size)
        """
        # Ensure input is padded to grid_size
        batch_size, input_rows, input_cols = x.shape
        if input_rows != self.grid_size or input_cols != self.grid_size:
            # Pad input to grid_size
            padded_x = torch.zeros(batch_size, self.grid_size, self.grid_size, dtype=x.dtype, device=x.device)
            padded_x[:, :input_rows, :input_cols] = x
            x = padded_x
        
        # Normalize input
        x = self.input_norm(x.float())
        
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, grid_size, grid_size)
        
        # Get logits with modulation
        logits = self.vit(x, support_samples)  # (batch_size, grid_size, grid_size, num_classes)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs)
        output = dist.sample()  # (batch_size, grid_size, grid_size)
        
        # Apply output mask
        output = output * self.output_mask.long()
        
        # If target size specified, crop to that size
        if target_size is not None:
            target_rows, target_cols = target_size
            output = output[:, :target_rows, :target_cols]
        
        return output
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for kNN search"""
        # Ensure input is padded to grid_size
        batch_size, input_rows, input_cols = x.shape
        if input_rows != self.grid_size or input_cols != self.grid_size:
            # Pad input to grid_size
            padded_x = torch.zeros(batch_size, self.grid_size, self.grid_size, dtype=x.dtype, device=x.device)
            padded_x[:, :input_rows, :input_cols] = x
            x = padded_x
        
        x = self.input_norm(x.float())
        x = x.unsqueeze(1)
        return self.vit.get_embeddings(x)


def create_arc_modulation_model(grid_size: int = 30, 
                               embed_dim: int = 256,
                               num_heads: int = 8,
                               num_layers: int = 6) -> ARCModulationModel:
    """Create an ARC modulation model with default parameters"""
    return ARCModulationModel(
        grid_size=grid_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )


if __name__ == "__main__":
    # Test the modulation model
    model = create_arc_modulation_model()
    
    # Create sample input
    batch_size = 2
    x = torch.randint(0, 10, (batch_size, 30, 30))
    
    # Create sample support samples
    support_samples = [
        {
            'input': torch.randint(0, 10, (15, 15)).numpy(),
            'output': torch.randint(0, 10, (15, 15)).numpy()
        },
        {
            'input': torch.randint(0, 10, (20, 20)).numpy(),
            'output': torch.randint(0, 10, (20, 20)).numpy()
        }
    ]
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass without support samples
    output = model(x)
    print(f"Output shape (no support): {output.shape}")
    
    # Test forward pass with support samples
    output_modulated = model(x, support_samples)
    print(f"Output shape (with support): {output_modulated.shape}")
    
    # Test with target size
    output_cropped = model(x, support_samples, target_size=(15, 20))
    print(f"Cropped output shape: {output_cropped.shape}")
    
    # Test embeddings
    embeddings = model.get_embeddings(x)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Modulation model test completed successfully!")
