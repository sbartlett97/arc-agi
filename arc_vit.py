import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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


class ARCViT(nn.Module):
    """Vision Transformer for ARC grid tasks"""
    
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)
            
        Returns:
            Output tensor of shape (batch_size, grid_size, grid_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Reshape to patches: (batch_size, num_patches, patch_size^2 * in_channels)
        x = x.view(batch_size, self.num_patches, self.patch_size ** 2 * self.in_channels)
        
        # Project to embeddings
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (num_patches, batch_size, embed_dim)
        x = self.pos_embed(x)
        x = x.transpose(0, 1)  # (batch_size, num_patches, embed_dim)
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, num_patches, embed_dim)
        
        # Project to output classes
        x = self.output_proj(x)  # (batch_size, num_patches, num_classes)
        
        # Reshape back to grid
        x = x.view(batch_size, self.grid_size, self.grid_size, -1)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for kNN search"""
        batch_size = x.shape[0]
        
        # Reshape to patches
        x = x.view(batch_size, self.num_patches, self.patch_size ** 2 * self.in_channels)
        
        # Project to embeddings
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)
        x = self.pos_embed(x)
        x = x.transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Return embeddings (batch_size, num_patches, embed_dim)
        return x


class ARCModel(nn.Module):
    """Complete ARC model with input/output handling"""
    
    def __init__(self, 
                 grid_size: int = 30,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 10):
        super().__init__()
        
        self.grid_size = grid_size
        self.vit = ARCViT(
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
    
    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)
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
        
        # Get logits
        logits = self.vit(x)  # (batch_size, grid_size, grid_size, num_classes)
        
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


def create_arc_model(grid_size: int = 30, 
                    embed_dim: int = 256,
                    num_heads: int = 8,
                    num_layers: int = 6) -> ARCModel:
    """Create an ARC model with default parameters"""
    return ARCModel(
        grid_size=grid_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )


if __name__ == "__main__":
    # Test the model
    model = create_arc_model()
    
    # Create sample input
    batch_size = 2
    x = torch.randint(0, 10, (batch_size, 30, 30))
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test with target size
    output_cropped = model(x, target_size=(15, 20))
    print(f"Cropped output shape: {output_cropped.shape}")
    
    # Test embeddings
    embeddings = model.get_embeddings(x)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Model test completed successfully!")
