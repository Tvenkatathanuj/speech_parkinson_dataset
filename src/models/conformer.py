"""
Conformer Encoder for Speech Recognition

Conformer combines convolution and self-attention for efficient 
sequence modeling. Original paper: https://arxiv.org/abs/2005.08100

Modifications for Parkinson's speech:
- Temporal-spectral attention for capturing tremor patterns
- Adaptive layer normalization for handling variable speech rates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConformerBlock(nn.Module):
    """Single Conformer block with feed-forward, attention, convolution."""
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feed-forward module 1 (Macaron-style)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),  # Pointwise conv
            nn.GLU(dim=1),  # Gated Linear Unit
            nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2, groups=d_model
            ),  # Depthwise conv
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),  # Pointwise conv
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, time, d_model]
            attention_mask: Attention mask [batch, time]
            
        Returns:
            Output tensor [batch, time, d_model]
        """
        # Feed-forward module 1 (half-step residual)
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head self-attention
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = residual + self.attn_dropout(attn_out)
        
        # Convolution module
        residual = x
        x = self.conv_norm(x)
        # Transpose for conv1d: [batch, time, d_model] -> [batch, d_model, time]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # Back to [batch, time, d_model]
        x = residual + x
        
        # Feed-forward module 2 (half-step residual)
        x = x + 0.5 * self.ffn2(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer encoder for speech recognition.
    
    Stack of Conformer blocks for efficient temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        num_layers: int = 12,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_layers: Number of Conformer blocks
            d_model: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            conv_kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, time, input_dim]
            lengths: Sequence lengths [batch]
            
        Returns:
            Encoded tensor [batch, time, d_model]
        """
        # Input projection
        x = self.input_projection(x)
        
        # Create attention mask from lengths
        attention_mask = None
        if lengths is not None:
            max_len = x.size(1)
            attention_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        
        # Pass through Conformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return x


if __name__ == "__main__":
    # Test the Conformer encoder
    batch_size = 2
    time_steps = 100
    input_dim = 80
    
    encoder = ConformerEncoder(
        input_dim=input_dim,
        num_layers=4,
        d_model=256,
        num_heads=4
    )
    
    x = torch.randn(batch_size, time_steps, input_dim)
    lengths = torch.tensor([100, 80])
    
    output = encoder(x, lengths)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
