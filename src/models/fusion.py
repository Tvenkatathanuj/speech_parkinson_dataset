"""
Multi-Modal Fusion Module

Fuses acoustic and prosodic features using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion layer for combining acoustic and prosodic features.
    
    Supports multiple fusion strategies:
    - 'concat': Simple concatenation + projection
    - 'attention': Cross-attention between modalities
    - 'gated': Gated fusion with learnable weights
    """
    
    def __init__(
        self,
        acoustic_dim: int,
        prosodic_dim: int,
        output_dim: int,
        fusion_type: str = "attention",
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            acoustic_dim: Dimension of acoustic features
            prosodic_dim: Dimension of prosodic features
            output_dim: Output dimension after fusion
            fusion_type: Type of fusion ('concat', 'attention', 'gated')
            num_heads: Number of attention heads (for attention fusion)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.acoustic_dim = acoustic_dim
        self.prosodic_dim = prosodic_dim
        self.output_dim = output_dim
        
        if fusion_type == "concat":
            # Simple concatenation + projection
            self.projection = nn.Linear(acoustic_dim + prosodic_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            
        elif fusion_type == "attention":
            # Cross-attention fusion
            assert acoustic_dim == prosodic_dim, "Dimensions must match for attention fusion"
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=acoustic_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(acoustic_dim)
            self.projection = nn.Linear(acoustic_dim, output_dim)
            
        elif fusion_type == "gated":
            # Gated fusion with learnable gates
            self.acoustic_gate = nn.Sequential(
                nn.Linear(acoustic_dim, acoustic_dim),
                nn.Sigmoid()
            )
            self.prosodic_gate = nn.Sequential(
                nn.Linear(prosodic_dim, prosodic_dim),
                nn.Sigmoid()
            )
            self.projection = nn.Linear(acoustic_dim + prosodic_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        acoustic: torch.Tensor,
        prosodic: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse acoustic and prosodic features.
        
        Args:
            acoustic: Acoustic features [batch, time, acoustic_dim]
            prosodic: Prosodic features [batch, time, prosodic_dim]
            
        Returns:
            Fused features [batch, time, output_dim]
        """
        if self.fusion_type == "concat":
            # Concatenate and project
            combined = torch.cat([acoustic, prosodic], dim=-1)
            output = self.projection(combined)
            output = self.dropout(output)
            
        elif self.fusion_type == "attention":
            # Cross-attention: acoustic attends to prosodic
            attn_out, _ = self.cross_attention(
                query=acoustic,
                key=prosodic,
                value=prosodic
            )
            # Residual connection
            output = self.norm(acoustic + attn_out)
            output = self.projection(output)
            
        elif self.fusion_type == "gated":
            # Gated fusion
            acoustic_gated = acoustic * self.acoustic_gate(acoustic)
            prosodic_gated = prosodic * self.prosodic_gate(prosodic)
            combined = torch.cat([acoustic_gated, prosodic_gated], dim=-1)
            output = self.projection(combined)
            output = self.dropout(output)
        
        return output


if __name__ == "__main__":
    # Test fusion module
    batch_size = 2
    time_steps = 100
    acoustic_dim = 256
    prosodic_dim = 256
    output_dim = 256
    
    for fusion_type in ["concat", "attention", "gated"]:
        print(f"\nTesting {fusion_type} fusion:")
        
        fusion = MultiModalFusion(
            acoustic_dim=acoustic_dim,
            prosodic_dim=prosodic_dim,
            output_dim=output_dim,
            fusion_type=fusion_type
        )
        
        acoustic = torch.randn(batch_size, time_steps, acoustic_dim)
        prosodic = torch.randn(batch_size, time_steps, prosodic_dim)
        
        output = fusion(acoustic, prosodic)
        print(f"  Input shapes: acoustic={acoustic.shape}, prosodic={prosodic.shape}")
        print(f"  Output shape: {output.shape}")
