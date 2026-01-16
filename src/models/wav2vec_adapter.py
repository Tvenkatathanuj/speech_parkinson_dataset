"""
Wav2Vec 2.0 Adapter for Pre-trained Feature Extraction

Wraps Hugging Face's Wav2Vec 2.0 models for use in our architecture.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional


class Wav2VecAdapter(nn.Module):
    """
    Adapter for Wav2Vec 2.0 pre-trained models.
    
    Loads pre-trained Wav2Vec 2.0 and optionally freezes it for 
    feature extraction or fine-tunes it end-to-end.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze: bool = False,
        output_dim: Optional[int] = None
    ):
        """
        Args:
            model_name: Name of Wav2Vec 2.0 model from Hugging Face
            freeze: Whether to freeze the Wav2Vec 2.0 parameters
            output_dim: If specified, add projection layer to this dimension
        """
        super().__init__()
        
        # Load pre-trained Wav2Vec 2.0
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_dim = self.wav2vec.config.hidden_size
        
        # Optionally freeze parameters
        if freeze:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        
        # Optional output projection
        if output_dim is not None:
            self.projection = nn.Linear(self.hidden_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projection = None
            self.output_dim = self.hidden_dim
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from raw waveform.
        
        Args:
            waveform: Raw audio waveform [batch, time]
            
        Returns:
            Features [batch, time_downsampled, hidden_dim]
        """
        # Wav2Vec 2.0 forward pass
        outputs = self.wav2vec(waveform)
        features = outputs.last_hidden_state  # [batch, time, hidden_dim]
        
        # Optional projection
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test Wav2Vec adapter
    print("Testing Wav2Vec 2.0 Adapter...")
    
    # Small model for testing
    adapter = Wav2VecAdapter(
        model_name="facebook/wav2vec2-base",
        freeze=False,
        output_dim=256
    )
    
    # Create dummy waveform (1 second at 16kHz)
    batch_size = 2
    waveform = torch.randn(batch_size, 16000)
    
    # Forward pass
    features = adapter(waveform)
    
    print(f"Input shape: {waveform.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Trainable parameters: {adapter.get_num_params():,}")
