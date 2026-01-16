"""
Multi-Task Model for Parkinson's Speech Analysis

Combines:
1. Speech recognition (CTC decoder)
2. Dysarthria severity assessment (regression)
3. Contrastive learning (original vs denoised pairs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.models.conformer import ConformerEncoder
from src.models.fusion import MultiModalFusion
from src.models.wav2vec_adapter import Wav2VecAdapter


class MultiTaskParkinsonsModel(nn.Module):
    """
    Multi-task model for Parkinson's speech analysis.
    
    Architecture:
    - Wav2Vec 2.0 feature extraction
    - Conformer encoder for temporal modeling
    - Multi-modal fusion (acoustic + prosodic)
    - Multi-task heads:
        * CTC decoder for transcription
        * Regression head for severity assessment
        * Projection head for contrastive learning
    """
    
    def __init__(
        self,
        encoder_type: str = "conformer",
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        vocab_size: int = 32,  # Character-level encoding
        num_prosodic_features: int = 25,
        use_prosodic_fusion: bool = True,
        dropout: float = 0.1,
        use_wav2vec: bool = True,
        wav2vec_model: str = "facebook/wav2vec2-base",
        freeze_wav2vec: bool = False,
        projection_dim: int = 256,  # For contrastive learning
    ):
        """
        Initialize the multi-task model.
        
        Args:
            encoder_type: Type of encoder ("conformer" or "transformer")
            num_layers: Number of encoder layers
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            vocab_size: Size of vocabulary for CTC decoder
            num_prosodic_features: Number of prosodic features
            use_prosodic_fusion: Whether to use prosodic feature fusion
            dropout: Dropout rate
            use_wav2vec: Whether to use Wav2Vec 2.0 pre-training
            wav2vec_model: Name of Wav2Vec 2.0 model
            freeze_wav2vec: Whether to freeze Wav2Vec 2.0 parameters
            projection_dim: Dimension for contrastive learning projection
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.use_prosodic_fusion = use_prosodic_fusion
        self.use_wav2vec = use_wav2vec
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        
        # 1. Wav2Vec 2.0 Feature Extractor (optional)
        if use_wav2vec:
            self.wav2vec_adapter = Wav2VecAdapter(
                model_name=wav2vec_model,
                freeze=freeze_wav2vec,
                output_dim=hidden_dim
            )
            acoustic_input_dim = hidden_dim
        else:
            # Direct mel-spectrogram input
            acoustic_input_dim = 80  # n_mels
            self.input_projection = nn.Linear(acoustic_input_dim, hidden_dim)
        
        # 2. Conformer/Transformer Encoder
        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                input_dim=hidden_dim,
                num_layers=num_layers,
                d_model=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
        else:
            # Standard Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Prosodic Feature Encoder (if used)
        if use_prosodic_fusion:
            self.prosodic_encoder = nn.Sequential(
                nn.Linear(num_prosodic_features, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Multi-modal fusion
            self.fusion = MultiModalFusion(
                acoustic_dim=hidden_dim,
                prosodic_dim=hidden_dim,
                output_dim=hidden_dim,
                fusion_type="attention"
            )
        
        # 4. Multi-Task Heads
        
        # 4a. CTC Decoder for Speech Recognition
        self.ctc_decoder = nn.Linear(hidden_dim, vocab_size)
        
        # 4b. Severity Assessment Head (regression)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),  # Severity score [0, 4]
            nn.Sigmoid()  # Output in range [0, 1], scale to [0, 4] later
        )
        
        # 4c. Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # 4d. Domain Classifier for Domain Adversarial Training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Original vs Denoised
        )
    
    def forward(
        self,
        acoustic_input: torch.Tensor,
        prosodic_features: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            acoustic_input: Acoustic features [batch, time, features] or raw audio [batch, time]
            prosodic_features: Prosodic features [batch, num_features]
            input_lengths: Lengths of sequences in batch
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing outputs from all task heads
        """
        batch_size = acoustic_input.size(0)
        
        # 1. Extract acoustic features
        if self.use_wav2vec:
            # acoustic_input should be raw waveform [batch, time]
            acoustic_features = self.wav2vec_adapter(acoustic_input)
        else:
            # acoustic_input is mel-spectrogram [batch, time, n_mels]
            acoustic_features = self.input_projection(acoustic_input)
        
        # 2. Encode with Conformer/Transformer
        encoded = self.encoder(acoustic_features)  # [batch, time, hidden_dim]
        
        # 3. Fuse with prosodic features (if provided)
        if self.use_prosodic_fusion and prosodic_features is not None:
            prosodic_encoded = self.prosodic_encoder(prosodic_features)  # [batch, hidden_dim]
            
            # Expand prosodic features to match temporal dimension
            prosodic_expanded = prosodic_encoded.unsqueeze(1).expand(-1, encoded.size(1), -1)
            
            # Fusion
            fused = self.fusion(encoded, prosodic_expanded)  # [batch, time, hidden_dim]
        else:
            fused = encoded
        
        # 4. Global pooling for non-sequential tasks
        # Use mean pooling over time dimension
        if input_lengths is not None:
            # Masked mean pooling
            mask = torch.arange(fused.size(1), device=fused.device)[None, :] < input_lengths[:, None]
            masked_fused = fused * mask.unsqueeze(-1)
            pooled = masked_fused.sum(dim=1) / input_lengths.unsqueeze(-1).float()
        else:
            pooled = fused.mean(dim=1)  # [batch, hidden_dim]
        
        # 5. Task-specific heads
        outputs = {}
        
        # 5a. CTC logits for transcription
        ctc_logits = self.ctc_decoder(fused)  # [batch, time, vocab_size]
        outputs['ctc_logits'] = F.log_softmax(ctc_logits, dim=-1)
        
        # 5b. Severity score
        severity = self.severity_head(pooled) * 4.0  # Scale to [0, 4]
        outputs['severity'] = severity.squeeze(-1)
        
        # 5c. Contrastive projection
        projection = self.projection_head(pooled)
        projection = F.normalize(projection, p=2, dim=-1)  # L2 normalization
        outputs['projection'] = projection
        
        # 5d. Domain classification
        domain_logits = self.domain_classifier(pooled)
        outputs['domain_logits'] = domain_logits
        
        # Optional: return embeddings
        if return_embeddings:
            outputs['encoded'] = encoded
            outputs['fused'] = fused
            outputs['pooled'] = pooled
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = MultiTaskParkinsonsModel(
        encoder_type="conformer",
        num_layers=6,  # Smaller for testing
        hidden_dim=256,
        use_prosodic_fusion=True,
        use_wav2vec=False  # Test without Wav2Vec
    )
    
    # Create dummy inputs
    batch_size = 2
    time_steps = 100
    n_mels = 80
    num_prosodic = 25
    
    acoustic = torch.randn(batch_size, time_steps, n_mels)
    prosodic = torch.randn(batch_size, num_prosodic)
    lengths = torch.tensor([100, 80])
    
    # Forward pass
    outputs = model(acoustic, prosodic, lengths, return_embeddings=True)
    
    print(f"\nModel parameters: {model.get_num_params():,}")
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
