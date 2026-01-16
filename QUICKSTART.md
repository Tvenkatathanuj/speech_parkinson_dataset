# Quick Start Guide

## 🚀 Getting Started with Your Novel Approach

This guide will help you implement and run experiments for your IEEE publication.

## Step 1: Environment Setup

```bash
# Navigate to project directory
cd "d:\SDP\speech\Parkinson-Patient-Speech-Dataset"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Step 2: Verify Dataset

```bash
# Check dataset structure
python -c "from pathlib import Path; print('Original dataset files:', len(list(Path('original-speech-dataset').rglob('*.wav')))); print('Denoised dataset files:', len(list(Path('denoised-speech-dataset').rglob('*.wav'))))"
```

Expected output: ~1160 files in each dataset

## Step 3: Data Preprocessing

```bash
# Create train/val/test splits
python -c "from src.data_preprocessing import DatasetPreprocessor; preprocessor = DatasetPreprocessor(); train, val, test = preprocessor.create_train_val_test_splits(train=0.7, val=0.15, test=0.15, split_by_patient=True)"
```

This will create:
- `processed/train.json`
- `processed/val.json`
- `processed/test.json`

## Step 4: Feature Extraction (Test)

```bash
# Test feature extraction on a sample file
python src/features.py
```

This will test both acoustic and prosodic feature extraction.

## Step 5: Model Testing

```bash
# Test the model architecture
python src/models/multitask_model.py
```

Expected output: Model parameter count and output shapes

## Implementation Roadmap

### Phase 1: Data Pipeline ✅ (Provided)
- [x] Data preprocessing
- [x] Feature extraction
- [x] Dataset pairing

### Phase 2: Model Implementation ✅ (Provided)
- [x] Conformer encoder
- [x] Multi-modal fusion
- [x] Multi-task heads
- [x] Wav2Vec 2.0 adapter

### Phase 3: Training (To Implement)
- [ ] Create training script
- [ ] Implement contrastive loss
- [ ] Add data augmentation
- [ ] Setup logging & checkpointing

### Phase 4: Evaluation (To Implement)
- [ ] Implement evaluation metrics
- [ ] Run baseline experiments
- [ ] Conduct ablation studies
- [ ] Generate result figures

### Phase 5: Paper Writing ✅ (Template Provided)
- [x] LaTeX template
- [ ] Fill in experimental results
- [ ] Create architecture diagrams
- [ ] Add result figures

## Creating the Training Script

Here's what you need to implement for training:

```python
# src/training/trainer.py (template)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import MultiTaskParkinsonsModel
from src.training.contrastive_loss import ContrastiveLoss

class ContrastiveTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Loss functions
        self.ctc_loss = nn.CTCLoss()
        self.severity_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Forward pass
            outputs = self.model(
                batch['acoustic'],
                batch['prosodic'],
                batch['lengths']
            )
            
            # Compute losses
            loss_ctc = self.ctc_loss(outputs['ctc_logits'], batch['transcript'])
            loss_severity = self.severity_loss(outputs['severity'], batch['severity'])
            loss_contrast = self.contrastive_loss(outputs['projection'], batch['pair_idx'])
            
            # Combined loss
            loss = (
                self.config.alpha * loss_ctc +
                self.config.beta * loss_severity +
                self.config.gamma * loss_contrast
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

## Creating the Dataset Class

```python
# src/data/dataset.py (to create)

import torch
from torch.utils.data import Dataset
import json
from src.features import MultiModalFeatureExtractor

class ParkinsonsDataset(Dataset):
    def __init__(self, split_file, feature_extractor):
        with open(split_file, 'r') as f:
            self.samples = json.load(f)
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract features
        original_features = self.feature_extractor.extract(sample['original_path'])
        denoised_features = self.feature_extractor.extract(sample['denoised_path'])
        
        return {
            'original': original_features,
            'denoised': denoised_features,
            'transcript': sample['transcript'],
            'patient': sample['patient']
        }
```

## Experiment Tracking

### Using Weights & Biases (Recommended)

```bash
# Install wandb
pip install wandb

# Login
wandb login

# In your training script:
import wandb
wandb.init(project="parkinsons-speech", name="multimodal-conformer")
```

### Using TensorBoard (Alternative)

```bash
# Start TensorBoard
tensorboard --logdir=results/logs
```

## Running Experiments

### 1. Baseline Experiment
```bash
python train.py --config experiments/baseline_config.yaml
```

### 2. Full Model Experiment
```bash
python train.py --config experiments/multimodal_config.yaml
```

### 3. Ablation Studies
```bash
# Without prosodic features
python train.py --config experiments/ablation/no_prosodic.yaml

# Without contrastive loss
python train.py --config experiments/ablation/no_contrastive.yaml
```

## Evaluation

```bash
# Evaluate on test set
python evaluate.py --checkpoint results/checkpoints/best_model.pt --test-file processed/test.json

# Generate paper figures
python scripts/generate_figures.py --results-dir results/
```

## Paper Compilation

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Solution**: Reduce batch size in config file
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Issue 2: Praat Installation Issues
**Solution**: Install binary separately
```bash
# Windows
choco install praat

# Or download from: https://www.fon.hum.uva.nl/praat/
```

### Issue 3: Wav2Vec 2.0 Download Slow
**Solution**: Pre-download model
```python
from transformers import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.save_pretrained("./models/wav2vec2-base")
```

## Performance Expectations

### Training Time (Single GPU)
- Baseline model: ~2 hours
- Full model: ~6 hours
- Ablation studies: ~2 hours each

### Expected Results (After Training)
- WER: 8-10% (should beat 18.5% baseline)
- Severity MAE: 0.4-0.5
- Clinical Accuracy: 90-95%

## Next Steps

1. ✅ **You are here**: Project setup complete
2. ⏳ **Next**: Implement training script
3. ⏳ **Then**: Run baseline experiments
4. ⏳ **Then**: Run full model experiments
5. ⏳ **Then**: Conduct ablation studies
6. ⏳ **Finally**: Write paper with results

## Resources

- **Conformer Paper**: https://arxiv.org/abs/2005.08100
- **Wav2Vec 2.0**: https://arxiv.org/abs/2006.11477
- **Contrastive Learning**: https://arxiv.org/abs/2002.05709
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Hugging Face**: https://huggingface.co/docs/transformers

## Getting Help

If you encounter issues:
1. Check the documentation in each module
2. Run unit tests: `pytest tests/`
3. Review example notebooks in `notebooks/`

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{parkinsons_multimodal_2026,
  title={Multi-Modal Deep Learning for Parkinsonian Speech Analysis: A Contrastive Learning Approach},
  author={Your Name},
  booktitle={IEEE ICASSP},
  year={2026}
}
```

---

**Good luck with your IEEE publication! 🎓🚀**
