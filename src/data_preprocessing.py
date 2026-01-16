"""
Data Preprocessing Module for Parkinson's Speech Dataset

This module handles:
- Loading original and denoised speech datasets
- Creating train/val/test splits
- Pairing original and denoised audio for contrastive learning
- Managing CSV metadata files
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetPreprocessor:
    """
    Preprocessor for Parkinson's Disease speech dataset.
    
    Handles both original and denoised speech datasets, creating paired
    samples for contrastive learning.
    """
    
    def __init__(
        self,
        original_dir: str = "original-speech-dataset",
        denoised_dir: str = "denoised-speech-dataset",
        output_dir: str = "processed",
        seed: int = 42
    ):
        """
        Initialize the dataset preprocessor.
        
        Args:
            original_dir: Path to original speech dataset
            denoised_dir: Path to denoised speech dataset
            output_dir: Path to save processed metadata
            seed: Random seed for reproducibility
        """
        self.original_dir = Path(original_dir)
        self.denoised_dir = Path(denoised_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        random.seed(seed)
        np.random.seed(seed)
        
        # Patient folders
        self.patients = ["DL", "LW", "Tessi", "Faces", "emma"]
        
    def load_csv_files(self) -> List[Dict]:
        """
        Load all CSV metadata files from the dataset.
        
        Returns:
            List of sample dictionaries with audio paths and transcripts
        """
        samples = []
        
        for patient in self.patients:
            # Check for CSV files in both original and denoised directories
            for dataset_type, base_dir in [
                ("original", self.original_dir),
                ("denoised", self.denoised_dir)
            ]:
                csv_files = list(base_dir.glob(f"{patient}/**/*.csv"))
                
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    
                    # Handle different CSV formats
                    if 'wav_filename' in df.columns:
                        path_col = 'wav_filename'
                    elif 'audio_filepath' in df.columns:
                        path_col = 'audio_filepath'
                    else:
                        continue
                        
                    for _, row in df.iterrows():
                        audio_path = row[path_col]
                        transcript = row['transcript'].strip()
                        
                        # Convert to relative path
                        audio_path = self._normalize_path(audio_path, base_dir, patient)
                        
                        samples.append({
                            'patient': patient,
                            'audio_path': str(audio_path),
                            'transcript': transcript,
                            'dataset_type': dataset_type,
                            'duration': None  # To be filled later
                        })
        
        print(f"Loaded {len(samples)} samples from CSV files")
        return samples
    
    def load_txt_transcripts(self) -> Dict[str, str]:
        """
        Load individual .txt transcript files.
        
        Returns:
            Dictionary mapping audio filename to transcript
        """
        transcripts = {}
        
        for patient in self.patients:
            # Load from denoised directory (has same structure)
            txt_files = list(self.denoised_dir.glob(f"{patient}/**/*.txt"))
            
            for txt_file in txt_files:
                if txt_file.name == "ref.lst":  # Skip reference list files
                    continue
                    
                with open(txt_file, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    
                # Map to corresponding audio file
                audio_name = txt_file.stem  # filename without extension
                transcripts[audio_name] = transcript
        
        print(f"Loaded {len(transcripts)} transcripts from .txt files")
        return transcripts
    
    def create_paired_dataset(self) -> List[Dict]:
        """
        Create paired dataset of original and denoised audio.
        
        Returns:
            List of paired samples for contrastive learning
        """
        # Load CSV samples
        csv_samples = self.load_csv_files()
        txt_transcripts = self.load_txt_transcripts()
        
        # Group by patient and filename
        original_samples = defaultdict(dict)
        denoised_samples = defaultdict(dict)
        
        for sample in csv_samples:
            patient = sample['patient']
            filename = Path(sample['audio_path']).stem
            
            if sample['dataset_type'] == 'original':
                original_samples[patient][filename] = sample
            else:
                denoised_samples[patient][filename] = sample
        
        # Also scan for audio files not in CSV
        for dataset_type, base_dir, sample_dict in [
            ('original', self.original_dir, original_samples),
            ('denoised', self.denoised_dir, denoised_samples)
        ]:
            for patient in self.patients:
                wav_files = list(base_dir.glob(f"{patient}/**/*.wav"))
                
                for wav_file in wav_files:
                    filename = wav_file.stem
                    
                    # Skip if already in CSV
                    if filename in sample_dict[patient]:
                        continue
                    
                    # Try to get transcript from txt file
                    transcript = txt_transcripts.get(filename, "")
                    if not transcript:
                        print(f"Warning: No transcript for {filename}")
                        continue
                    
                    sample_dict[patient][filename] = {
                        'patient': patient,
                        'audio_path': str(wav_file),
                        'transcript': transcript,
                        'dataset_type': dataset_type,
                        'duration': None
                    }
        
        # Create paired samples
        paired_samples = []
        
        for patient in self.patients:
            original_files = set(original_samples[patient].keys())
            denoised_files = set(denoised_samples[patient].keys())
            common_files = original_files & denoised_files
            
            print(f"Patient {patient}: {len(common_files)} paired samples")
            
            for filename in common_files:
                original = original_samples[patient][filename]
                denoised = denoised_samples[patient][filename]
                
                paired_samples.append({
                    'patient': patient,
                    'filename': filename,
                    'original_path': original['audio_path'],
                    'denoised_path': denoised['audio_path'],
                    'transcript': original['transcript'],
                    'duration': None
                })
        
        print(f"\nTotal paired samples: {len(paired_samples)}")
        return paired_samples
    
    def create_train_val_test_splits(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        split_by_patient: bool = True
    ) -> Tuple[List, List, List]:
        """
        Create train/validation/test splits.
        
        Args:
            train: Proportion of training data
            val: Proportion of validation data
            test: Proportion of test data
            split_by_patient: If True, split by patient (for generalization test)
        
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        paired_samples = self.create_paired_dataset()
        
        if split_by_patient:
            # Split by patient for better generalization evaluation
            patients = list(set(s['patient'] for s in paired_samples))
            random.shuffle(patients)
            
            n_patients = len(patients)
            n_train = int(n_patients * train)
            n_val = int(n_patients * val)
            
            train_patients = patients[:n_train]
            val_patients = patients[n_train:n_train + n_val]
            test_patients = patients[n_train + n_val:]
            
            train_samples = [s for s in paired_samples if s['patient'] in train_patients]
            val_samples = [s for s in paired_samples if s['patient'] in val_patients]
            test_samples = [s for s in paired_samples if s['patient'] in test_patients]
            
            print(f"\nPatient-based split:")
            print(f"  Train patients: {train_patients} ({len(train_samples)} samples)")
            print(f"  Val patients: {val_patients} ({len(val_samples)} samples)")
            print(f"  Test patients: {test_patients} ({len(test_samples)} samples)")
        else:
            # Random split
            train_samples, temp_samples = train_test_split(
                paired_samples, test_size=(val + test), random_state=self.seed
            )
            val_samples, test_samples = train_test_split(
                temp_samples, test_size=test / (val + test), random_state=self.seed
            )
            
            print(f"\nRandom split:")
            print(f"  Train: {len(train_samples)} samples")
            print(f"  Val: {len(val_samples)} samples")
            print(f"  Test: {len(test_samples)} samples")
        
        # Save splits
        self._save_split(train_samples, "train")
        self._save_split(val_samples, "val")
        self._save_split(test_samples, "test")
        
        return train_samples, val_samples, test_samples
    
    def _normalize_path(self, path: str, base_dir: Path, patient: str) -> Path:
        """Normalize audio path to relative path from project root."""
        path = Path(path)
        
        # If absolute path, make relative
        if path.is_absolute():
            # Find the patient folder in the path
            try:
                parts = path.parts
                patient_idx = parts.index(patient)
                relative_parts = parts[patient_idx:]
                return base_dir / Path(*relative_parts)
            except ValueError:
                pass
        
        return path
    
    def _save_split(self, samples: List[Dict], split_name: str):
        """Save dataset split to JSON file."""
        output_file = self.output_dir / f"{split_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(samples)} samples to {output_file}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DatasetPreprocessor()
    train, val, test = preprocessor.create_train_val_test_splits(
        train=0.7, val=0.15, test=0.15,
        split_by_patient=True
    )
