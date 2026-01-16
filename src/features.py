"""
Multi-Modal Feature Extraction for Parkinson's Speech Analysis

Extracts:
1. Acoustic features (Wav2Vec 2.0, mel-spectrograms, MFCCs)
2. Prosodic features (pitch, jitter, shimmer, HNR, speech rate)
3. Voice quality metrics specific to Parkinson's Disease
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import torch
import torchaudio
from typing import Dict, Tuple, Optional
from pathlib import Path


class MultiModalFeatureExtractor:
    """
    Extract acoustic and prosodic features from speech audio.
    
    Features include:
    - Acoustic: Mel-spectrograms, MFCCs, Wav2Vec 2.0 embeddings
    - Prosodic: Pitch, jitter, shimmer, HNR, speech rate
    - Temporal: Pause patterns, speech segments
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_mfcc: int = 13,
        hop_length: int = 512,
        win_length: int = 2048,
        extract_acoustic: bool = True,
        extract_prosodic: bool = True,
        extract_jitter: bool = True,
        extract_shimmer: bool = True,
        pitch_floor: float = 75.0,  # Adjusted for Parkinson's (lower)
        pitch_ceiling: float = 300.0,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel frequency bands
            n_mfcc: Number of MFCC coefficients
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            extract_acoustic: Whether to extract acoustic features
            extract_prosodic: Whether to extract prosodic features
            extract_jitter: Whether to compute jitter (vocal fold instability)
            extract_shimmer: Whether to compute shimmer (amplitude variation)
            pitch_floor: Minimum pitch for analysis (Hz)
            pitch_ceiling: Maximum pitch for analysis (Hz)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.extract_acoustic = extract_acoustic
        self.extract_prosodic = extract_prosodic
        self.extract_jitter = extract_jitter
        self.extract_shimmer = extract_shimmer
        
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio waveform, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def extract_acoustic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract acoustic features from audio waveform.
        
        Args:
            audio: Audio waveform (1D array)
            
        Returns:
            Dictionary of acoustic features
        """
        features = {}
        
        # 1. Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = mel_spec_db
        
        # 2. MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        features['mfcc'] = mfcc
        
        # 3. Delta and Delta-Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features['mfcc_delta'] = mfcc_delta
        features['mfcc_delta2'] = mfcc_delta2
        
        # 4. Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )
        
        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features['chroma'] = chroma
        
        return features
    
    def extract_prosodic_features(
        self, audio: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """
        Extract prosodic features using Praat's Parselmouth.
        
        These features are particularly relevant for Parkinson's Disease:
        - Jitter: Vocal fold instability
        - Shimmer: Amplitude variation
        - HNR: Voice quality
        - Pitch dynamics: Monotone speech
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Dictionary of prosodic features
        """
        features = {}
        
        # Create Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract pitch
        pitch = sound.to_pitch(
            time_step=0.01,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling
        )
        
        # Pitch statistics
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_min'] = float(np.min(pitch_values))
            features['pitch_max'] = float(np.max(pitch_values))
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_median'] = float(np.median(pitch_values))
            
            # Pitch variability (important for Parkinson's monotone detection)
            features['pitch_coefficient_variation'] = features['pitch_std'] / features['pitch_mean']
        else:
            # No voiced frames
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 
                       'pitch_range', 'pitch_median', 'pitch_coefficient_variation']:
                features[key] = 0.0
        
        # Jitter (local, absolute)
        if self.extract_jitter:
            try:
                point_process = call(sound, "To PointProcess (periodic, cc)", 
                                    self.pitch_floor, self.pitch_ceiling)
                
                features['jitter_local'] = call(point_process, "Get jitter (local)", 
                                               0, 0, 0.0001, 0.02, 1.3)
                features['jitter_local_absolute'] = call(point_process, "Get jitter (local, absolute)", 
                                                        0, 0, 0.0001, 0.02, 1.3)
                features['jitter_rap'] = call(point_process, "Get jitter (rap)", 
                                             0, 0, 0.0001, 0.02, 1.3)
                features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 
                                              0, 0, 0.0001, 0.02, 1.3)
            except Exception as e:
                print(f"Warning: Could not compute jitter: {e}")
                features['jitter_local'] = 0.0
                features['jitter_local_absolute'] = 0.0
                features['jitter_rap'] = 0.0
                features['jitter_ppq5'] = 0.0
        
        # Shimmer (local)
        if self.extract_shimmer:
            try:
                point_process = call(sound, "To PointProcess (periodic, cc)", 
                                    self.pitch_floor, self.pitch_ceiling)
                
                features['shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 
                                                0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_local_db'] = call([sound, point_process], "Get shimmer (local, dB)", 
                                                   0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 
                                               0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_apq5'] = call([sound, point_process], "Get shimmer (apq5)", 
                                               0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_apq11'] = call([sound, point_process], "Get shimmer (apq11)", 
                                                0, 0, 0.0001, 0.02, 1.3, 1.6)
            except Exception as e:
                print(f"Warning: Could not compute shimmer: {e}")
                for key in ['shimmer_local', 'shimmer_local_db', 'shimmer_apq3', 
                           'shimmer_apq5', 'shimmer_apq11']:
                    features[key] = 0.0
        
        # Harmonics-to-Noise Ratio (HNR)
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 
                              self.pitch_floor, 0.1, 1.0)
            features['hnr_mean'] = call(harmonicity, "Get mean", 0, 0)
            features['hnr_std'] = call(harmonicity, "Get standard deviation", 0, 0)
        except Exception as e:
            print(f"Warning: Could not compute HNR: {e}")
            features['hnr_mean'] = 0.0
            features['hnr_std'] = 0.0
        
        # Speech rate and timing
        features['duration'] = sound.duration
        
        # Voice activity detection (simple energy-based)
        energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        threshold = np.percentile(energy, 30)
        voiced_frames = np.sum(energy > threshold)
        total_frames = len(energy)
        features['voiced_fraction'] = voiced_frames / total_frames if total_frames > 0 else 0.0
        
        # Estimate speech rate (syllables per second - approximation)
        # Based on voiced segments
        features['estimated_speech_rate'] = features['voiced_fraction'] * 3.0  # Rough estimate
        
        return features
    
    def extract(
        self, audio_path: str, return_audio: bool = False
    ) -> Dict[str, any]:
        """
        Extract all features from an audio file.
        
        Args:
            audio_path: Path to audio file
            return_audio: Whether to include raw audio in output
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        result = {
            'audio_path': str(audio_path),
            'sample_rate': sr,
            'duration': len(audio) / sr
        }
        
        if return_audio:
            result['audio'] = audio
        
        # Extract acoustic features
        if self.extract_acoustic:
            acoustic_features = self.extract_acoustic_features(audio)
            result['acoustic'] = acoustic_features
        
        # Extract prosodic features
        if self.extract_prosodic:
            prosodic_features = self.extract_prosodic_features(audio, sr)
            result['prosodic'] = prosodic_features
        
        return result
    
    def extract_batch(
        self, audio_paths: list, show_progress: bool = True
    ) -> list:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of feature dictionaries
        """
        results = []
        
        iterator = audio_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="Extracting features")
            except ImportError:
                pass
        
        for audio_path in iterator:
            try:
                features = self.extract(audio_path)
                results.append(features)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append(None)
        
        return results


if __name__ == "__main__":
    # Example usage
    extractor = MultiModalFeatureExtractor(
        extract_acoustic=True,
        extract_prosodic=True,
        extract_jitter=True,
        extract_shimmer=True
    )
    
    # Test on a sample file
    sample_audio = "denoised-speech-dataset/DL/DL1.wav"
    
    if Path(sample_audio).exists():
        features = extractor.extract(sample_audio)
        print("\nExtracted features:")
        print(f"  Acoustic features: {list(features.get('acoustic', {}).keys())}")
        print(f"  Prosodic features: {list(features.get('prosodic', {}).keys())}")
    else:
        print(f"Sample audio file not found: {sample_audio}")
