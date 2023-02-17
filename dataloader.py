import os
import torch
import whisper
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset
from collections import defaultdict
from torchaudio.utils import download_asset
from audiomentations import AddGaussianNoise
from pathlib import Path
from typing import Union, Tuple, List, Dict

r""" 
download from torchaudio 
1. 방 잔향 증강
2. 

"""
SAMPLING_RATE = 16_000
RIR: str = download_asset('tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav')

def load_sound_data(wav_path: Union[str, List[str]], return_mel=False) -> Dict[str, List]:
    if type(wav_path) == str:
        paths = [wav_path]
    else:
        paths = [os.path.join(Path(__file__).parent, path) for path in wav_path]

    dataset = defaultdict(list)
    for path in paths:
        audio = whisper.load_audio(path)
        dataset['path'].append(path)
        dataset['audio'].append(torch.Tensor(whisper.pad_or_trim(audio)))

    if return_mel:
        for i in range(len(dataset['path'])):
            mel = whisper.log_mel_spectrogram(dataset['audio'][i])  # (80, 3000)
            mel = np.expand_dims(mel, 0)                            # (1, 80, 3000)
            dataset['mel'].append(torch.Tensor(mel))
    return dataset

def augment_sound_data(wav_path: str) -> Dict:
    def augment_sound_rir(audio: torch.Tensor) -> List[torch.Tensor]:
        rir = torch.Tensor(whisper.load_audio(RIR)).unsqueeze(0)
        rir = rir[:, int(SAMPLING_RATE * 1.01) : int(SAMPLING_RATE * 1.3)]
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, (1,))

        aug = F.pad(audio, (rir.shape[1]-1, 0))
        aug = F.conv1d(aug[None, ...], rir[None, ...])[0]
        aug = aug.squeeze()
        return aug
    
    def augment_sound_codec(audio: torch.Tensor, config_idx) -> List[torch.Tensor]:
        configs = [
            {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
            {"format": "wav", "compression": -9},
        ]
        
        param = configs[config_idx]
        audio = audio.unsqueeze(0)
        aug = torchaudio.functional.apply_codec(audio, 16000, **param)
        aug = aug.squeeze()
        return aug

    def augment_sound_gaussian_noize(audio: torch.Tensor) -> List[torch.Tensor]:
        augment = AddGaussianNoise(min_amplitude=1e-3, max_amplitude=15e-3, p=1.0)
        aug = augment(audio, 16000)
        aug = aug.squeeze()
        return aug

    dataset = load_sound_data(wav_path)

    aug_dataset = defaultdict(list)
    for path, audio in zip(dataset['path'], dataset['audio']):
        audios = []    
        audios.append(augment_sound_rir(audio))               
        audios.append(augment_sound_codec(audio, 0))
        audios.append(augment_sound_codec(audio, 1))
        audios.append(augment_sound_gaussian_noize(audio))
        
        for audio in audios:
            aug_dataset['path'].append(path)
            aug_dataset['audio'].append(audio)
        
    dataset['path'] += aug_dataset['path']
    dataset['audio'] += aug_dataset['audio']
    
    return dataset


class SoundDataset(Dataset):
    def __init__(self, dataset: Dict):
        self.dataset = dataset
        self.dataset['audio'] = [self.to_mel_spectrogram(audio) for audio in self.dataset['audio']]
    
    def __len__(self):
        return len(self.dataset['audio'])

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        return self.dataset['path'][idx], torch.Tensor(self.dataset['audio'][idx])
        
    def to_mel_spectrogram(self, audio):
        mel = whisper.log_mel_spectrogram(np.array(audio, dtype=np.float32))    # (80, 3000)
        return mel