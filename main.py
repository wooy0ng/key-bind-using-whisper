from dataloader import augment_sound_data, SoundDataset
from crypto_model import CryptoModel

import numpy as np
import whisper
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

"""
TODO:
    TODO : sound 데이터 증강(O)
    TODO : Dataset 제작 (O)
    TODO : inference 제작 (O)
    TODO : trainer 제작 (O)
        TODO: last_hidden_state 차원 줄이기 (b, s, f) -> (b, f)
            TODO: Mean Pooling 추가 (X) [폐기]
        TODO: 손실함수 정의
            TODO: 적대적 생성 신경망 (O)
                TODO: key generate 모델 학습을 더 어렵게 만들기 (O)
    TODO : 데이터셋 위치 바꾸기
"""

def main(user_input, stage='train'):
    config = OmegaConf.load('config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load audio data
    dataset = augment_sound_data('dataset/train')
    dataset = SoundDataset(dataset)
    
    model = CryptoModel().to(device)
    if stage == 'train':
        model.trainer(getattr(config, 'training'), dataset, user_input)
    elif stage == 'test':    
        model.test(getattr(config, 'inference'), 'dataset/test/compare2.wav', user_input)
    elif stage == 'inference':
        model.inference(getattr(config, 'inference'), 'dataset/test/sample1.wav', user_input)    
    return

if __name__ == "__main__":
    main(stage='test', user_input='안녕하세요')