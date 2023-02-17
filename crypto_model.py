from dataloader import load_sound_data

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from transformers import WhisperModel

class CryptoEncoder(nn.Module):
    def __init__(self):
        super(CryptoEncoder, self).__init__()
        self.in_features = 3000 // 2        
        self.latent_dim = 256               # key size * 2
    
        self.model = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(512, self.latent_dim)
        self.logvar = nn.Linear(512, self.latent_dim)
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1), self.latent_dim)))).to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        z = sampled_z * std + mu
        return z

    def forward(self, inputs):
        out = self.model(inputs)
        mu = self.mu(out)
        logvar = self.logvar(out)

        z = self.reparameterization(mu, logvar)
        return z
        

class CryptoDecoder(nn.Module):
    def __init__(self):
        super(CryptoDecoder, self).__init__()
        self.out_features = 3000 // 2
        self.latent_dim = 256       # key size * 2

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, self.out_features),
            nn.Tanh()
        )
    
    def forward(self, z):
        q = self.model(z)   # (batch, sequence, features)
        return q

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.latent_dim = 256       # key size * 2
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.mean_pooling(z)  # (b, f)
        out = self.model(out)
        return out

    def mean_pooling(self, last_hidden_state):
        summarize = torch.sum(last_hidden_state, 1)
        return summarize / last_hidden_state.size(1)


class CryptoModel(nn.Module):
    def __init__(self):
        super(CryptoModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.whisper_model = WhisperModel.from_pretrained(
            pretrained_model_name_or_path='openai/whisper-tiny'
        ).to(self.device)

        self.whisper_model = self._freeze(self.whisper_model)
        
        self.crypto_encoder = CryptoEncoder()
        self.crypto_decoder = CryptoDecoder()
        self.discriminator = Discriminator()
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def print_log(self, stage='train', **kwargs):
        if stage == 'train':
            print(f"\n{stage} epoch : {kwargs['epoch']+1}, iteration : {kwargs['iteration']}")
            print(f"{stage} g_loss : {kwargs['g_loss']}")
            print(f"{stage} d_loss : {kwargs['d_loss']}")
            
    def forward(self, inputs):
        whisper_out = self.whisper_model.encoder(inputs).last_hidden_state
        whisper_out = whisper_out.permute(0, 2, 1)

        encoded = self.crypto_encoder(whisper_out)
        decoded = self.crypto_decoder(encoded)
        
        return {'embedding_vector': whisper_out, 'encoded': encoded, 'decoded': decoded}

    def trainer(self, config, dataset: Dataset):
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        master_key = utils.make_random_key(key_size=128)

        optimizer_G = torch.optim.Adam(
            itertools.chain(self.crypto_encoder.parameters(), self.crypto_decoder.parameters()),
            lr=config.lr,
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr
        )

        adversarial_loss = torch.nn.BCELoss()
        pixelwise_loss = torch.nn.L1Loss()

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for epoch in range(config.epoch):
            for idx, (path, inputs) in enumerate(dataloader):
                # Adversarial ground truth
                valid = master_key.unsqueeze(-1).type(Tensor)
                valid = valid.repeat(inputs.size(0), 1, 1)
                valid.requires_grad = False
                
                fake = torch.randint(0, 2, (inputs.size(0), master_key.size(0), 1)).type(Tensor)

                inputs = Variable(inputs.type(Tensor))
                outputs = self(inputs)
                
                # ---------------
                # train generator
                # ---------------

                optimizer_G.zero_grad()
                
                embedding_vector = outputs['embedding_vector']
                encoded = outputs['encoded']
                decoded = outputs['decoded']
                
                # calculate adversarial loss
                g_loss = 0.001 * adversarial_loss(self.discriminator(encoded).unsqueeze(-1), valid) \
                        + 0.999 * pixelwise_loss(decoded, embedding_vector)

                # backward
                g_loss.backward()
                optimizer_G.step()

                # -------------------
                # train discriminator
                # -------------------

                optimizer_D.zero_grad()
                
                # sample noise as discriminator ground truth
                z = Variable(Tensor(np.random.normal(0, 1, encoded.shape)))
                
                real_loss = adversarial_loss(self.discriminator(z).unsqueeze(-1), valid)
                fake_loss = adversarial_loss(self.discriminator(encoded.detach()).unsqueeze(-1), fake)
                d_loss = 0.01 * fake_loss + 0.99 * real_loss        # 학습에 큰 비중을 차지하는 loss

                d_loss.backward()
                optimizer_D.step()
                
                train_logs = {'epoch': epoch, 'iteration': idx, 'g_loss': g_loss, 'd_loss': d_loss}
                self.print_log(stage='train', **train_logs)
                
        # save model
        torch.save(self.state_dict(), config.save_path)
        return

    def inference(self, config, wav_path):
        sound_data = load_sound_data(wav_path, return_mel=True)
        self.load_state_dict(torch.load(config.load_path))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        with torch.no_grad():
            inputs = sound_data['mel'][0]
            outputs = self(inputs.type(Tensor))
            encoded = outputs['encoded']

            predicted = self.discriminator(encoded).squeeze()

            # round & compare 
            master_key = utils.make_random_key(key_size=128).cpu().detach()
            predicted_key = torch.round(predicted).cpu().detach()

            print('master key : \t', utils.bit_to_string(master_key))
            print('predicted key : ', utils.bit_to_string(predicted_key))
            
        return
