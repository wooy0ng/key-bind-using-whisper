from dataloader import load_sound_data

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from transformers import WhisperModel, WhisperTokenizer, WhisperForConditionalGeneration

class CryptoEncoder(nn.Module):
    def __init__(self):
        super(CryptoEncoder, self).__init__()
        self.in_features = 384      
        self.latent_dim = 128
    
        self.model = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(256, self.latent_dim)
        self.logvar = nn.Linear(256, self.latent_dim)
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), self.latent_dim)))).to(
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
        self.out_features = 384
        self.latent_dim = 128       # key size가 되지 않을까?

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, self.out_features),
            nn.Tanh()
        )
    
    def forward(self, z):
        q = self.model(z)   # (batch, sequence, features)
        return q

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.latent_dim = 128       # key size
        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.Dropout(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, z):
        layer1 = self.layer1(z)
        layer2 = self.layer2(layer1)

        return layer2

    def mean_pooling(self, last_hidden_state):
        summarize = torch.sum(last_hidden_state, 1)
        return summarize / last_hidden_state.size(1)


class CryptoModel(nn.Module):
    def __init__(self):
        super(CryptoModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path='openai/whisper-tiny'
        ).to(self.device)

        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path='openai/whisper-tiny'
        )
        self.decoder_input_ids = self._get_decoder_input_ids()
        
        self.whisper_model = self._freeze(self.whisper_model)
        
        self.crypto_encoder = CryptoEncoder()
        self.crypto_decoder = CryptoDecoder()
        self.discriminator = Discriminator()

    def _get_decoder_input_ids(self) -> torch.Tensor:
        compression_token = '[compression]'
        input_tokens = f'<|startoftranscript|><|ko|><|transcribe|><|notimestamps|> {compression_token}<|endoftext|>'
        input_token_ids = self.whisper_tokenizer(input_tokens)['input_ids']
        input_token_ids = input_token_ids[2: -1]

        decoder_input_ids = torch.tensor(input_token_ids).to(device=self.device, dtype=torch.long)
        return decoder_input_ids
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def mean_pooling(self, logits):
        summarize = torch.sum(logits, 1)
        return summarize / logits.size(1)
    
    def print_log(self, stage='train', **kwargs):
        if stage == 'train':
            print(f"\n{stage} epoch : {kwargs['epoch']+1}, iteration : {kwargs['iteration']}")
            print(f"{stage} g_loss : {kwargs['g_loss']}")
            print(f"{stage} d_loss : {kwargs['d_loss']}")
            
    def forward(self, inputs):
        out = self.whisper_model(inputs, decoder_input_ids=self.decoder_input_ids.repeat(inputs.size(0), 1))
        logits = out.logits 
        logits = logits[:, 4:-1, :] # TODO: logits 값에서 의미 있는 feature 추출하기 (batch, token_size)
        whisper_output = self.mean_pooling(logits)

        encoded = self.crypto_encoder(whisper_output)
        decoded = self.crypto_decoder(encoded)
        
        return {'embedding_vector': whisper_output, 'encoded': encoded, 'decoded': decoded}

    def trainer(self, config, dataset: Dataset):
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        master_key = utils.make_random_key(key_size=128)

        optimizer_G = torch.optim.Adam(
            itertools.chain(self.crypto_encoder.parameters(), self.crypto_decoder.parameters()),
            lr=config.lr
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr
        )

        pixelwise_loss = torch.nn.L1Loss()
        classification_loss = torch.nn.BCELoss()

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for epoch in range(config.epoch):
            for idx, (path, inputs) in enumerate(dataloader):
                # Adversarial ground truth
                valid = Variable(Tensor(inputs.size(0), 1).fill_(1.0), requires_grad=False)  
                fake = Variable(Tensor(inputs.size(0), 1).fill_(0.0), requires_grad=False)   # (b, 1)

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
                g_loss = 0.001 * classification_loss(self.discriminator(encoded), valid) \
                        + 0.999 * pixelwise_loss(decoded, embedding_vector)
                
                # backward
                g_loss.backward()
                optimizer_G.step()

                # -------------------
                # train discriminator
                # -------------------

                optimizer_D.zero_grad()
                
                # sample noize as discriminator ground truth 
                z = Variable(Tensor(np.random.normal(0, 1, (inputs.size(0), master_key.size(0)))))
                
                real_loss = classification_loss(self.discriminator(z), valid)
                fake_loss = classification_loss(self.discriminator(encoded.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)        # 학습에 큰 비중을 차지하는 loss

                d_loss.backward()
                optimizer_D.step()
                
                train_logs = {'epoch': epoch, 'iteration': idx, 'g_loss': g_loss, 'd_loss': d_loss}
                self.print_log(stage='train', **train_logs)
                
        # save model
        torch.save(self.state_dict(), config.save_path)
        return
    
    @torch.no_grad()
    def inference(self, config, wav_path):
        sound_data = load_sound_data(wav_path, return_mel=True)
        self.load_state_dict(torch.load(config.load_path))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
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
