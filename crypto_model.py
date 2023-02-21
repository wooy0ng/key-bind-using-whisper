from dataloader import load_sound_data

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import pickle as pkl

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
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(256, self.latent_dim)
        self.logvar = nn.Linear(256, self.latent_dim)
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, mu.size()))).to(
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
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, self.out_features),
        )
    
    def forward(self, z):
        q = self.model(z)   # (batch, sequence, features)
        return q

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.latent_dim = 128       # key size

        self.fc1 = nn.Linear(384, self.latent_dim)
        self.w = torch.nn.Parameter(data=torch.Tensor(self.latent_dim, self.latent_dim), requires_grad=True)
        self._init_w_param()

    def _init_w_param(self):
        nn.init.xavier_normal_(self.w)

    def limit_param_range(self):
        self.w.data.clamp_(-10., 10.)

    def forward(self, z, negative_key):
        hidden_states = z
        hidden_states = self.fc1(hidden_states)  # (b, 128)
        
        hidden_states = hidden_states * negative_key
        hidden_states = hidden_states @ self.w.T
        
        hidden_states = torch.tanh(hidden_states)   # -1 ~ 1
        return hidden_states


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
        self.whisper_model = self._freeze(self.whisper_model)
        
        self.decoder_input_ids = self._get_decoder_input_ids()
        self.whisper_model.forced_decoder_ids = self.decoder_input_ids['forced_decoder_ids']
        
        self.crypto_encoder = CryptoEncoder()
        self.crypto_decoder = CryptoDecoder()
        self.discriminator = Discriminator()

    def _get_decoder_input_ids(self) -> torch.Tensor:
        forced_decoder_ids = self.whisper_tokenizer.get_decoder_prompt_ids(language='ko', task='transcribe')
        decoded_input_ids = [50258] + [ids for _, ids in forced_decoder_ids]
        return {
            'forced_decoder_ids': forced_decoder_ids,
            'decoder_input_ids': torch.tensor(decoded_input_ids).to(self.device)
        }
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def print_log(self, stage='train', **kwargs):
        if stage == 'train':
            print(f"\n{stage} epoch : {kwargs['epoch']+1}, iteration : {kwargs['iteration']}")
            print(f"{stage} g_loss : {kwargs['g_loss']}")
            print(f"{stage} d_loss : {kwargs['d_loss']}")
            
    def forward(self, inputs: torch.Tensor, user_input: str, check_user_input: bool=True):
        out = self.whisper_model(
            inputs, 
            decoder_input_ids=self.decoder_input_ids['decoder_input_ids'].repeat(inputs.size(0), 1),
            output_hidden_states=True
        )
        decoder_last_hidden_state = out.decoder_hidden_states[-1]
        logits = out.logits 

        logprobs = F.log_softmax(logits, dim=-1)
        logprobs = logprobs[:, 3:, :]
        predicted_token_ids = torch.argmax(logprobs, dim=-1)
        predicted_transcribe = [pred.strip() for pred in self.whisper_tokenizer.batch_decode(predicted_token_ids)]
        
        # predict_logits = torch.gather(logits, dim=-1, index=predicted_token_ids.repeat(4, 1, 1))

        correct_embed_vectors = []
        for idx, pred in enumerate(predicted_transcribe):
            if check_user_input is True:
                if pred == user_input:
                    correct_embed_vectors.append(decoder_last_hidden_state[idx, 3:, :].unsqueeze(0))
            else:
                correct_embed_vectors.append(decoder_last_hidden_state[idx, 3:, :].unsqueeze(0))
        if len(correct_embed_vectors) < 1:
            return None
        correct_embed_vectors = torch.cat(correct_embed_vectors)    # (b, T, F)
        if correct_embed_vectors.size(1) > 1:
            correct_embed_vectors = torch.mean(correct_embed_vectors, dim=1) # (b, F)
        else: 
            correct_embed_vectors = correct_embed_vectors.squeeze(1)
        correct_embed_vectors = F.layer_norm(
            correct_embed_vectors, 
            normalized_shape=(correct_embed_vectors.size(-1), )
        )

        return {'embed_vectors': correct_embed_vectors}

    def trainer(self, config, dataset: Dataset, user_input: str) -> None:
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        master_key = utils.make_random_key(key_size=128, return_negative=False).type(Tensor)   # (key_size, )

        optimizer_G = torch.optim.Adam(
            itertools.chain(self.crypto_encoder.parameters(), self.crypto_decoder.parameters()),
            lr=config.lr
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr
        )

        pixelwise_loss = torch.nn.MSELoss()
        classification_loss = torch.nn.MSELoss()

        for epoch in range(config.epoch):
            for idx, (path, inputs) in enumerate(dataloader):
                inputs = Variable(inputs.type(Tensor))
                outputs = self(inputs, user_input)
                if outputs is None:
                    continue
                
                embed_vectors = outputs['embed_vectors']
                valid_key = master_key.repeat(embed_vectors.size(0), 1)

                # ---------------
                # training weight
                # ---------------
                out = self.discriminator(embed_vectors, torch.where(valid_key==0, 1., -1.))

                optimizer_G.zero_grad()

                g_loss = pixelwise_loss(out, torch.where(valid_key==0, -1., 1.))
                g_loss.backward()
                optimizer_G.step()

                self.discriminator.limit_param_range()

                print(f"g loss : {g_loss.item()}")

        # save model
        torch.save(self.state_dict(), config.save_path)

        return
    
    @torch.no_grad()
    def inference(self, config, wav_path, user_input):
        sound_data = load_sound_data(wav_path, return_mel=True)
        self.load_state_dict(torch.load(config.load_path))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        inputs = sound_data['mel'][0]
        outputs = self(inputs.type(Tensor), user_input, check_user_input=False)
        encoded = outputs['encoded']

        predicted = torch.sigmoid(self.discriminator(encoded).squeeze())

        # classification
        print(f"classification : {predicted.item()}")
            
        return
