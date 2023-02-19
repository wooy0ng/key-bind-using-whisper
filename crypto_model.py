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
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(),
            nn.PReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        residual = z
        hidden_states = self.fc1(z)
        hidden_states = residual + hidden_states
        hidden_states = self.fc2(hidden_states)

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
            
    def forward(self, inputs, user_input):
        user_input_token_ids = torch.tensor(self.whisper_tokenizer.encode(user_input)[4:-1]).unsqueeze(-1).to(self.device)

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
            if pred == user_input:
                correct_embed_vectors.append(decoder_last_hidden_state[idx, 3:, :].unsqueeze(0))
        if len(correct_embed_vectors) < 1:
            return None
        correct_embed_vectors = torch.cat(correct_embed_vectors)    # (b, T, F)
        correct_embed_vectors = F.layer_norm(
            correct_embed_vectors, 
            normalized_shape=(correct_embed_vectors.size(-1), )
        )

        encoded = self.crypto_encoder(correct_embed_vectors)
        decoded = self.crypto_decoder(encoded)
        
        return {'embedding_vector': correct_embed_vectors, 'encoded': encoded, 'decoded': decoded}

    def trainer(self, config, dataset: Dataset, user_input: str):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        master_key = utils.make_random_key(key_size=128, return_negative=False).type(Tensor)   # (key_size, )

        optimizer_A = torch.optim.Adam(
            itertools.chain(self.crypto_encoder.parameters(), self.crypto_decoder.parameters()),
            lr=config.lr
        )
        optimizer_G = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr
        )

        pixelwise_loss = torch.nn.MSELoss()
        bitwise_loss = torch.nn.MSELoss()

        for epoch in range(config.epoch):
            for idx, (path, inputs) in enumerate(dataloader):
                inputs = Variable(inputs.type(Tensor))
                outputs = self(inputs, user_input)
                if outputs is None:
                    continue
                
                embedding_vector = outputs['embedding_vector']
                encoded = outputs['encoded']
                decoded = outputs['decoded']
                
                # --------------------
                # training autoencoder
                # --------------------
                optimizer_A.zero_grad()

                a_loss = pixelwise_loss(embedding_vector, decoded)    # autoencoder
                a_loss.backward()
                optimizer_A.step()

                # ---------------------
                # training key generator
                # ---------------------
                optimizer_G.zero_grad()

                g_loss = 0.01 * pixelwise_loss(embedding_vector, decoded.detach()) \
                        + 0.99 * bitwise_loss(self.discriminator(encoded.detach()), master_key.repeat(encoded.size(0), 1, 1))
                g_loss.backward()
                optimizer_G.step()

                print(f"epoch : {epoch+1}, a_loss : {a_loss.item()}, g_loss : {g_loss.item()}")

                # train_logs = {'epoch': epoch, 'iteration': idx, 'g_loss': g_loss, 'd_loss': d_loss}
                # self.print_log(stage='train', **train_logs)
                
        # save model
        torch.save(self.state_dict(), config.save_path)

        print("path : ", path)
        with open(f"encoded.pkl", "wb+") as f:
            pkl.dump(encoded, f)
        return
    
    @torch.no_grad()
    def inference(self, config, wav_path, user_input):
        sound_data = load_sound_data(wav_path, return_mel=True)
        self.load_state_dict(torch.load(config.load_path))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        inputs = sound_data['mel'][0]
        outputs = self(inputs.type(Tensor), user_input)
        encoded = outputs['encoded']

        with open(f'encoded.pkl', 'rb+') as f:
            instance = pkl.load(f)

        predicted = self.discriminator(encoded).squeeze()

        # round & compare 
        master_key = utils.make_random_key(key_size=128).cpu().detach()
        predicted_key = torch.round(predicted).cpu().detach()

        print('master key : \t', utils.bit_to_string(master_key))
        print('predicted key : ', utils.bit_to_string(predicted_key))
            
        return
