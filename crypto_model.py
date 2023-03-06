from dataloader import load_sound_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import pickle as pkl
import os

from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from transformers import WhisperModel, WhisperTokenizer, WhisperForConditionalGeneration


class CryptoModelOutputLayer(nn.Module):
    def __init__(self):
        super(CryptoModelOutputLayer, self).__init__()
        self.latent_dim = 128       # key size

        self.fc1 = nn.Linear(384, self.latent_dim)
        self.w = torch.nn.Parameter(data=torch.Tensor(np.random.normal(0, 1, (self.latent_dim,))), requires_grad=True)

    def limit_param_range(self):
        self.w.data.clamp_(-10., 10.)

    def forward(self, z):
        hidden_states = z
        hidden_states = self.fc1(hidden_states)  # (b, 128)
        hidden_states = F.dropout(hidden_states, p=0.1)
        
        # # binding step
        # hidden_states = hidden_states * torch.where(key==0, -1., 1.)
        hidden_states = hidden_states * self.w
        # hidden_states = F.dropout(hidden_states, p=0.2)
        
        hidden_states = torch.sigmoid(hidden_states)   # 0 ~ 1
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
        
        self.output_layer = CryptoModelOutputLayer()

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

    def gaussian_normalization(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = x.mean(dim=-1).unsqueeze(-1), x.std(dim=-1).unsqueeze(-1)
        x = (x - mean) / std
        return x
    
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

        correct_embed_vectors = self.gaussian_normalization(correct_embed_vectors)
        
        # layer normalization
        # correct_embed_vectors = F.layer_norm(
        #     correct_embed_vectors, 
        #     normalized_shape=(correct_embed_vectors.size(-1), )
        # )

        return {'embed_vectors': correct_embed_vectors}

    def trainer(self, config, dataset: Dataset, user_input: str) -> None:
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        master_key = utils.make_random_key(key_size=128, return_negative=False).type(Tensor)   # (key_size, )

        optimizer = torch.optim.Adam(
            self.output_layer.parameters(),
            lr=config.lr
        )

        criterion = nn.MSELoss()
        cosine_similarity = nn.CosineSimilarity()
        
        for epoch in range(config.epoch):
            for idx, (path, inputs) in enumerate(dataloader):
                inputs = Variable(inputs.type(Tensor))
                outputs = self(inputs, user_input)
                if outputs is None:
                    continue
                
                valid_embed_vectors = outputs['embed_vectors']
                valid_key = master_key.repeat(valid_embed_vectors.size(0), 1)   # (b, 384)
                
                fake_embed_vectors = Tensor(np.random.normal(0, 1, valid_embed_vectors.shape ))

                batch_embed_vectors = torch.cat([valid_embed_vectors, fake_embed_vectors])
                batch_key = torch.cat([valid_key, valid_key])

                # ---------------
                # training weight
                # ---------------
                out = self.output_layer(batch_embed_vectors)

                optimizer.zero_grad()

                cos = cosine_similarity(out, batch_key).unsqueeze(-1)

                loss = criterion(cos, Tensor(cos.size()).fill_(1.))
                loss.backward()
                optimizer.step()

                print(f"cosine similarity loss : {loss.item()}")

        # save model
        if not os.path.exists('model'):
            os.makedirs('model', exist_ok=True)
        torch.save(self.state_dict(), config.save_path)
        
        return
    
    @torch.no_grad()
    def test(self, config:OmegaConf , wav_path: str, user_input: str) -> None:
        sound_data = load_sound_data(wav_path, return_mel=True)
        self.load_state_dict(torch.load(config.load_path))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        inputs = sound_data['mel'][0]
        outputs = self(inputs.type(Tensor), user_input, check_user_input=False)
        encoded = outputs['embed_vectors']

        predicted = self.output_layer(encoded)
        predicted = torch.where(predicted < 0.5, 0, 1)
        predicted = predicted.squeeze().cpu().detach()
        
        answer = utils.make_random_key(key_size=128)

        print(f"p : {utils.bit_to_string(predicted)}")
        print(f"a : {utils.bit_to_string(answer)}")
            
        return
    
    @torch.no_grad()
    def inference(self, config: OmegaConf, wav_path: str, user_input: str) -> None:
        sound_data = load_sound_data(wav_path, return_mel=True)
        self.load_state_dict(torch.load(config.load_path))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        inputs = sound_data['mel'][0]
        outputs = self(inputs.type(Tensor), user_input, check_user_input=True)

        if outputs:
            encoded = outputs['embed_vectors']

            predicted = self.output_layer(encoded)
            predicted = torch.where(predicted < 0.5, 0, 1)
            predicted = predicted.squeeze().cpu().detach()

            print(f"predicted : {utils.bit_to_string(predicted)}")
        else:
            print("you are not user")

        return
        