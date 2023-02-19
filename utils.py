import os
import torch

def bit_to_string(data):
    data = data.to(dtype=torch.int8)
    data = str(data.tolist()).lstrip('[').rstrip(']').split(', ')
    data = ''.join(data)
    return data

def string_to_bit(data):
    return [int(bit) for bit in data]

def make_random_key(key_size: int, return_negative: bool=False):
    if os.path.exists('./key.txt'):
        with open('./key.txt', 'r+') as f:
            random_key = string_to_bit(f.readline())
    else:
        random_key = torch.randint(0, 2, (key_size, ))
        with open('./key.txt', 'w+') as f:
            f.write(bit_to_string(random_key))
    
    if return_negative:
        random_key = [-1 if bit == 0 else 1 for bit in random_key]
    return torch.tensor(random_key)
