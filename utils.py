"""
Utility helpers for loading the model and tokenizer used by scripts.
"""
import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT


def load_model(init_from, ckpt_path, device, override_args=None):
    """
    Load a model either from a checkpoint ('resume') or from a pretrained gpt2 variant.
    Returns (model, checkpoint_or_none).
    """
    checkpoint = None
    if init_from == 'resume':
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, override_args or {})
    else:
        raise ValueError("init_from must be 'resume' or start with 'gpt2'")
    model.eval()
    model.to(device)
    return model, checkpoint


def prepare_tokenizer(checkpoint, init_from):
    """
    Return (encode, decode) functions. Mirrors logic in sample.py.
    """
    if init_from == 'resume' and checkpoint is not None and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
            return encode, decode
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


