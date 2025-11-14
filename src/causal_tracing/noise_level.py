import torch
from torch import nn

from utils.token_utils import make_inputs
from utils.torch_utils import get_module
from utils.trace import Trace


def pick_noise_level(model, embedding_layer, std_multiplier=3):
    """
    Pick a noise level to corrupt the input text with, such that the
    noise is a multiplier of the stdev of the token embeddings.
    """
    with torch.no_grad():
        embedding_weights = get_module(model, embedding_layer).weight
        noise_level_std = embedding_weights.std().item()
    return std_multiplier * noise_level_std


def resize_list_by_samples(ls, samples):
    if not isinstance(ls, list):
        return [None] * samples

    return (ls * (samples // len(ls)) + ls[:samples % len(ls)])[:samples]


def retrieved_embeddings(model, tokenizer, layer, prompts, add_special_tokens=False):
    prompts = prompts if isinstance(prompts, list) else [prompts]
    input_dict = make_inputs(tokenizer, prompts, add_special_tokens=add_special_tokens)

    with Trace(model, layer, stop=True) as t:
        model(input_ids=input_dict['input_ids'], attention_mask=input_dict["attention_mask"])

    embeddings = t.output[:, :, :].detach().cpu().tolist()
    
    return embeddings