from torch import nn


class LayerConfig:
    def __init__(self, hidden_layers_matcher, attention_layers_matcher, mlp_layers_matcher, embedding_layer):
        self.hidden_layers_matcher = hidden_layers_matcher
        self.attention_layers_matcher = attention_layers_matcher
        self.mlp_layers_matcher = mlp_layers_matcher
        self.embedding_layer = embedding_layer

GPT2_LAYER_CONFIG = LayerConfig(
    hidden_layers_matcher="transformer.h.{num}",
    attention_layers_matcher="transformer.h.{num}.attn",
    mlp_layers_matcher="transformer.h.{num}.mlp",
    embedding_layer="transformer.wte",
)

GPTJ_LAYER_CONFIG = LayerConfig(
    hidden_layers_matcher="h.{num}",
    attention_layers_matcher="h.{num}.attn",
    mlp_layers_matcher="h.{num}.mlp",
    embedding_layer="wte",
)

LLAMA_LAYER_CONFIG = LayerConfig(
    hidden_layers_matcher="model.layers.{num}",
    attention_layers_matcher="model.layers.{num}.self_attn",
    mlp_layers_matcher="model.layers.{num}.mlp",
    embedding_layer="model.embed_tokens",
)

def get_layer_config(model):
    if hasattr(model, "config"):
        config = model.config
        model_type = config.model_type
        if model_type == "gpt2":
            return GPT2_LAYER_CONFIG
        elif model_type == "gptj":
            return GPTJ_LAYER_CONFIG
        elif model_type == "llama":
            return LLAMA_LAYER_CONFIG
    raise ValueError("Unknown model type. Provide a LayerConfig to identify relevant model layers for tracing.")
