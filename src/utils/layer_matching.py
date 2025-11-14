from torch import nn


def collect_matching_layers(model, layer_matcher):
    """
    Find all layers in the model that match the layer_matcher, in order by layer_num.
    layer_matcher can be a string formatted like "transformer.h.{num}.mlp" or a callable.
    """
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    all_layer_names = dict(model.named_modules()).keys()
    matching_layers = []
    for layer_num, layer in enumerate(model.modules()):
        layer_name = matcher_callable(model, layer_num)
        if layer_name in all_layer_names:
            matching_layers.append(layer_name)
        else:
            break
    return matching_layers

def get_layer_name(model, layer_matcher, layer_num):
    matcher_callable = _layer_matcher_to_callable(layer_matcher)
    return matcher_callable(model, layer_num)

def get_layer_by_name(model, layer_name):
    return dict(model.named_modules())[layer_name]

def _layer_matcher_to_callable(layer_matcher):
    if isinstance(layer_matcher, str):
        if "{num}" not in layer_matcher:
            raise ValueError("layer_matcher must be a callable or a string containing {num}")
        return lambda _model, layer_num: layer_matcher.format(num=layer_num)
    return layer_matcher
