import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def copy_weights(source_model: nn.Module, target_model: nn.Module):
    packed_modules_mapping = getattr(target_model, "packed_modules_mapping", {})
    for name, param in source_model.named_parameters():
        for k in packed_modules_mapping:
            if k in name:
                v, shard_id = packed_modules_mapping[k]
                target_name = name.replace(k, v)
                target_param = target_model.get_parameter(target_name)
                weight_loader = getattr(target_param, "weight_loader")
                weight_loader(target_param, param.data, shard_id)
                break
        else:
            try:
                target_param = target_model.get_parameter(name)
                weight_loader = getattr(target_param, "weight_loader", default_weight_loader)
                weight_loader(target_param, param.data)
            except AttributeError:
                continue