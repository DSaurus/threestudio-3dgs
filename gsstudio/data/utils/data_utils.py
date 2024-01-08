import torch
from threestudio.utils.typing import *


def batch_merge_output(output_list):
    merged_output = type(output_list[0])()
    for key in dir(output_list[0]):
        if isinstance(key, torch.Tensor):
            prop_list = []
            for output in output_list:
                prop_list.append(getattr(output, key))
            merged_output[key] = torch.cat(prop_list, dim=0)
        elif isinstance(key, List):
            prop_list = []
            for output in output_list:
                prop_list = prop_list + (getattr(output, key))
            merged_output[key] = prop_list
        else:
            merged_output[key] = getattr(output_list[0], key)
    return merged_output
