import torch
from gsstudio.utils.typing import *


def batch_merge_output(output_list):
    merged_output = type(output_list[0])()
    for key in dir(output_list[0]):
        if key.startswith("__"):
            continue
        prop = getattr(output_list[0], key)
        if isinstance(prop, torch.Tensor):
            prop_list = []
            for output in output_list:
                prop_list.append(getattr(output, key))
            prop_list = torch.cat(prop_list, dim=0)
            setattr(merged_output, key, prop_list)
        elif isinstance(prop, list):
            prop_list = []
            for output in output_list:
                prop_list = prop_list + (getattr(output, key))
            setattr(merged_output, key, prop_list)
        elif isinstance(prop, (int, float, str, bool)):
            setattr(merged_output, key, prop)
    return merged_output


class DataOutput:
    key_mapping = {}

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

    def to_dict(self):
        output = {}
        for key in dir(self):
            if key.startswith("__"):
                continue
            prop = getattr(self, key)
            if not isinstance(prop, (torch.Tensor, list, int, float, str, bool)):
                continue
            if prop is not None:
                if key in self.key_mapping:
                    key = self.key_mapping[key]
                output[key] = prop
        return output

    def get_index(self, index, is_batch=False, is_tensor=False):
        output = type(self)()
        for key in dir(self):
            if key.startswith("__"):
                continue
            prop = getattr(self, key)
            if not isinstance(prop, (torch.Tensor, list, int, float, str, bool)):
                continue
            if prop is not None:
                if key in self.key_mapping:
                    key = self.key_mapping[key]
                if isinstance(prop, torch.Tensor):
                    if is_batch:
                        setattr(output, key, prop[index : index + 1])
                    else:
                        setattr(output, key, prop[index])
                elif is_tensor == False:
                    if isinstance(prop, list):
                        setattr(output, key, prop[index])
                    else:
                        setattr(output, key, prop)
        return output
