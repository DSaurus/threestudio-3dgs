try:
    import threestudio
except ImportError:
    print("threestudio is not installed and some features will not be available.")
    threestudio = None


__modules__ = {}
__version__ = "0.0.1"


def register(name):
    def decorator(cls):
        if name in __modules__:
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        else:
            __modules__[name] = cls
            if threestudio is not None:
                threestudio.__modules__[name] = cls
        return cls

    return decorator


def find_module(name):
    if __modules__.__contains__(name):
        return __modules__[name]
    else:
        print(f"Module {name} is not found in gsstudio, try to find it in threestudio.")
        return threestudio.__modules__[name]


def find(name):
    if ":" in name:
        main_name, sub_name = name.split(":")
        if "," in sub_name:
            name_list = sub_name.split(",")
        else:
            name_list = [sub_name]
        name_list.append(main_name)
        NewClass = type(
            f"{main_name}.{sub_name}",
            tuple([find_module(name) for name in name_list]),
            {},
        )
        return NewClass
    return find_module(name)


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, diffusion, loss, renderer, representation, system
