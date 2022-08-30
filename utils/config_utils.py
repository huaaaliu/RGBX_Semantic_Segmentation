import importlib
import os
import sys


def get_config_by_file(config_file):
    try:
        sys.path.append(os.path.dirname(config_file))
        current_config = importlib.import_module(os.path.basename(config_file).split(".")[0])
        config = current_config.config
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(config_file))
    return config