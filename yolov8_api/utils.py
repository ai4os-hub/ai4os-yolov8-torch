"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to white all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
import yolov8_api.config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


def create_model(**kwargs):
    """Main/public method to create AI model"""
    # define model parameters

    # build model based on the deep learning framework

    pass


def remove_keys_from_dict(input_dict, keys_to_remove):
    for key in keys_to_remove:
        input_dict.pop(key, None)
