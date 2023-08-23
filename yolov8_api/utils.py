"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to white all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
from pathlib import Path
import yolov8_api.config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


# make data
# = HAVE TO MODIFY FOR YOUR NEEDS =
def mkdata(input_filepath, output_filepath):
    """Main/public function to run data processing to turn raw data
    from (data/raw) into cleaned data ready to be analyzed.
    """

    logger.info("Making final data set from raw data")

    # EXAMPLE for finding various files
    project_dir = Path(__file__).resolve().parents[2]


# create model
# = HAVE TO MODIFY FOR YOUR NEEDS =
def create_model(**kwargs):
    """Main/public method to create AI model"""
    # define model parameters

    # build model based on the deep learning framework

    pass


def remove_keys_from_dict(input_dict, keys_to_remove):
    for key in keys_to_remove:
        input_dict.pop(key, None)
