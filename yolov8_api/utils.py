"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to white all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
import logging
import yolov8_api.config as cfg
import yaml
import os

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


def check_annotations_format(path):
    """Check if annotations are in the correct format.
    Check and preprocess annotation files in specified directories.

    Args:
        data (str): YAML-formatted string containing directory paths.

    Raises:
        ValueError: If an annotations directory path is invalid.

    Returns:
        None
    """
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    data_keys = data.keys()
    for key in data_keys:
        if os.path.exists(data[key]):
            annotations_path = data.get(key, None)

            if annotations_path is None or not os.path.isdir(
                annotations_path
            ):
                raise ValueError("Invalid annotations directory path")

            supported_formats = (".txt", ".json", ".xml")
            annotations = [
                os.path.join(annotations_path, x)
                for x in os.listdir(annotations_path)
                if any(
                    x.endswith(format) for format in supported_formats
                )
            ]

            json_annotations = [
                ann for ann in annotations if ann.endswith(".json")
            ]
            xml_annotations = [
                ann for ann in annotations if ann.endswith(".xml")
            ]
            txt_annotations = [
                ann for ann in annotations if ann.endswith(".txt")
            ]

            if json_annotations or xml_annotations:
                raise ValueError(
                    "Invalid annotations format (json, xml): "
                    "please convert the annotations format into .txt. "
                    "You can use either 'yolov8_api/preprocess_ann.py'"
                    " (for a detection task) "
                    "or 'yolov8_api/seg_coco_json_to_yolo.py'"
                    "(for segmentation)."
                )

            elif not txt_annotations:
                raise ValueError(
                    "No valid .txt annotations found in "
                    f"'{key}' directory: Please use .txt format"
                )
