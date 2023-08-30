"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However, you may
need to modify them for your needs.
"""
import logging
import subprocess
import sys
import os
from subprocess import TimeoutExpired
import ultralytics
import yaml
from types import SimpleNamespace
from . import config
import yolov8_api as aimodel

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def ls_dirs(path):
    """Utility to return a list of directories available in `path` folder.

    Arguments:
        path -- Directory path to scan for folders.

    Returns:
        A list of strings for found subdirectories.
    """
    logger.debug("Scanning directories at: %s", path)
    dirscan = (x.name for x in path.iterdir() if x.is_dir())
    return sorted(dirscan)


def ls_files(path, pattern):
    """Utility to return a list of files available in `path` folder.

    Arguments:
        path -- Directory path to scan.
        pattern -- File pattern to filter found files. See glob.glob() python.

    Returns:
        A list of strings for files found according to the pattern.
    """
    logger.debug("Scanning for %s files at: %s", pattern, path)
    dirscan = (x.name for x in path.glob(pattern))
    return sorted(dirscan)


def copy_remote(frompath, topath, timeout=600):
    """Copies remote (e.g. NextCloud) folder in your local deployment or
    vice versa for example:
        - `copy_remote('rshare:/data/images', '/srv/myapp/data/images')`

    Arguments:
        frompath -- Source folder to be copied.
        topath -- Destination folder.
        timeout -- Timeout in seconds for the copy command.

    Returns:
        A tuple with stdout and stderr from the command.
    """
    with subprocess.Popen(
        args=["rclone", "copy", f"{frompath}", f"{topath}"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        text=True,  # Return strings rather than bytes
    ) as process:
        try:
            outs, errs = process.communicate(None, timeout)
        except TimeoutExpired:
            logger.error(
                "Timeout when copying from/to remote directory."
            )
            process.kill()
            outs, errs = process.communicate()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Error copying from/to remote directory\n %s", exc
            )
            process.kill()
            outs, errs = process.communicate()
    return outs, errs


def modify_model_name(model_name, task_type):
    """
    Modify the model name based on the task type.

    Args:
        model_name (str): The original model name (e.g., "yolov8n.yaml").
        task_type (str): The task type ("det", "seg", "cls").

    Returns:
        str: The modified model name.
    """
    logger.info(f"Original model name: {model_name}")
    logger.info(f"Task type: {task_type}")

    if task_type in ["seg", "cls"]:
        base_name, extension = os.path.splitext(model_name)
        modified_model_name = f"{base_name}-{task_type}{extension}"
    else:
        modified_model_name = model_name
    logger.info(f"Modified model name: {modified_model_name}")
    return modified_model_name


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema





def load_config(default_cfg_path):
    try:
        with open(default_cfg_path, "r") as yaml_file:
            default_cfg_dict = yaml.load(
                yaml_file, Loader=yaml.Loader
            )

        for k, v in default_cfg_dict.items():
            if isinstance(v, str) and v.lower() == "none":
                default_cfg_dict[k] = None

        default_cfg_keys = default_cfg_dict.keys()
        default_cfg = ultralytics.utils.IterableSimpleNamespace(
            **default_cfg_dict
        )

        return default_cfg, default_cfg_keys

    except Exception as err:
        raise Exception(f"Error loading default config: {err}")
def check_annotations_format(data):
    """Check if annotations are in the correct format.
    Check and preprocess annotation files in specified directories.

    This function takes YAML data containing directory paths for annotation files.
    It checks if the specified directories exist and contain valid annotation files,
    and then preprocesses the annotation files based on their format.

    Args:
        data (str): YAML-formatted string containing directory paths.

    Raises:
        ValueError: If an annotations directory path is invalid.

    Returns:
        None
    """
    data = yaml.safe_load(data)
    data_keys = data.keys()
    for key in data_keys:
        if os.path.exists(data[key]):
            annotations_path = data.get(key, None)

            if annotations_path is None or not os.path.isdir(annotations_path):
                raise ValueError("Invalid annotations directory path")

            supported_formats = (".txt", ".json", ".xml")
            annotations = [
                os.path.join(annotations_path, x)
                for x in os.listdir(annotations_path)
                if any(x.endswith(format) for format in supported_formats)
            ]

            json_annotations = [ann for ann in annotations if ann.endswith(".json")]
            xml_annotations = [ann for ann in annotations if ann.endswith(".xml")]

            if json_annotations:
                format = "json"
                args = {
                    "format": format,
                    "annotation_path": annotations_path
                }
                aimodel.preprocess_ann.main(**args)
            elif xml_annotations:
                format = "xml"
                args = {
                    "format": format,
                    "annotation_path": annotations_path
                }
                aimodel.preprocess_ann.main(**args)
            else:
                print("No JSON or XML annotation files found.")

    
class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)
def pop_keys_from_dict(dictionary, keys_to_pop):
    for key in keys_to_pop:
        dictionary.pop(key, None)                
