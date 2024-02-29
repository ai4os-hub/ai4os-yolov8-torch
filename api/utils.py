"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However,
you may need to modify them for your needs.
"""

import logging
import subprocess  # nosec B404
import sys
import os
from marshmallow import fields
from subprocess import TimeoutExpired  # nosec B404
import ultralytics
import yaml
from . import config
import git

import subprocess



import cv2
import random
 
import numpy as np

import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc
import torch

from mlflow.entities import Dataset
from mlflow.models import infer_signature
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


def list_directories_with_rclone(remote_name, directory_path):
    """
    Function to list directories within a given directory in Nextcloud
    using rclone.

    Args:
        remote_name (str): Name of the configured Nextcloud remote in rclone.
        directory_path (str): Path of the parent directory to list the
            directories from.

    Returns:
        list: List of directory names within the specified parent directory.
    """
    command = ["rclone", "lsf", remote_name + ":" + directory_path]
    result = subprocess.run(
        command, capture_output=True, text=True, shell=False
    )  # nosec B603

    if result.returncode == 0:
        directory_names = result.stdout.splitlines()
        directory_names = [
            d.rstrip("/") for d in directory_names if d[0].isdigit()
        ]
        return directory_names
    else:
        print("Error executing rclone command:", result.stderr)
        return []


def ls_remote():
    """
    Utility to return a list of current backbone models stored in the
    remote folder configured in the backbone url.

    Returns:
        A list of strings.
    """
    remote_directory = config.REMOTE_PATH
    return list_directories_with_rclone("rshare", remote_directory)


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
    ) as process:  # nosec B603
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

    def inject_function_schema(func):
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
    """
    Load and parse a YAML configuration file into a Python object.

    Args:
        default_cfg_path (str): The path to the YAML configuration file.

    Returns:
        Tuple[ultralytics.utils.IterableSimpleNamespace, dict_keys]:
        A tuple containing two elements:
            1. A Python object representing the configuration.
            2. A dictionary_keys object containing the keys in
            the loaded configuration.
    """
    try:
        with open(default_cfg_path, "r") as yaml_file:
            default_cfg_dict = yaml.safe_load()(
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


def check_paths_in_yaml(yaml_path, base_path):
    """
    Check and potentially update file paths specified in a YAML
    configuration file.

    Args:
        yaml_path (str): The path to the YAML configuration file.
        base_path (str): The base directory to prepend to relative
        file paths.

    Returns:
        bool: True if all paths exist or have been successfully updated,
        False otherwise.
    """
    with open(yaml_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    paths_to_check = []
    if "train" in data:
        paths_to_check.append(data["train"])
    if "val" in data:
        paths_to_check.append(data["val"])

    for i, path in enumerate(paths_to_check):
        if not os.path.exists(path):
            new_path = os.path.join(base_path, path)
            if os.path.exists(new_path):
                data["train" if i == 0 else "val"] = new_path

                with open(yaml_path, "w") as yaml_file:
                    yaml.dump(data, yaml_file)
            else:
                return False

    return True


def validate_and_modify_path(path, base_path):
    """
    Validate and modify a file path, ensuring it exists

    Args:
        path (str): The input file path to validate.
        base_path (str): The base path to join with 'path' if it
        doesn't exist as-is.

    Returns:
        str: The validated and possibly modified file path.
    """
    if not os.path.exists(path):
        path = os.path.join(base_path, path)
        if not os.path.exists(path):
            raise ValueError(
                f"The path {path} does not exist."
                "Please provide a valid path."
            )
    return path


def add_arguments_from_schema(schema, parser):
    """
    Iterates through the fields defined in a schema and adds
    corresponding commandline arguments to the provided
    ArgumentParser object.

    Args:
        schema (marshmallow.Schema): The schema object containing field
        definitions.
        parser (argparse.ArgumentParser): The ArgumentParser object
        to which arguments will be added.

    Returns:
        None
    """
    for field_name, field_obj in schema.fields.items():
        arg_name = f"--{field_name}"

        arg_kwargs = {
            "help": field_name,
        }

        if isinstance(field_obj, fields.Int):
            arg_kwargs["type"] = int
        elif isinstance(field_obj, fields.Bool):
            arg_kwargs["action"] = (
                "store_false"
                if field_obj.load_default
                else "store_true"
            )

        elif isinstance(field_obj, fields.Float):
            arg_kwargs["type"] = float
        else:
            arg_kwargs["type"] = str

        if field_obj.required:
            arg_kwargs["required"] = True

        if field_obj.load_default and not isinstance(
            field_obj, fields.Bool
        ):
            arg_kwargs["default"] = field_obj.load_default

        if field_obj.metadata.get("description"):
            arg_kwargs["help"] = field_obj.metadata["description"]
        # Debug print statements
        parser.add_argument(arg_name, **arg_kwargs)


def generate_directory_tree(path):
    tree = {
        "name": os.path.basename(path),
        "type": "directory",
        "children": [],
    }

    if os.path.exists(path) and os.path.isdir(path):
        subdirectories = [
            d
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        subdirectories.sort()

        for subdir in subdirectories:
            subdir_path = os.path.join(path, subdir)
            tree["children"].append(
                generate_directory_tree(subdir_path)
            )

    return tree



def get_git_info():
    try:
        # Initialize the Git repository object
        repo = git.Repo(search_parent_directories=True)

        # Get the remote URL of the repository
        remote_url = repo.remotes.origin.url
        remote_repo = git.cmd.Git().ls_remote(remote_url)
        version = remote_repo.split()[0]
        return remote_url , version
    except git.InvalidGitRepositoryError:
        print("Error: Not a valid Git repository.")
        return None


def mlflow_update():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
    )
    # check the latest version of the model
    model_version_infos = client.search_model_versions(
        f"name = '{config.MLFLOW_MODEL_NAME}'"
    )
    last_model_version = max(
        [
            model_version_info.version
            for model_version_info in model_version_infos
        ]
    )
    model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/{last_model_version}"
    print("model_uri", model_uri)

    # Add a description to the registered model
    client.update_registered_model(
        name=config.MLFLOW_MODEL_NAME,
        description=config.MLFLOW_RUN_DESCRIPTION,
    )

    # set tags, alias, update and delete them
    # create "champion" alias for version x of model "MLFLOW_MODEL_NAME"
    client.set_registered_model_alias(
        config.MLFLOW_MODEL_NAME, "champion", last_model_version
    )

    # get a model version by alias
    print(
        f"\n Model version alias: ",
        client.get_model_version_by_alias(
            config.MLFLOW_MODEL_NAME, "champion"
        ),
    )

    # delete the alias
    # client.delete_registered_model_alias(MLFLOW_MODEL_NAME, "Champion")

    # Set registered model tag
    client.set_registered_model_tag(
        config.MLFLOW_MODEL_NAME, "task", "detection"
    )
    client.set_registered_model_tag(
        config.MLFLOW_MODEL_NAME, "author", config.MLFLOW_AUTHOR
    )
    client.set_registered_model_tag(
        config.MLFLOW_MODEL_NAME, "framework", "pytorch"
    )

    # Set a transition to the model: Production, Stage, Archived, None
    client.transition_model_version_stage(
        name= config.MLFLOW_MODEL_NAME,
        version=last_model_version,
        stage="None",
    )

    # Get the current value of model transition
    model_version_details = client.get_model_version(
        name=config.MLFLOW_MODEL_NAME,
        version=last_model_version,
    )
    model_stage = model_version_details.current_stage
    print(f"The current model stage is: '{model_stage}'")

    # Set model transition to archive
    client.transition_model_version_stage(
        name=config.MLFLOW_MODEL_NAME,
        version=1,
        stage="Archived",
    )


def mlflow_logging(model, num_epochs, args):
    mlflow.end_run()  # stop any previous active mlflow run
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    with mlflow.start_run(run_name=args["name"]):
        artifact_uri = mlflow.get_artifact_uri()
        SANITIZE = lambda x: {
            k.replace("(", "").replace(")", ""): float(v)
            for k, v in x.items()
        }
        # logs metrics in mlflow
        for epoch in range(1, num_epochs + 1):
            mlflow.log_metrics(
                metrics=SANITIZE(
                    model.trainer.label_loss_items(
                        model.trainer.tloss, prefix="train"
                    )
                ),
                step=epoch,
            )
            mlflow.log_metrics(
                metrics=SANITIZE(model.trainer.lr),
                step=epoch,
            )
            mlflow.log_metrics(
                metrics=SANITIZE(model.trainer.metrics),
                step=epoch,
            )

        # Log the data.yaml file  as an artifact #

        # Assuming config.DATA_PATH contains the directory where data.yaml is located
        #data_path = config.DATA_PATH
        data_file_path = args["data"]
        with open(data_file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)

        # Use mlflow.log_artifact to log the contents of data.yaml
        mlflow.log_artifact(data_file_path, artifact_path="artifacts")

        ##           Log the Dataset                         ##

        # Print OpenCV version for debugging
        print(f"OpenCV Version: {cv2.__version__}")

        # Ensure that no random seed is set
        random.seed(None)

        # Specify the path to the folder containing images
        img_folder = data['train']

        # List all files in the folder
        img_files = [
            f
            for f in os.listdir(img_folder)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]

        # Check if there are any images in the folder
        if not img_files:
            print(f"No images found in {img_folder}")
        else:
            # Randomly select an image from the list
            selected_img = random.choice(img_files)

            # Construct the full path to the selected image
            img_path = os.path.join(img_folder, selected_img)

            # Read the selected image using OpenCV
            img = cv2.imread(img_path)

            # Check if the image was read unsuccessfully
            if img is None:
                print(
                    f"Error: Unable to read the image from {img_path}"
                )
            else:
                # Continue with the rest of your code using the selected image
                resized_img = cv2.resize(
                    img, (640, 640), interpolation=cv2.INTER_LINEAR
                )
                normalized_img = resized_img / 255.0
                torch_img = torch.from_numpy(normalized_img).float()
                torch_img = torch_img.permute(2, 0, 1).unsqueeze(0)
                torch_img_cpu = torch_img.cpu().numpy()

        # Convert the NumPy array to a MLflow Dataset entity
        dataset_entity = mlflow.data.from_numpy(torch_img_cpu)
        print("dataset_info", dataset_entity)

        # Log the Dataset entity as input
        mlflow.log_input(dataset_entity, context="training")

        run, active_run = mlflow, mlflow.active_run()
        print("active run id", active_run.info.run_id)

        # logs params in mlflow
        git_repo, version= get_git_info()
        git_info={'git_repo':git_repo, 'git_version':version}
        merged_params = {**vars(model.trainer.model.args), **git_info}
        run.log_params(
          merged_params
        )

        # Infer signature to a model, i.e. the input data used to feed the model
        # and output of the trained model #
        #                                                               #

        # Assuming model is an instance of YOLO and img is an input image
        prediction_inf = model(img)

        train_inf_np = model.trainer.model.info()
        print("\ntrain_inf_np", train_inf_np)
        train_inf_array = np.array(train_inf_np)

        # Convert the tuple to a dictionary
        train_inf = {"value": np.array(train_inf_np)}
        print("\ntrain_inf", train_inf)
        # print ("\npred_inf", prediction_inf)

        # Create list of detection dictionaries
        results_all = []

        for result in prediction_inf:
            # Assuming result is an ultralytics.engine.results.Results object
            data = result.boxes.data.cpu().tolist()
            h, w = result.orig_shape
            for i, row in enumerate(
                data
            ):  # xyxy, track_id if tracking, conf, class_id
                box = {
                    "x1": row[0] / w,
                    "y1": row[1] / h,
                    "x2": row[2] / w,
                    "y2": row[3] / h,
                }
                conf = row[-2]
                class_id = int(row[-1])
                name = result.names[class_id]
                detection_result = {
                    "name": name,
                    "class": class_id,
                    "confidence": conf,
                    "box": box,
                }
                if result.boxes.is_track:
                    detection_result["track_id"] = int(
                        row[-3]
                    )  # track ID
                if result.masks:
                    x, y = (
                        result.masks.xy[i][:, 0],
                        result.masks.xy[i][:, 1],
                    )  # numpy array
                    detection_result["segments"] = {
                        "x": (x / w).tolist(),
                        "y": (y / h).tolist(),
                    }
                if result.keypoints is not None:
                    x, y, visible = (
                        result.keypoints[i]
                        .data[0]
                        .cpu()
                        .unbind(dim=1)
                    )  # torch Tensor
                    detection_result["keypoints"] = {
                        "x": (x / w).tolist(),
                        "y": (y / h).tolist(),
                        "visible": visible.tolist(),
                    }

                results_all.append(detection_result)
            MODELS_PATH= os.path.join(args["project"], args["name"])
            output_file_path = os.path.join(
                MODELS_PATH, "detection_results.txt"
            )
            with open(output_file_path, "w") as output_file:
                # Now write all results to the file after the loop
                for result_entry in results_all:
                    output_file.write(f"{str(result_entry)}\n")

        # Use mlflow.log_artifact to log the contents of the detection_results
        mlflow.log_artifact(
            output_file_path, artifact_path="artifacts"
        )

        # Use infer_signature with train_inf and results_all
        signature = infer_signature(
            train_inf, {"detections": results_all}
        )

        # Get the base directory of artifact_path
        base_dir = os.path.basename(str(model.trainer.save_dir))
        print("\nbase_dir", base_dir)

        model_t = mlflow.pyfunc.log_model(
            artifact_path="artifacts",
            python_model=mlflow.pyfunc.PythonModel(),
            signature=signature,
        )

        # Log additional artifacts
        mlflow.log_artifacts(
            str(model.trainer.save_dir), artifact_path="artifacts"
        )

        # create a new version of that model
        client = MlflowClient(
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
        )
        run_id = active_run.info.run_id
        model_uri = mlflow.get_artifact_uri("artifacts")
        print('model url is ',model_uri)
        
        # register trained model to the Model Registry
        result = mlflow.register_model(
           f"runs:/{run_id}/artifacts/", config.MLFLOW_MODEL_NAME
        )

        # Update model description, tags, alias, transitions.
        mlflow_update()

    return {
        "artifact_path": args["name"],
        "artifact_uri": artifact_uri,
    }


def mlflow_fetch():

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"]

    )

    # check the latest version of the model
    model_version_infos = client.search_model_versions(
        f"name = '{config.MLFLOW_MODEL_NAME}'"
    )
    if not model_version_infos:
        # No model found in MLflow
        return None

    # get the last model
    # last_model_version = max([model_version_info.version for model_version_info in model_version_infos])

    # model_uri = F"models:/{MLFLOW_MODEL_NAME}/{last_model_version}"
    # get the production model
    model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/'Production'"
    # print("model_uri", model_uri)

    # define a path dir where to store locally the mlflow loaded model
    dst_path = config.MODELS_PATH
    # Extract the 'RUN_ID' field value for each ModelVersion
    run_id = [
        model_version.run_id for model_version in model_version_infos
    ]

    # Fetch a model using model_uri using the default path where the MLflow logged the model
    loaded_model = mlflow.pyfunc.load_model(
        model_uri, dst_path=dst_path
    )

    # Define the destination path in MODELS_PATH  and
    # load the best pretrained logged model
    path = os.path.join(dst_path, "weights/last.pt")
    print("Path", path)

    return path
