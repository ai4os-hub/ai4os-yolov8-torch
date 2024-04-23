"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to white all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""

import logging
import yolov8_api.config as cfg

import yaml
import os
import git
import cv2
import secrets
import csv

# I replacd the random beacuse of the security issues
from pathlib import Path
import random

import numpy as np

import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc
import torch

from mlflow.models import infer_signature
from yolov8_api import config


logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)

BASE_PATH = Path(__file__).resolve(strict=True).parents[1]
mlflow_runs_path = os.path.join(BASE_PATH, "models/mlruns")
if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri(mlflow_runs_path)
else:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())


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


def get_git_info():
    try:
        # Initialize the Git repository object
        repo = git.Repo(search_parent_directories=True)

        # Get the remote URL of the repository
        remote_url = repo.remotes.origin.url
        remote_repo = git.cmd.Git().ls_remote(remote_url)
        version = remote_repo.split()[0]
        return remote_url, version
    except git.InvalidGitRepositoryError:
        print("Error: Not a valid Git repository.")
        return None


def mlflow_update():

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
    model_uri = (
        f"models:/{config.MLFLOW_MODEL_NAME}/{last_model_version}"
    )
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
        "\n Model version alias: ",
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
        name=config.MLFLOW_MODEL_NAME,
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
    # mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    metrics = {}

    with mlflow.start_run(run_name=args["name"], nested=True):
        artifact_uri = mlflow.get_artifact_uri()
        results = os.path.join(
            args["project"], args["name"], "results.csv"
        )
        with open(results, mode="r") as file:

            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    key = key.strip()
                    key = key.strip().replace("(B)", "")
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(float(value))
        metrics.pop("epoch", None)
        # logs metrics in mlflow
        for metric_name, metric_values in metrics.items():
            for step, value in enumerate(metric_values, start=1):
                metric_name = metric_name.replace(")", "-").replace(
                    "(", "-"
                )
                mlflow.log_metric(metric_name, value, step=step)

        # Assuming config.DATA_PATH contains the directory where
        #  data.yaml is located
        data_file_path = args["data"]
        with open(data_file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)

        # Use mlflow.log_artifact to log the contents of data.yaml
        mlflow.log_artifact(data_file_path, artifact_path="artifacts")

        # Log the Dataset
        # Print OpenCV version for debugging
        print(f"OpenCV Version: {cv2.__version__}")

        # Ensure that no random seed is set
        random.seed(None)

        # Specify the path to the folder containing images
        img_folder = data["train"]

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
            selected_img = secrets.choice(img_files)
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
        if get_git_info is not None:
            git_repo, version = get_git_info()
            git_info = {"git_repo": git_repo, "git_version": version}
            merged_params = {
                **vars(model.trainer.model.args),
                **git_info,
            }
        run.log_params(merged_params)

        # Assuming model is an instance of YOLO and img is an input image
        prediction_inf = model(img)

        train_inf_np = model.trainer.model.info()
        print("\ntrain_inf_np", train_inf_np)

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
            MODELS_PATH = os.path.join(args["project"], args["name"])
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
        results_all = [str(value) for value in results_all]
        signature = infer_signature(
            train_inf, {"detections": results_all}
        )

        # Get the base directory of artifact_path
        base_dir = os.path.basename(str(model.trainer.save_dir))
        print("\nbase_dir", base_dir)

        mlflow.pyfunc.log_model(
            artifact_path="artifacts",
            python_model=mlflow.pyfunc.PythonModel(),
            signature=signature,
        )

        # Log additional artifacts
        mlflow.log_artifacts(
            str(model.trainer.save_dir), artifact_path="artifacts"
        )

        run_id = active_run.info.run_id
        model_uri = mlflow.get_artifact_uri("artifacts")
        print("model url is ", model_uri)

        # register trained model to the Model Registry
        # I remove the create_model_version that users can register
        # their own trained model to the Model Registry
        # result = mlflow.register_model(
        #    f"runs:/{run_id}/artifacts/", config.MLFLOW_MODEL_NAME
        # )

        # Update model description, tags, alias, transitions.
        # mlflow_update()

    return {
        "artifact_path": args["name"],
        "artifact_uri": artifact_uri,
    }


def mlflow_fetch():

    # check the latest version of the model
    model_version_infos = client.search_model_versions(
        f"name = '{config.MLFLOW_MODEL_NAME}'"
    )
    if not model_version_infos:
        # No model found in MLflow
        return None

    # get the last model
    last_model_version = max(
        [
            model_version_info.version
            for model_version_info in model_version_infos
        ]
    )

    model_uri = (
        f"models:/{config.MLFLOW_MODEL_NAME}/{last_model_version}"
    )
    # get the production model
    # model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/'Production'"
    # print("model_uri", model_uri)

    # define a path dir where to store locally the mlflow loaded model

    dst_path = os.getenv("MODELS_PATH", default=Path("models"))

    # Fetch a model using model_uri using the default path
    # where the MLflow logged the model
    mlflow.pyfunc.load_model(model_uri, dst_path=dst_path)

    # Define the destination path in MODELS_PATH  and
    # load the best pretrained logged model
    path = os.path.join(dst_path, "weights/last.pt")

    return path


if __name__ == "__main__":
    mlflow_fetch()
