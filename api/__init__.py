"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import getpass
import os
import logging
import datetime
import tempfile
import shutil
import argparse
import json
import torch
import mlflow
import requests

from ultralytics import YOLO, settings
from ultralytics.data.dataset import YOLODataset

from aiohttp.web import HTTPException
from deepaas.model.v2.wrapper import UploadedFile

import yolov8_api as aimodel
from yolov8_api.api import config, responses, schemas, utils

from mlflow import MlflowClient
import mlflow.pyfunc

from mlflow.entities import Dataset
from mlflow.models import infer_signature
import cv2
import random
from PIL import Image



logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


#global var

MLFLOW_MODEL_NAME = "yolov8_footballPlayersDetection"

def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.MODEL_NAME)
        metadata = {
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get(
                "author-emails"
            ),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            "models_local": utils.ls_dirs(config.MODELS_PATH),
            "models_remote": utils.ls_remote(),
            "datasets": utils.generate_directory_tree(
                config.DATA_PATH
            ),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err

    
def mlflow_fetch():
       
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient(tracking_uri = os.environ["MLFLOW_TRACKING_URI"])            

    #check the latest version of the model            
    model_version_infos = client.search_model_versions(F"name = '{MLFLOW_MODEL_NAME}'")
    if not model_version_infos:
        # No model found in MLflow
        return None
        
    # get the last model
    #last_model_version = max([model_version_info.version for model_version_info in model_version_infos])
    
   # model_uri = F"models:/{MLFLOW_MODEL_NAME}/{last_model_version}"
    #get the production model
    model_uri = F"models:/{MLFLOW_MODEL_NAME}/'Production'"
    #print("model_uri", model_uri)
    
    #define a path dir where to store locally the mlflow loaded model
    dst_path = config.MODELS_PATH
    # Extract the 'RUN_ID' field value for each ModelVersion
    run_id = [model_version.run_id for model_version in model_version_infos]
          
    #Fetch a model using model_uri using the default path where the MLflow logged the model
    loaded_model = mlflow.pyfunc.load_model(model_uri, dst_path=dst_path)
          
    # Define the destination path in MODELS_PATH  and 
    #load the best pretrained logged model                  
    path = os.path.join(dst_path, "weights/last.pt")
    print("Path", path)
        
    return path


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(**args):
    """Performs model prediction from given input data and parameters.

    Arguments:
        **args -- Arbitrary keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values json, png, pdf or mp4 file.
    """
    try:
        logger.debug("Predict with args: %s", args)    
        if args["model"] is None:
            #Load the (pretrained) model from mlflow registry if exists
            path = mlflow_fetch()
            if os.path.exists(path):
                args["model"] = utils.validate_and_modify_path(
                                path, config.MODELS_PATH
                            )    
                print("args_model", args["model"])
            else:
                # No model fetched from MLflow, use the default model
                args["model"] = utils.modify_model_name("yolov8n.pt", args["task_type"])
                #print("None")

        else:
            path = os.path.join(args["model"], "weights/best.pt")
            args["model"] = utils.validate_and_modify_path(
                path, config.MODELS_PATH
            )
        task_type = args["task_type"]

        if task_type == "seg" and args["augment"]:
            # https://github.com/ultralytics/ultralytics/issues/859
            raise ValueError(
                "augment for segmentation has not been supported yet"
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in [args["input"]]:
                shutil.copy(
                    f.filename, tmpdir + "/" + f.original_filename
                )

            args["input"] = [
                os.path.join(tmpdir, t) for t in os.listdir(tmpdir)
            ]
            result = aimodel.predict(**args)
            logger.debug("Predict result: %s", result)
            logger.info(
                "Returning content_type for: %s", args["accept"]
            )
            return responses.response_parsers[args["accept"]](
                result, **args
            )
    except Exception as err:
        raise HTTPException(reason=err) from err

def mlflow_update():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient(tracking_uri = os.environ["MLFLOW_TRACKING_URI"])            
    #check the latest version of the model            
    model_version_infos = client.search_model_versions(F"name = '{MLFLOW_MODEL_NAME}'")
    last_model_version = max([model_version_info.version for model_version_info 
                                                          in model_version_infos])
    model_uri = F"models:/{MLFLOW_MODEL_NAME}/{last_model_version}"
    print("model_uri", model_uri)

    # Add a description to the registered model   
    client.update_registered_model(
      name=MLFLOW_MODEL_NAME,
      description="This model detect players in a football play field "
    )

    # set tags, alias, update and delete them
    # create "champion" alias for version x of model "MLFLOW_MODEL_NAME"
    client.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", 
                                      last_model_version)

    # get a model version by alias
    print(f"\n Model version alias: ",client.get_model_version_by_alias(
                                        MLFLOW_MODEL_NAME, "champion"))

    # delete the alias
    #client.delete_registered_model_alias(MLFLOW_MODEL_NAME, "Champion")

    # Set registered model tag
    client.set_registered_model_tag(MLFLOW_MODEL_NAME, "task", 
                                    "detection")
    client.set_registered_model_tag(MLFLOW_MODEL_NAME, "author", 
                                    "lisana.berberi@kit.edu")
    client.set_registered_model_tag(MLFLOW_MODEL_NAME, "framework", 
                                    "pytorch")

    # Set a transition to the model: Production, Stage, Archived, None
    client.transition_model_version_stage(
      name=MLFLOW_MODEL_NAME,
      version=last_model_version,
      stage='None'    
    )

    # Get the current value of model transition   
    model_version_details = client.get_model_version(
      name=MLFLOW_MODEL_NAME,
      version=last_model_version,
    )
    model_stage = model_version_details.current_stage 
    print(F"The current model stage is: '{model_stage}'")

    # Set model transition to archive
    client.transition_model_version_stage(
      name=MLFLOW_MODEL_NAME,
      version=1,
      stage="Archived",
    )

    # Delete  model version    
    # client.delete_model_version(
    #  name=MLFLOW_MODEL_NAME,
    #  version=1,
    # )

def mlflow_logging(model, num_epochs, args):
    mlflow.end_run() #stop any previous active mlflow run
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
        data_path = config.DATA_PATH
        data_file_path = os.path.join(data_path, "data.yaml")
        
        # Use mlflow.log_artifact to log the contents of data.yaml
        mlflow.log_artifact(data_file_path, artifact_path="artifacts")
        
        ##           Log the Dataset                         ##
        

        
        # Print OpenCV version for debugging
        print(f"OpenCV Version: {cv2.__version__}")
        
        # Ensure that no random seed is set
        random.seed(None)
        
        
        # Specify the path to the folder containing images
        img_folder = os.path.join(config.DATA_PATH, "train/images")
        
        # List all files in the folder
        img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
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
                print(f"Error: Unable to read the image from {img_path}")
            else:
                # Continue with the rest of your code using the selected image
                resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
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
        run.log_params(vars(model.trainer.model.args))

        import numpy as np
        # Infer signature to a model, i.e. the input data used to feed the model 
        #and output of the trained model #
        #                                                               #
        
        # Assuming model is an instance of YOLO and img is an input image
        prediction_inf = model(img)

        train_inf_np = model.trainer.model.info()
        print ("\ntrain_inf_np", train_inf_np)
        train_inf_array = np.array(train_inf_np)
        
        # Convert the tuple to a dictionary
        train_inf = {"value": np.array(train_inf_np)} 
        print ("\ntrain_inf", train_inf)
        #print ("\npred_inf", prediction_inf)
       
        # Create list of detection dictionaries
        results_all = []

        for result in prediction_inf:
            # Assuming result is an ultralytics.engine.results.Results object
            data = result.boxes.data.cpu().tolist()
            h, w = result.orig_shape
            for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
                box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
                conf = row[-2]
                class_id = int(row[-1])
                name = result.names[class_id]
                detection_result = {'name': name, 'class': class_id, 'confidence': conf, 'box': box}
                if result.boxes.is_track:
                    detection_result['track_id'] = int(row[-3])  # track ID
                if result.masks:
                    x, y = result.masks.xy[i][:, 0], result.masks.xy[i][:, 1]  # numpy array
                    detection_result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
                if result.keypoints is not None:
                    x, y, visible = result.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                    detection_result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 
                                                     'visible': visible.tolist()}

                results_all.append(detection_result)

            output_file_path = os.path.join(config.MODELS_PATH, "detection_results.txt")
            with open(output_file_path, "w") as output_file:
            # Now write all results to the file after the loop
                for result_entry in results_all:
                    output_file.write(f"{str(result_entry)}\n")
                    

        # Use mlflow.log_artifact to log the contents of the detection_results
        mlflow.log_artifact(output_file_path, artifact_path="artifacts")
        
        # Use infer_signature with train_inf and results_all
        signature = infer_signature(train_inf, {"detections": results_all})  
        
        # Get the base directory of artifact_path
        base_dir = os.path.basename(str(model.trainer.save_dir))
        print("\nbase_dir", base_dir)
        
        model_t = mlflow.pyfunc.log_model(
            artifact_path="artifacts",
            python_model=mlflow.pyfunc.PythonModel(),
            signature=signature
        )
              
        #Log additional artifacts
        mlflow.log_artifacts(str(model.trainer.save_dir), artifact_path="artifacts")

        #create a new version of that model
        client = MlflowClient(tracking_uri = os.environ["MLFLOW_TRACKING_URI"])   
        run_id = active_run.info.run_id
        result = client.create_model_version(
                name=MLFLOW_MODEL_NAME,
                source=f"runs:/{run_id}/artifacts/{MLFLOW_MODEL_NAME}",
                run_id=run_id,
                )
        #register that version to Model Registry
        result = mlflow.register_model(
            f"runs:/{run_id}/artifacts/",
            MLFLOW_MODEL_NAME
        )      
        # Update model description, tags, alias, transitions.
        mlflow_update()      

    return {
        "artifact_path": args["name"],
        "artifact_uri": artifact_uri,
    }

@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**args):
    """
    Trains a yolov8 model using the specified arguments.
    
    Args:
        **args (dict): A dictionary of arguments for training the model
        defined in the schema.
    
    Returns:
        dict: A dictionary containing a success message and the path
        where the trained model was saved.
    
    Raises:
        HTTPException: If an error occurs during training.
    Note:
        - The `project` argument should correspond to the name of
        your project and should only include the project directory,
        not the full path.
        - The `name` argument specifies the subdirectory where the
        model will be saved within the project directory.
        - The `weights` argument can be used to load pre-trained
        weights from a file.
    """
    try:
        logger.info("Training model...")
        logger.debug("Train with args: %s", args)
        Enable_MLFLOW = args["Enable_MLFLOW"]
        settings.update({"mlflow": False})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # The project should correspond to the name of the project
        # and should only include the project directory, not the full path.
        args["project"] = "models"

        # The directory where the model will be saved after training
        # by joining the values of args["project"] and args["name"].
        args["name"] = timestamp

        # Modify the model name based on task type
        args["model"] = utils.modify_model_name(
            args["model"], args["task_type"]
        )
        # Check and update data path if necessary
        base_path = os.path.join(config.DATA_PATH, "processed")
        args["data"] = utils.validate_and_modify_path(
            args["data"], base_path
        )
        task_type = args["task_type"]
        if task_type in ["det", "seg"]:
            # Check and update data paths of val and training in config.yaml
            if not utils.check_paths_in_yaml(args["data"], base_path):
                raise ValueError(
                    "The path to the either train or validation "
                    "data does not exist. Please provide a valid path."
                )

        # Check if there are weights to load from an already trained model
        # Otherwise, load the pretrained model from the model registry

        if args["weights"] is not None:
            path = utils.validate_and_modify_path(
                args["weights"], config.MODELS_PATH
            )

            model = YOLO(path)

        else:
            model = YOLO(args["model"])

        device = args.get("device", "cpu")
        if device != "cpu" and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU mode.")
            device = "cpu"
        os.environ["WANDB_DISABLED"] = str(args["disable_wandb"])

        utils.pop_keys_from_dict(
            args,
            [
                "task_type",
                "disable_wandb",
                "weights",
                "device",
                "Enable_MLFLOW",
            ],
        )
        if Enable_MLFLOW:
            num_epochs = args["epochs"]
            model.train(exist_ok=True, device=device, **args)
            print("num_epochs", num_epochs)

            # Call the mlflow_logging function for MLflow-related operations
            return mlflow_logging(model, num_epochs, args)
        else:
            model.train(exist_ok=True, device=device, **args)
            return {
                f'The model was trained successfully and was saved to: \
                {os.path.join(args["project"], args["name"])}'
            }

    except Exception as err:
        logger.critical(err, exc_info=True)
        raise HTTPException(reason=err) from err

def main():
    """
    Runs above-described methods from CLI
    uses: python3 path/to/api/__init__.py method --arg1 ARG1_VALUE
     --arg2 ARG2_VALUE
    """
    method_dispatch = {
        "get_metadata": get_metadata,
        "predict": predict,
        "train": train,
    }

    chosen_method = args.method
    logger.debug("Calling method: %s", chosen_method)
    if chosen_method in method_dispatch:
        method_function = method_dispatch[chosen_method]

        if chosen_method == "get_metadata":
            results = method_function()
        else:
            logger.debug("Calling method with args: %s", args)
            del vars(args)["method"]
            if hasattr(args, "input"):
                file_extension = os.path.splitext(args.input)[1]
                args.input = UploadedFile(
                    "input",
                    args.input,
                    "application/octet-stream",
                    f"input{file_extension}",
                )
            results = method_function(**vars(args))
        print(json.dumps(results))
        logger.debug("Results: %s", results)
        return results
    else:
        print("Invalid method specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model parameters", add_help=False
    )
    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
        help='methods. Use "api.py method --help" to get more info',
        dest="method",
    )
    get_metadata_parser = subparsers.add_parser(
        "get_metadata", help="get_metadata method", parents=[parser]
    )

    predict_parser = subparsers.add_parser(
        "predict", help="commands for prediction", parents=[parser]
    )

    utils.add_arguments_from_schema(
        schemas.PredArgsSchema(), predict_parser
    )

    train_parser = subparsers.add_parser(
        "train", help="commands for training", parents=[parser]
    )

    utils.add_arguments_from_schema(
        schemas.TrainArgsSchema(), train_parser
    )

    args = cmd_parser.parse_args()

    main()

    """
    python3 api/__init__.py  train --model yolov8n.yaml\
    --task_type  det  --data /srv/yolov8_api/data/processed/seg/label.yaml
    python3 api/__init__.py  predict --input \
    /srv/yolov8_api/tests/data/det/test/cat1.jpg\
    --task_type  det --accept application/json
    """
