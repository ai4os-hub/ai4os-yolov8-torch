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


from ultralytics import YOLO,  settings
from aiohttp.web import HTTPException
from deepaas.model.v2.wrapper import UploadedFile

import yolov8_api as aimodel
from yolov8_api.api import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


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
            args["model"] = utils.modify_model_name(
                "yolov8n.pt", args["task_type"]
            )
            print('model_name: ', args["model"])
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
        Enable_MLFLOW=  args['Enable_MLFLOW']
        settings.update({'mlflow':False})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # The project should correspond to the name of the project
        # and should only include the project directory, not the full path.
        args["project"] = 'models'

        # The directory where the model will be saved after training
        # by joining the values of args["project"] and args["name"].
        args["name"] = timestamp
        
      #  if Enable_MLFLOW:
           # settings.update({'mlflow':args['Enable_MLFLOW']})
         #   run_name=os.getenv('MLFLOW_RUN', default=args["name"])
          #  active_run = mlflow.start_run(run_name=run_name)
          #  run_id=active_run.info.run_id
            
         
            

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
            args, ["task_type", "disable_wandb", "weights", "device",  "Enable_MLFLOW"]
        )
        # The use of exist_ok=True ensures that the model will
        # be saved in the same path if resume=True.
       # model.train(exist_ok=True, device=device, **args)
        if Enable_MLFLOW:
            mlflow.end_run()
            mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
            with mlflow.start_run(run_name=args['name']):#(run_id=run_id):
                artifact_uri = mlflow.get_artifact_uri( )
                #print('The artifact_uri is', artifact_uri)
                
                model.train(exist_ok=True, device=device, **args)
                run, active_run = mlflow, mlflow.active_run()
                print("active run id",active_run.info.run_id)

                # logs params in mlflow
                run.log_params(vars(model.trainer.model.args))
                # log best fit and last fit in mlflow
                mlflow.log_artifact(model.trainer.last)
                mlflow.log_artifact(model.trainer.best)
                SANITIZE = lambda x: {k.replace('(', '').replace(')', ''): float(v) for k, v in x.items()}
                # logs metrics in mlflow
                mlflow.log_metrics(metrics=SANITIZE(model.trainer.label_loss_items(model.trainer.tloss, prefix='train')),
                           step=model.trainer.epoch)
                mlflow.log_metrics(metrics=SANITIZE(model.trainer.lr), step=model.trainer.epoch)
                mlflow.log_metrics(metrics=SANITIZE(model.trainer.metrics), step=model.trainer.epoch)

                # Log the PyTorch model to the artifact location specified by 'artifact_path'
                mlflow.pyfunc.log_model(artifact_path='model',
                                        artifacts={'model_path': str(model.trainer.save_dir)},
                                        python_model=mlflow.pyfunc.PythonModel())
                
                return {"artifact_path": args["name"], "artifact_uri": artifact_uri}
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
