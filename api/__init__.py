"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import os
import logging
import datetime
import tempfile
import shutil


from ultralytics import YOLO
from aiohttp.web import HTTPException

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
            "datasets": utils.ls_dirs(config.DATA_PATH / "processed"),
            "models": utils.ls_dirs(config.MODELS_PATH),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


def warm():
    """Function to run preparation phase before anything else can start.

    Raises:
        RuntimeError: Unexpected errors aim to stop model loading.
    """
    try:  # Call your AI model warm() method
        logger.info("Warming up the model.api...")
        aimodel.warm()
    except Exception as err:
        raise RuntimeError(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(**args):
    """Performs model prediction from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for prediction values.
        input_file -- File with data to perform predictions from model.
        accept -- Response parser type, default is json.
        **args -- Arbitrary keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """
    try:  

        logger.debug("Predict with args: %s", args)

        if args["model"] is None:
            args["model"] = utils.modify_model_name(
                "yolov8n.pt", args["task_type"]
            )
        else:
            args["model"] = os.path.join(
                config.MODELS_PATH, args["model"], "weights/best.pt"
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
            logger.info("Returning content_type for: %s", args["accept"])
            return responses.response_parsers[args["accept"]](
                result, **args
            )

    except Exception as err:
      raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**args):
    try:
        logger.info("Training model...")
        logger.debug("Train with args: %s", args)
        args["model"] = utils.modify_model_name(
            args["model"], args["task_type"]
        )
        if not os.path.isfile(args["data"]):
            args["data"] = os.path.join(
                config.DATA_PATH, "raw", args["data"]
            )
            assert os.path.isfile(args["data"]), \
                'The data file does not exist. Please provide a valid path.'
            assert utils.check_paths_in_yaml(args["data"]),\
                'The path to the either train or validation'\
                'data does not exist. Please provide a valid path.'   
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args["project"] = config.MODEL_NAME
        args["name"] = os.path.join("models", timestamp)

        if args["weights"] is not None:
            if os.path.isfile(args["weights"]):
                path = args["weights"]
            else:
                path = os.path.join(
                    config.MODELS_PATH, args["weights"]
                )

            model = YOLO(path)

        else:
            model = YOLO(args["model"])

        os.environ["WANDB_DISABLED"] = str(args["disable_wandb"])
        utils.pop_keys_from_dict(
            args, ["task_type", "disable_wandb", "weights"]
        )

        model.train(exist_ok=True, **args)

        return {
            f'The model was trained successfully \
            and was saved to: {os.path.join(args["project"], args["name"])}'
        }

    except Exception as err:
        raise HTTPException(reason=err) from err


if __name__ == "__main__":
    fields = schemas.TrainArgsSchema().fields

    args = {}
    for key, value in fields.items():
        print(key, value)
        if value.missing:
            args[key] = value.missing
    args["model"] = "yolov8s.pt"
    args["data"] = "/srv/yolov8_api/data/raw/seg/label.yaml"
    args["task_type"] = "seg"
    args["epochs"] = 5
    args["resume"] = True
    args[
        "weights"
    ] = "/srv/yolov8_api/models/20230831_074708/weights/last.pt"

    train(**args)
    fields = schemas.PredArgsSchema().fields
    from deepaas.model.v2.wrapper import UploadedFile

    args = {}

    for key, value in fields.items():
        print(key, value)
        if value.missing:
            args[key] = value.missing

    input = "/srv/yolov8_api/data/raw/PlantDoc.v1-resize-  \
        416x416.yolov8/train/images/02_-Rust-2017-207u24s_jpg.\
            f.cb22459400f68cb6d111d18db2f7d834.jpg"
    args["input"] = UploadedFile(
        "input", input, "application/octet-stream", "input.jpg"
    )
    args["model"] = None
    args["accept"] = "application/pdf"
    args["task_type"] = "seg"
