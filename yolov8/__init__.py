"""Package to create dataset, build training and prediction pipelines.

This file should define or import all the functions needed to operate the
methods defined at yolov8/api.py. Complete the TODOs
with your own code or replace them importing your own functions.
For example:
```py
from your_module import your_function as predict
from your_module import your_function as training
```
"""

# TODO: add your imports here
import logging
import yolov8.config as cfg
from ultralytics import YOLO
import yolov8.utils as utils


logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


# TODO: warm (Start Up)
# = HAVE TO MODIFY FOR YOUR NEEDS =
def warm(
    **kwargs,
):
    """Main/public method to start up the model"""
    # if necessary, start the model
    pass


# TODO: predict


def predict(
    **args,
):
    """Main/public method to perform prediction"""
    # if necessary, preprocess data

    # choose AI model, load weights

    # return results of prediction
    # Load a pretrained YOLOv8n model
    print("arg of prediction are", args)

    model = YOLO(args["model"])
    test_image_path = args["files"]
    results = []
    for image_path in test_image_path:
        print(
            "Evaluating:",
            image_path,
        )
        utils.remove_keys_from_dict(
            args,
            [
                "files",
                "accept",
                "task_type",
            ],
        )
        result = model.predict(
            image_path,
            **args,
        )
        logger.debug(f"[predict()]: {result}")
        results.append(result)
    return results


if __name__ == "__main__":
    args = {
        "files": ["/home/se1131/cat1.jpg"],
        "model": "yolov8n.pt",
        "imgsz": [704, 512],
        "conf": 0.25,
        "iou": 0.5,
        "show_labels": True,
        "show_conf": True,
        "augment": True,
        "classes": None,
        "show_boxes": True,
    }
    predict(**args)
