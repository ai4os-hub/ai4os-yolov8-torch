"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""

import marshmallow
from webargs import ValidationError, fields, validate

from . import config, responses, utils


class ModelName(fields.String):
    """Field that takes a string and validates against current available
    models at config.MODELS_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dir(config.MODELS_PATH):
            raise ValidationError(f"Checkpoint `{value}` not found.")
        return str(config.MODELS_PATH / value)


class Dataset(fields.String):
    """Field that takes a string and validates against current available
    data files at config.DATA_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dir(config.DATA_PATH / "processed"):
            raise ValidationError(f"Dataset `{value}` not found.")
        return str(config.DATA_PATH / "processed" / value)


class PredArgsSchema(marshmallow.Schema):
    class Meta:
        ordered = True

    files = fields.Field(
        required=True,
        type="file",
        location="form",
        metadata={
            "description": "Input an image or Video.\n"
            "accepted image formats: .bmo, .dng, .jpg, .jpeg, "
            ".mpo, .png, .tif, .tiff, .pfm, and .webp. \n"
            "accepted video formats: .asf, .avi, .gif, .m4v, .mkv,"
            ".mov, .mp4, .mpeg, .mpg, .ts, .wmv, .webm",
        },
    )

    model = fields.Str(
        metadata={
            "description": "The timestamp inside the 'models' directory "
            "indicates the time when you saved your trained model, "
            "The directory structure should resemble "
            "'models/your_timestamp/weights/best.pt'. "
            "To see the available timestamp, please run the "
            "get_metadata function and check model_local. "
            "If not provided, either a model from the MLflow "
            "registry will be loaded (if mlflow_fetch=true) "
            "or the pre-trained default model will be loaded "
            "depending on the task type."
        },
        load_default=config.YOLOV8_DEFAULT_WEIGHTS[0],
    )
    mlflow_fetch = fields.Boolean(
        metadata={
            "description": "Load a model from your MLflow model registry. "
            "Please set the MLFLOW_MODEL_NAME in the "
            "yolov8_api/config.py file to be loaded for "
            "prediction. Make sure you have passed the environment "
            "variables related to your MLflow (See readme)."
        },
        load_default=False,
    )
    task_type = fields.Str(
        metadata={
            "description": "The type of task for load the pretrained model:\n"
            '"det" for object detection model\n'
            '"seg" for object segmentation model\n'
            '"cls" for object classification model\n'
            '"obb" for  oriented bounding boxes object detection\n'
            'The default is "det"',
            "enum": config.YOLOV8_DEFAULT_TASK_TYPE,
        },
        load_default=config.YOLOV8_DEFAULT_TASK_TYPE[0],
    )

    imgsz = fields.List(
        fields.Int(),
        validate=validate.Length(max=2),
        metadata={
            "description": "image size as scalar or (h, w) list,"
            " i.e. (704, 512). Note: must be multiple of max stride 32"
        },
        load_default=[704, 512],
    )

    conf = fields.Float(
        metadata={
            "description": "object confidence threshold for detection"
        },
        load_default=0.25,
    )

    iou = fields.Float(
        metadata={
            "description": "intersection over union (IoU) threshold for NMS",
        },
        load_default=0.5,
    )

    show_labels = fields.Boolean(
        metadata={
            "description": "Show object labels in plots",
        },
        load_default=True,
    )
    show_conf = fields.Boolean(
        metadata={
            "description": "Show object confidence scores in plots."
            "if show_labels is False, show_conf is also False",
        },
        load_default=True,
    )

    augment = fields.Boolean(
        metadata={
            "description": "Apply image augmentation to prediction sources. "
            "augment for segmentation has not supported yet.",
        },
        load_default=False,
    )
    classes = fields.List(
        fields.Int(),
        metadata={
            "description": "Filter results by class, i.e. class=0, "
            "or class=[0,2,3]. Only detections belonging to the "
            "specified classes will be returned. Useful for focusing"
            " on relevant objects in multi-class detection tasks."
        },
        load_default=None,
    )

    show_boxes = fields.Boolean(
        metadata={
            "description": "Show boxes in segmentation predictions"
        },
        load_default=True,
    )
    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=False,
        load_default="application/json",
        validate=validate.OneOf(responses.content_types),
    )


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:
        ordered = True

    task_type = fields.Str(
        metadata={
            "description": "The type of task for the model:\n"
            '"det" for object detection model\n'
            '"seg" for object segmentation model\n'
            '"cls" for object classification model\n'
            '"obb" for  oriented bounding boxes object detection\n'
            'The default is "det"',
            "enum": config.YOLOV8_DEFAULT_TASK_TYPE,
        },
        load_default="det",
    )

    model = fields.Str(
        metadata={
            "description": " name of the model to train\n"
            '"yolov8X.yaml" bulid a model from scratch\n'
            '"yolov8X.pt" load a pretrained model (recommended for training)',
            "enum": config.MODEL_LIST,
        },
        required=True,
    )

    data = fields.Str(
        metadata={
            "description": "Path to the config data file (for seg and det) or "
            "data (cls task), e.g., 'root/path/to/mydata/data.yaml' or "
            "if it is in the 'path/to/ai4os-yolov8-torch/data/raw' just"
            "mydata/data.yaml"
        },
        allow_none=True,
        required=True,
    )
    epochs = fields.Int(
        metadata={
            "description": "Number of epochs to train for",
        },
        load_default=100,
    )
    patience = fields.Int(
        metadata={
            "description": "Epochs to wait for no observable improvement for"
            " early stopping of training"
        },
        load_default=10,
    )
    batch = fields.Int(
        metadata={
            "description": "Number of images per batch (-1 for AutoBatch)",
        },
        load_default=2,
    )
    imgsz = fields.Int(
        metadata={
            "description": "Input images size as int for train and val modes,"
        },
        load_default=640,
    )
    weights = fields.Str(
        metadata={
            "description": "If you want to initialize weights for training "
            "from a custom checkpoint, add the path to the checkpoint, "
            'for example: "timestamp/last.pt" where timestamp is in model'
            " directory or an absolute path to a checkpoint like "
            "'path/to/ckpt_dir/last.pt'",
        },
        load_default=None,
    )

    resume = fields.Bool(
        metadata={
            "description": "If the training was stopped before completing all"
            " epochs, you can resume training by setting resume=True"
            " to continue from the last checkpoint and put the path to the "
            "checkpoint into the weight argument. ",
            "enum": [True, False],
        },
        load_default=False,
    )

    save_period = fields.Int(
        metadata={
            "description": "Save checkpoint every x epochs (disabled if < 1)",
        },
        load_default=-1,
    )

    device = fields.Str(
        metadata={
            "description": 'Device to run on, e.g., "cuda:0" or "cpu"',
        },
        load_default="cuda:0",
    )
    workers = fields.Int(
        metadata={
            "description": "Number of worker threads for data loading"
            " (per RANK if DDP)",
        },
        load_default=4,
    )

    optimizer = fields.Str(
        metadata={
            "description": "Optimizer to use, choices="
            "[SGD, Adam, Adamax,AdamW, NAdam, RAdam, RMSProp, auto]",
            "enum": [
                "SGD",
                "Adam",
                "Adamax",
                "AdamW",
                "NAdam",
                "RAdam",
                "RMSProp",
                "auto",
            ],
        },
        load_default="auto",
    )

    seed = fields.Int(
        metadata={
            "description": "Random seed for reproducibility",
        },
        load_default=42,
    )
    deterministic = fields.Bool(
        metadata={
            "description": "Whether to enable deterministic mode",
            "enum": [True, False],
        },
        load_default=False,
    )
    single_cls = fields.Bool(
        metadata={
            "description": "Train multi-class data as single-class",
            "enum": [True, False],
        },
        load_default=False,
    )
    rect = fields.Bool(
        metadata={
            "description": "Rectangular training (mode='train') or rectangular"
            " validation (mode='val')",
            "enum": [True, False],
        },
        load_default=False,
    )

    fraction = fields.Float(
        metadata={
            "description": "Dataset fraction to train on (default is 1.0,"
            " all images in train set)",
        },
        load_default=1.0,
    )
    mask_ratio = fields.Int(
        metadata={
            "description": "Mask downsample ratio (segment train only)",
        },
        load_default=4,
    )
    dropout = fields.Float(
        metadata={
            "description": "Use dropout regularization (classify train only)",
        },
        load_default=0.0,
    )
    amp = fields.Bool(
        metadata={
            "description": "Automatic Mixed Precision (AMP) training,"
            " choices=[True, False], True runs AMP check",
            "enum": [True, False],
        },
        load_default=False,
    )
    cos_lr = fields.Bool(
        metadata={
            "description": "Use cosine learning rate scheduler",
            "enum": [True, False],
        },
        load_default=False,
    )
    lr0 = fields.Float(
        metadata={
            "description": "Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)",
        },
        load_default=0.01,
    )
    lrf = fields.Float(
        metadata={"description": "Final learning rate (lr0 * lrf)"},
        load_default=0.01,
    )
    momentum = fields.Float(
        metadata={
            "description": "SGD momentum/Adam beta1",
        },
        load_default=0.937,
    )
    weight_decay = fields.Float(
        metadata={"description": "Optimizer weight decay 5e-4"},
        load_default=0.0005,
    )
    warmup_epochs = fields.Float(
        metadata={
            "description": "Warmup epochs (fractions ok)",
        },
        load_default=3.0,
    )
    warmup_momentum = fields.Float(
        metadata={
            "description": "Warmup initial momentum",
        },
        load_default=0.8,
    )
    warmup_bias_lr = fields.Float(
        metadata={
            "description": "Warmup initial bias lr",
        },
        load_default=1.0,
    )
    close_mosaic = fields.Int(
        metadata={
            "description": "Disable mosaic augmentation for final epochs",
        },
        load_default=10,
    )

    box = fields.Float(
        metadata={"description": "Box loss gain"}, load_default=7.5
    )
    cls = fields.Float(
        metadata={
            "description": "Cls loss gain (scale with pixels)",
        },
        load_default=0.5,
    )
    dfl = fields.Float(
        metadata={
            "description": " Distribution Focal Loss gain",
        },
        load_default=1.5,
    )

    kobj = fields.Float(
        metadata={
            "description": "Keypoint obj loss gain",
        },
        load_default=1.0,
    )

    label_smoothing = fields.Float(
        metadata={
            "description": "Label smoothing (fraction)",
        },
        load_default=0.0,
    )
    nbs = fields.Int(
        metadata={"description": "Nominal batch size"},
        load_default=64,
    )
    hsv_h = fields.Float(
        metadata={
            "description": "Augmentation option: adjusts the hue "
            "of the image by a fraction of"
            " the color wheel, introducing color "
            "variability. Helps the model generalize "
            "across different lighting"
            " conditions. Range: 0.0 - 1.0"
        },
        load_default=0.015,
    )
    hsv_s = fields.Float(
        metadata={
            "description": "Augmentation option: Alters the "
            "saturation of the image by a fraction,"
            " affecting the intensity of colors. Useful for "
            "simulating different environmental conditions. "
            "Range: 0.0 - 1.0"
        },
        load_default=0.7,
    )
    hsv_v = fields.Float(
        metadata={
            "description": "Augmentation option: Modifies the value "
            "(brightness) of the "
            "image by a fraction, helping the model "
            "to perform well under various lighting"
            " conditions. Range: 0.0 - 1.0"
        },
        load_default=0.4,
    )
    degrees = fields.Float(
        metadata={
            "description": "Augmentation option: Rotates the"
            " image randomly within "
            "the specified degree range, improving"
            " the model's ability to recognize objects"
            " at various orientations. Range: -180 - +180"
        },
        load_default=0.0,
    )
    translate = fields.Float(
        metadata={
            "description": "Augmentation option: Translates the "
            "image horizontally and"
            " vertically by a fraction of the image size,"
            " aiding in learning to detect partially"
            " visible objects. Range: 0.0 - 1.0"
        },
        load_default=0.1,
    )
    scale = fields.Float(
        metadata={
            "description": "Augmentation option: Scales the"
            " image by a gain factor,"
            " simulating objects at different "
            "distances from the camera. "
            "Range: >=0.0"
        },
        load_default=0.5,
    )
    shear = fields.Float(
        metadata={
            "description": "Augmentation option: Shears the"
            " image by a specified "
            "degree, mimicking the effect of "
            "objects being viewed from different"
            " angles. Range: -180 - +180"
        },
        load_default=0.0,
    )
    perspective = fields.Float(
        metadata={
            "description": "Augmentation option: Applies a"
            " random perspective transformation"
            " to the image, enhancing the model's ability"
            " to understand objects in 3D space. "
            "Range 0-0.001"
        },
        load_default=0.0,
    )
    flipud = fields.Float(
        metadata={
            "description": "Augmentation option: Flips the"
            " image upside down "
            "with the specified probability,"
            " increasing the data variability "
            "without affecting the object's"
            " characteristics. Range 0.0-1.0"
        },
        load_default=0.0,
    )
    fliplr = fields.Float(
        metadata={
            "description": "Augmentation option: Flips the"
            " image left to right "
            "with the specified probability, "
            "useful for learning symmetrical "
            "objects and increasing dataset "
            "diversity. Range 0.0-1.0"
        },
        load_default=0.5,
    )
    mosaic = fields.Float(
        metadata={
            "description": "Augmentation option:Combines four"
            " training images "
            "into one, simulating different "
            "scene compositions and object "
            "interactions. Highly effective "
            "for complex scene understanding."
            "Range 0.0- 1.0"
        },
        load_default=0.1,
    )
    mixup = fields.Float(
        metadata={
            "description": "Augmentation option: Blends two "
            "images and their labels, "
            "creating a composite image. Enhances "
            "the model's ability to generalize by "
            "introducing label noise and visual "
            "variability. Range 0.0- 1.0"
        },
        load_default=0.0,
    )

    copy_paste = fields.Float(
        metadata={
            "description": "Augmentation option: Copies objects"
            " from one image "
            "and pastes them onto another, "
            "useful for increasing object "
            "instances and learning object "
            "occlusion. Range 0.0- 1.0"
        },
        load_default=0.0,
    )

    erasing = fields.Float(
        metadata={
            "description": "Augmentation option: Randomly erases"
            " a portion of the"
            " image during classification "
            "training, encouraging the model "
            "to focus on less obvious features "
            "for recognition."
            " Range 0.0- 0.9"
        },
        load_default=0.4,
    )

    crop_fraction = fields.Float(
        metadata={
            "description": "Augmentation option: Crops the "
            "classification image to a "
            "fraction of its size to emphasize "
            "central features and adapt to object "
            "scales, reducing background distractions."
            " Range 0.1- 1.0"
        },
        load_default=1.0,
    )
    auto_augment = fields.String(
        metadata={
            "description": "Automatically applies a predefined "
            "augmentation policy (randaugment, "
            "autoaugment, augmix), optimizing for"
            " classification tasks by diversifying "
            "the visual features.",
        },
        required=False,
        validate=validate.OneOf(
            ["randaugment", "autoaugment", "augmix"]
        ),
    )

    disable_wandb = fields.Bool(
        metadata={
            "description": "Whether disables wandb logging",
            "enum": [True, False],
        },
        load_default=True,
    )
    Enable_MLFLOW = fields.Bool(
        metadata={
            "description": "Whether eables MLFOW logging",
        },
        load_default=False,
    )


if __name__ == "__main__":
    from marshmallow import fields

    schema = TrainArgsSchema()
    for field_name, field_obj in schema.fields.items():
        arg_name = f"--{field_name}"

        arg_kwargs = {
            "help": field_name,
        }

        if isinstance(field_obj, fields.Bool):
            arg_kwargs["type"] = int
            print(f"field_object is {field_name}")
