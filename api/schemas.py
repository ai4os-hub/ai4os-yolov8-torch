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


# EXAMPLE of Prediction Args description
# = HAVE TO MODIFY FOR YOUR NEEDS =


class PredArgsSchema(marshmallow.Schema):
    class Meta: 
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    input = fields.Field(
        required=True,
        type="file",
        location="form",
        description="Input an image or Video.\n"
        "accepted image formats: .bmo, .dng, .jpg, .jpeg, "
        ".mpo, .png, .tif, .tiff, .pfm, and.webp. \n"
        "accepted video formats: .asf, .avi, .gif, .m4v, .mkv, .mov,.mpv4, .mpeg, "
        ".mpeg, .mpg, .ts, .wmv, webm",
    )

    model = fields.Str(
        description="The timestamp when you saved your trained model, if not provided,"
        "the pre-trained YOLOv8n model will be loaded based on the selected task_type.",
        missing=None,
    )
    task_type = fields.Str(
        description="The type of task for load the pretrained model:\n"
        '"det" for object detection model\n'
        '"seg" for object segmentation model\n'
        '"cls" for object classification model\n'
        'The default is "det"',
        required=False,
        missing="det",
        enum=["det", "seg", "cls", "pose"],
    )

    conf = fields.Float(
        description="object confidence threshold for detection",
        missing=0.25,
    )

    iou = fields.Float(
        description="intersection over union (IoU) threshold for NMS",
        missing=0.5,
    )

    show_labels = fields.Boolean(
        description="Show object labels in plots", missing=True
    )
    show_conf = fields.Boolean(
        description="Show object confidence scores in plots."
        "if show_labels is False, show_conf is also False",
        missing=True,
    )
    #  vid_stride = fields.Int(
    #      description='Video frame-rate stride',
    #     missing=1
    # )
    #line_width = fields.Int(
   #     description="Line width of the bounding boxes",
    #    required=False,
   #     missing=None,
   # )
#    visualize = fields.Boolean(
#        description="Visualize model features", missing=False
#    )
    augment = fields.Boolean(
        description="Apply image augmentation to prediction sources",
        missing=False,
    )
#    agnostic_nms = fields.Boolean(
#        description="Class-agnostic NMS", missing=False
#    )
    classes = fields.Field(
        description="Filter results by class, i.e. class=0, or class=[0,2,3]",
        required=False,
        missing=None,
    )
#    retina_masks = fields.Boolean(
#        description="Use high-resolution segmentation masks",
#        missing=False,
#    )
    boxes = fields.Boolean(
        description="Show boxes in segmentation predictions",
        missing=True,
    )
    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(responses.content_types),
    )


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    task_type = fields.Str(
        description="The type of task for the model:\n"
        '"det" for object detection model\n'
        '"seg" for object segmentation model\n'
        '"cls" for object classification model\n'
        'The default is "det"',
        required=False,
        missing="det",
        enum=["det", "seg", "cls"],
    )

    model = fields.Str(  # FIXME
        description=" name of the model to train\n"
        '"yolov8X.yaml" bulid a model from scratch\n'
        '"yolov8X.pt" load a pretrained model (recommended for training)',
        required=True,
        enum=config.MODEL_LIST
    )

    data = fields.Str(
        description='Path to the config data file, e.g., "coco128.yaml"',
        required=True,
        allow_none=True,
    )
    epochs = fields.Int(
        description="Number of epochs to train for",
        required=False,
        missing=100,
    )
    patience = fields.Int(
        description="Epochs to wait for no observable improvement for early stopping of training",
        required=False,
        missing=10,
    )
    batch = fields.Int(
        description="Number of images per batch (-1 for AutoBatch)",
        required=False,
        missing=2,
    )
    imgsz = fields.Int(
        description="Input images size as int for train and val modes, or list [w, h] for predict and export modes",
        required=False,
        missing=640,
    )
    save = fields.Bool(
        description="Save train checkpoints and predict results",
        missing=True,
        enum=[True, False],
    )

    save_period = fields.Int(
        description="Save checkpoint every x epochs (disabled if < 1)",
        missing=-1,
    )
    cache = fields.Bool(
        description="True/ram, disk or False. Use cache for data loading",
        required=False,
        enum=[True, False],
        missing=False,
    )
    device = fields.Str(
        description='Device to run on, e.g., "cuda:0" or "cpu"',
        missing="cuda:0",
    )
    workers = fields.Int(
        description="Number of worker threads for data loading (per RANK if DDP)",
        missing=4,
    )
    
    weights = fields.Str(
        description='If you want to initialize weights for training from a checkpoint, '
                    'add the path to the checkpoint, '
                    'for example: "timestamp/last.pt" where timestamp is '
                    'in the model directory, or a complete path to a checkpoint.'
                    ' If you want to resume training, set the "resume" field to True '
                    'to resume training from the last checkpoint.',
        missing=None,
    )
    
    
    resume = fields.Bool(
        description="Resume training from the last checkpoint",
        required=False,
        missing=False,
        enum=[True, False],  # Use a list for the enum
    )
    pretrained = fields.Str(
        description="Whether to use a pretrained model (bool) or a model to load weights from (str)",
        missing=True,
        required=False,
    )

    optimizer = fields.Str(
        description="Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]",
        missing="auto",
        required=False,
        enum=[
            "SGD",
            "Adam",
            "Adamax",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
            "auto",
        ],
    )
    verbose = fields.Bool(
        description="Whether to print verbose output",
        required=False,
        missing=False,
        enum=[True, False],
    )
    seed = fields.Int(
        description="Random seed for reproducibility", missing=42
    )
    deterministic = fields.Bool(
        description="Whether to enable deterministic mode",
        missing=True,
        enum=[True, False],
    )
    single_cls = fields.Bool(
        description="Train multi-class data as single-class",
        missing=False,
        enum=[True, False],
    )
    rect = fields.Bool(
        description="Rectangular training (mode='train') or rectangular validation (mode='val')",
        missing=False,
        enum=[True, False],
    )
    cos_lr = fields.Bool(
        description="Use cosine learning rate scheduler",
        missing=False,
        enum=[True, False],
    )

    overlap_mask = fields.Bool(
        description="Masks should overlap during training (segment train only)",
        required=False,
        missing=False,
        enum=[True, False],
    )
    mask_ratio = fields.Int(
        description="Mask downsample ratio (segment train only)",
        required=False,
        missing=4,
    )
    dropout = fields.Float(
        description="Use dropout regularization (classify train only)",
        required=False,
        missing=0.0,
    )
    lr0 = fields.Float(
        description="Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)",
        required=False,
        missing=0.01,
    )
    lrf = fields.Float(
        description="Final learning rate (lr0 * lrf)",
        required=False,
        missing=0.01,
    )
    momentum = fields.Float(
        description="SGD momentum/Adam beta1",
        required=False,
        missing=0.937,
    )
    weight_decay = fields.Float(
        description="Optimizer weight decay 5e-4",
        required=False,
        missing=0.0005,
    )
    warmup_epochs = fields.Float(
        description="Warmup epochs (fractions ok)",
        required=False,
        missing=3.0,
    )
    warmup_momentum = fields.Float(
        description="Warmup initial momentum",
        required=False,
        missing=0.8,
    )
    warmup_bias_lr = fields.Float(
        description="Warmup initial bias lr",
        required=False,
        missing=1.0,
    )
    close_mosaic = fields.Int(
        description="Disable mosaic augmentation for final epochs",
        missing=10,
    )

    amp = fields.Bool(
        description="Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check",
        required=False,
        missing=False,
        enum=[True, False],
    )
    fraction = fields.Float(
        description="Dataset fraction to train on (default is 1.0, all images in train set)",
        required=False,
        missing=1.0,
    )
    profile = fields.Bool(
        description="Profile ONNX and TensorRT speeds during training for loggers",
        required=False,
        missing=False,
        enum=[True, False],
    )
    box = fields.Float(
        description="Box loss gain", required=False, missing=7.5
    )
    cls = fields.Float(
        description="Cls loss gain (scale with pixels)",
        required=False,
        missing=0.5,
    )
    dfl = fields.Float(
        description="Dfl loss gain", required=False, missing=1.5
    )
    pose = fields.Float(
        description="Pose loss gain", required=False, missing=12.0
    )
    kobj = fields.Float(
        description="Keypoint obj loss gain",
        required=False,
        missing=1.0,
    )
    label_smoothing = fields.Float(
        description="Label smoothing (fraction)",
        required=False,
        missing=0.0,
    )
    nbs = fields.Int(
        description="Nominal batch size", required=False, missing=64
    )
    hsv_h = fields.Float(
        description="Image HSV-Hue augmentation (fraction)",
        required=False,
        missing=0.015,
    )
    hsv_s = fields.Float(
        description="Image HSV-Saturation augmentation (fraction)",
        required=False,
        missing=0.7,
    )
    hsv_v = fields.Float(
        description="Image HSV-Value augmentation (fraction)",
        required=False,
        missing=0.4,
    )
    degrees = fields.Float(
        description="Image rotation (+/- deg)",
        required=False,
        missing=0.0,
    )
    translate = fields.Float(
        description="Image translation (+/- fraction)",
        required=False,
        missing=0.5,
    )
    scale = fields.Float(
        description="Image scale (+/- gain)",
        required=False,
        missing=0.5,
    )
    shear = fields.Float(
        description="Image shear (+/- deg)",
        required=False,
        missing=0.0,
    )
    perspective = fields.Float(
        description="Image perspective (+/- fraction), range 0-0.001",
        required=False,
        missing=0.0,
    )
    flipud = fields.Float(
        description="Image flip up-down (probability)",
        required=False,
        missing=0.0,
    )
    fliplr = fields.Float(
        description="Image flip left-right (probability)",
        required=False,
        missing=0.5,
    )
    mosaic = fields.Float(
        description="Image mosaic (probability)",
        required=False,
        missing=1.0,
    )
    mixup = fields.Float(
        description="Image mixup (probability)",
        required=False,
        missing=0.0,
    )
    copy_paste = fields.Float(
        description="Segment copy-paste (probability)",
        required=False,
        missing=0.0,
    )

    exist_ok = fields.Bool(
        description="Whether to overwrite existing experiment",
        missing=False,
        required=False,
        enum=[True, False],
    )

    exist_ok = fields.Bool(
        description="Whether to overwrite existing experiment",
        missing=False,
        required=False,
        enum=[True, False],
    )
    disable_wandb = fields.Bool(
        description="Whether disables wandb logging",
        missing=True,
        enum=[True, False],
    )
