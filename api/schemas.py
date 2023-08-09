"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""
import marshmallow
from webargs import ValidationError, fields, validate

from yolov8_api.api import config, responses, utils


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
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    model_name = ModelName(
        metadata={
            "description": "String/Path identification for models.",
        },
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": "File with np.arrays for predictions.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(responses.content_types),
    )


# EXAMPLE of Training Args description
# = HAVE TO MODIFY FOR YOUR NEEDS =
class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

        model = fields.Str(
        description='Path to the model file, e.g., "yolov8n.pt" or "yolov8n.yaml"',
        required=False,
        allow_none=True
    )
    data = fields.Str(
        description='Path to the data file, e.g., "coco128.yaml"',
        required=False,
        allow_none=True
    )
    epochs = fields.Int(
        description='Number of epochs to train for',
        required=False
    )
    patience = fields.Int(
        description='Epochs to wait for no observable improvement for early stopping of training',
        required=False
    )
    batch = fields.Int(
        description='Number of images per batch (-1 for AutoBatch)',
        required=False
    )
    imgsz = fields.Int(
        description='Input images size as int for train and val modes, or list [w, h] for predict and export modes',
        required=False
    )
    save = fields.Bool(
        description='Save train checkpoints and predict results',
        required=True
    )
    save_period = fields.Int(
        description='Save checkpoint every x epochs (disabled if < 1)',
        required=True
    )
    cache = fields.Bool(
        description='True/ram, disk or False. Use cache for data loading',
        required=True
    )
    device = fields.Str(
        description='Device to run on, e.g., "cuda:0" or "cpu"',
        required=True,
        allow_none=True
    )
    workers = fields.Int(
        description='Number of worker threads for data loading (per RANK if DDP)',
        required=True
    )
    project = fields.Str(
        description='Project name',
        required=False,
        allow_none=True
    )
    name = fields.Str(
        description='Experiment name, results saved to \'project/name\' directory',
        required=False,
        allow_none=True
    )
    exist_ok = fields.Bool(
        description='Whether to overwrite existing experiment',
        required=True
    )
    pretrained = fields.Str(
        description='Whether to use a pretrained model (bool) or a model to load weights from (str)',
        required=True
    )
    optimizer = fields.Str(
        description='Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]',
        required=True
    )
    verbose = fields.Bool(
        description='Whether to print verbose output',
        required=True
    )
    seed = fields.Int(
        description='Random seed for reproducibility',
        required=True
    )
    deterministic = fields.Bool(
        description='Whether to enable deterministic mode',
        required=True
    )
    single_cls = fields.Bool(
        description='Train multi-class data as single-class',
        required=True
    )
    rect = fields.Bool(
        description='Rectangular training (mode=\'train\') or rectangular validation (mode=\'val\')',
        required=True
    )
    cos_lr = fields.Bool(
        description='Use cosine learning rate scheduler',
        required=True
    )
    close_mosaic = fields.Int(
        description='Disable mosaic augmentation for final epochs',
        required=True
    )
    resume = fields.Bool(
        description='Resume training from last checkpoint',
        required=True
    )
    amp = fields.Bool(
        description='Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check',
        required=True
    )
    fraction = fields.Float(
        description='Dataset fraction to train on (default is 1.0, all images in train set)',
        required=True
    )
    profile = fields.Bool(
        description='Profile ONNX and TensorRT speeds during training for loggers',
        required=True
    )
    overlap_mask = fields.Bool(
        description='Masks should overlap during training (segment train only)',
        required=True
    )
    mask_ratio = fields.Int(
        description='Mask downsample ratio (segment train only)',
        required=True
    )
    dropout = fields.Float(
        description='Use dropout regularization (classify train only)',
        required=True
    )
    lr0 = fields.Float(
        description='Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)',
        required=False
    )
    lrf = fields.Float(
        description='Final learning rate (lr0 * lrf)',
        required=False
    )
    momentum = fields.Float(
        description='SGD momentum/Adam beta1',
        required=False
    )
    weight_decay = fields.Float(
        description='Optimizer weight decay 5e-4',
        required=False
    )
    warmup_epochs = fields.Float(
        description='Warmup epochs (fractions ok)',
        required=False
    )
    warmup_momentum = fields.Float(
        description='Warmup initial momentum',
        required=False
    )
    warmup_bias_lr = fields.Float(
        description='Warmup initial bias lr',
        required=False
    )
    box = fields.Float(
        description='Box loss gain',
        required=False
    )
    cls = fields.Float(
        description='Cls loss gain (scale with pixels)',
        required=False
    )
    dfl = fields.Float(
        description='Dfl loss gain',
        required=False
    )
    pose = fields.Float(
        description='Pose loss gain',
        required=False
    )
    kobj = fields.Float(
        description='Keypoint obj loss gain',
        required=False
    )
    label_smoothing = fields.Float(
        description='Label smoothing (fraction)',
        required=False
    )
    nbs = fields.Int(
        description='Nominal batch size',
        required=False
    )
    hsv_h = fields.Float(
        description='Image HSV-Hue augmentation (fraction)',
        required=False
    )
    hsv_s = fields.Float(
        description='Image HSV-Saturation augmentation (fraction)',
        required=False
    )
    hsv_v = fields.Float(
        description='Image HSV-Value augmentation (fraction)',
        required=False
    )
    degrees = fields.Float(
        description='Image rotation (+/- deg)',
        required=False
    )
    translate = fields.Float(
        description='Image translation (+/- fraction)',
        required=False
    )
    scale = fields.Float(
        description='Image scale (+/- gain)',
        required=False
    )
    shear = fields.Float(
        description='Image shear (+/- deg)',
        required=False
    )
    perspective = fields.Float(
        description='Image perspective (+/- fraction), range 0-0.001',
        required=False
    )
    flipud = fields.Float(
        description='Image flip up-down (probability)',
        required=False
    )
    fliplr = fields.Float(
        description='Image flip left-right (probability)',
        required=False
    )
    mosaic = fields.Float(
        description='Image mosaic (probability)',
        required=False
    )
    mixup = fields.Float(
        description='Image mixup (probability)',
        required=False
    )
    copy_paste = fields.Float(
        description='Segment copy-paste (probability)',
        required=False
    )

