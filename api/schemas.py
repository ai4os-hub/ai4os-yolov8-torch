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

    model_name = ModelName(
        metadata={
            "description": "String/Path identification for models.",
        },
        required=True,
    )

    dataset = fields.String(
        metadata={
            "description": "Path to the training dataset.",
        },
        required=False,
    )

    epochs = fields.Integer(
        metadata={
            "description": "Number of epochs to train the model.",
        },
        required=False,
        load_default=1,
        validate=validate.Range(min=1),
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(responses.content_types),
    )
