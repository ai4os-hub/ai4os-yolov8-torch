"""Module to define CONSTANTS used across the AI-model package.

This module is used to define CONSTANTS used across the AI-model package.
Do not misuse this module to define variables that are not CONSTANTS or
that are exclusive to the `api` package. You can use the `config.py`
inside `api` to define exclusive CONSTANTS related to your interface.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""

import logging
import os


# configure logging:
# logging level across various modules can be setup via USER_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("USER_LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())

# EXAMPLE on how to load environment variables
MY_PARAMETER_INT = int(os.getenv("MY_PARAMETER_INT", default="10"))


MLFLOW_RUN_DESCRIPTION = os.getenv(
    "MODEL_NAME", default="object detection using yolov8"
)
MLFLOW_AUTHOR = os.getenv("MLFLOW_AUTHOR", default="Fahimeh/Lisana")
MLFLOW_MODEL_NAME = (
    "yolov8_PlayersDetection"  # "CHOOSE YOUR MODEL NAME HERE"
)

# Path definition for the pre-trained models
