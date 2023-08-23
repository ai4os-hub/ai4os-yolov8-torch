# yolov8 

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/yolov8_api/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/yolov8_api/job/master)

Ultralytics YOLOv8 represents the forefront of object detection models, incorporating advancements from prior YOLO iterations while introducing novel features to enhance performance and versatility. YOLOv8 prioritizes speed, precision, and user-friendliness, positioning itself as an exceptional solution across diverse tasks such as object detection, tracking, instance segmentation, image classification, and pose estimation. Its refined architecture and innovations make it an ideal choice for cutting-edge applications in the field of computer vision.

# Adding DeepaaS API into the existing codebase
In this repository, we have integrated a DeepaaS API into the  Ultralytics YOLOv8, enabling the seamless utilization of this pipeline. The inclusion of the DeepaaS API enhances the functionality and accessibility of the code, making it easier for users to leverage and interact with the pipeline efficiently.

# Install the API and the external submodule requirement
To launch the API, first, install the package, and then run DeepaaS:
``` 
git clone --depth 1 https://git.scc.kit.edu/m-team/ai/yolov8_api.git
cd  yolov8_api
git submodule init
pip install -e .
```

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):

```bash
git clone https://git.scc.kit.edu/m-team/ai/yolov8_api
cd yolov8_api
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```

><span style="color:Blue">**Note:**</span> Before installing the API and submodule requirements, please make sure to install the following system packages: `gcc`, `unzip`, and `libgl1` as well. These packages are essential for a smooth installation process and proper functioning of the framework.
```
apt update
apt install -y unzip
apt install -y gcc
apt install -y libgl1
```

><span style="color:Blue">**Note:**</span>  The associated Docker container for this module can be found at: https://git.scc.kit.edu/m-team/ai/DEEP-OC-yolov8_api.git

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── yolov8_api
│   ├── README.md           <- Instructions on how to integrate your model with DEEPaaS.
│   ├── __init__.py         <- Makes <your-model-source> a Python module
│   ├── ...                 <- Other source code files
│   └── config.py           <- Module to define CONSTANTS used across the AI-model python package
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│   ├── external            <- Data from third party sources.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                 <- Folder to store your models
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials (if many user development),
│                             and a short `_` delimited description, e.g.
│                             `1.0-jqp-initial_data_exploration.ipynb`.
│
├── references             <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml         <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                <- tox file with settings for running tox; see tox.testrun.org
```

## Dataset Preparation
To train the yolov8 model, your annotations should be saved as yolo formats (.txt). Please organize your data in the following structure:
```
data
│
└── my_dataset
    ├── train_imgs
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── ...
    ├── train_labels
    │   ├── img1.txt
    │   ├── img2.txt
    │   ├── ...
    ├── valid_imgs
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   ├── ...
    ├── valid_labels
    │   ├── img_1.txt
    │   ├── img_2.txt
    │   ├── ...
    └── config.yaml

```

The `config.yaml` file contains the following information about the data:

```yaml
# Images and labels directory should be insade 'fasterrcnn_pytorch_api/data' directory.
TRAIN_DIR_IMAGES: 'my_dataset/train_imgs'
TRAIN_DIR_LABELS: 'my_dataset/train_labels'
VALID_DIR_IMAGES: 'my_dataset/valid_imgs'
VALID_DIR_LABELS: 'my_dataset/valid_labels'
# Class names.
CLASSES: [
    class1, class2, ...
]
# Number of classes.
NC: n
```
><span style="color:Blue">**Note:**</span>  If you have annotations files in Coco json format or Pascal VOC xml format, you can use the following script to convert them to the proper format for yolo. 

## Available Models

The Ultralytics YOLOv8 model can be used to train multiple tasks including classification, detection, segmentation and pose detection.
To train the model based on your project, you can select on of the task_type option in the training arguments and the corresponding model will be loaded and trained.
for each task, you can select the model arguments among the following options:

``` 
"yolov8n.yaml",
"yolov8n.pt",
"yolov8s.yaml",
"yolov8s.pt",
"yolov8m.yaml",
"yolov8m.pt",
"yolov8l.yaml",
"yolov8l.pt",
"yolov8x.yaml",
"yolov8x.pt",
```
`yolov8X.yaml` bulid a model from scratch and
`yolov8X.pt` load a pretrained model (recommended for training).

 