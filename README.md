# Deep Species Detection

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/deep-species-detection/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/deep-species-detection/job/main/)

Ultralytics YOLOv8 represents the forefront of object detection models, incorporating advancements from prior YOLO iterations while introducing novel features to enhance performance and versatility. YOLOv8 prioritizes speed, precision, and user-friendliness, positioning itself as an exceptional solution across diverse tasks such as object detection, ororiented bounding boxes detection, tracking, instance segmentation, and image classification. Its refined architecture and innovations make it an ideal choice for cutting-edge applications in the field of computer vision.

# Adding DeepaaS API into the existing codebase
In this repository, we have integrated a DeepaaS API into the  Ultralytics YOLOv8, enabling the seamless utilization of this pipeline. The inclusion of the DeepaaS API enhances the functionality and accessibility of the code, making it easier for users to leverage and interact with the pipeline efficiently.

# Install the API 
To launch the API, first, install the package, and then run DeepaaS:
``` bash
git clone --depth 1 https://github.com/ai4os-hub/deep-species-detection.git
cd  deep-species-detection
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```

><span style="color:Blue">**Note:**</span> Before installing the API, please make sure to install the following system packages: `gcc`, `libgl1`, and `libglib2.0-0` as well. These packages are essential for a smooth installation process and proper functioning of the framework.
```
apt update
apt install -y gcc
apt install -y libgl1
apt install -y libglib2.0-0
```

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── yolov8
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

# Environment variables settings
"In `./api/config.py` you can configure several environment variables:

- `DATA_PATH`: Path definition for the data folder; the default is './data'.
-  `MODELS_PATH`: Path definition for saving trained models; the default is './models'.
- `REMOTE_PATH`: Path to the remote directory containing your trained models. Rclone uses this path for downloading or listing the trained models.
- `YOLOV8_DEFAULT_TASK_TYPE`: Specify the default tasks related to your work among detection (det), segmentation (seg), and classification (cls).
- `YOLOV8_DEFAULT_WEIGHTS`: Define default timestamped weights for your trained models to be used during prediction. If no timestamp is specified by the user during prediction, the first model in YOLOV8_DEFAULT_WEIGHTS will be used. If it is set to None, the Yolov8n trained on coco/imagenet will be used. Format them as timestamp1, timestamp2, timestamp3, ..."

# Track your experiments with Mlfow
If you want to use Mflow to track and log your experiments, you should first set the following environment variables:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_EXPERIMENT_NAME` (for the first experiment)

optional options:

- `MLFLOW_RUN`
- `MLFLOW_RUN_DESCRIPTION`
- `MLFLOW_AUTHOR`
- `MLFLOW_MODEL_NAME`: This name will be used as the name for your model registered in the MLflow Registry.
- Then you should set the argument `Enable_MLFLOW` to `True` during the execution of the training.


# Dataset Preparation
- Detection (det), oriented bounding boxes detection (obb) and Segmentation Tasks (seg):

    - To train the yolov8 model, your annotations should be saved as yolo formats (.txt). Please organize your data in the following structure:
```

│
└── my_dataset
    ├──  train
    │    ├── imgs
    │    │   ├── img1.jpg
    │    │   ├── img2.jpg
    │    │   ├── ...
    │    ├── labels
    │    │   ├── img1.txt
    │    │   ├── img2.txt
    │    │   ├── ...
    │    
    ├── val    
    │    ├── imgs
    │    │   ├── img_1.jpg
    │    │   ├── img_2.jpg
    │    │   ├── ...
    │    ├── labels
    │    │   ├── img_1.txt
    │    │   ├── img_2.txt
    │    │   ├── ...
    │    
    ├── test    
    │    ├── imgs
    │    │   ├── img_1.jpg
    │    │   ├── img_2.jpg
    │    │   ├── ...
    │    ├── labels
    │    │   ├── img_1.txt
    │    │   ├── img_2.txt
    │    │   ├── ...
    │    
    └── config.yaml
```

The `config.yaml` file contains the following information about the data:

```yaml
# Images and labels directory should be insade 'fasterrcnn_pytorch_api/data' directory.
train: 'path/to/my_dataset/train/imgs'
val: 'path/to/my_dataset/val/imgs'
test: 'path/to/my_dataset/test/imgs' #optional
# Class names.
names: 
    0: class1, 
    1: class2,
     ...

# Number of classes.
NC: n
```
The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.
`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

><span style="color:Blue">**Note:**</span>The train and val path should be a complete path or relative from
data directory e.g. `root/path/to/mydata/train/images` or if it is in the `path/to/deep-species-detection/data/raw` just 
`mydata/train/images`


-  Classification Task (cls):
For the classification task, the dataset format should be as follows:
```
data/
|-- class1/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- class2/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- class3/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- ...
```
><span style="color:Blue">**Note:**</span>  For the classification task, you don't need the config.yaml file. Simply provide the path to the data directory in the data argument for training.

><span style="color:Blue">**Note:**</span>  If you have annotations files in Coco json format or Pascal VOC xml format, you can use the following script to convert them to the proper format for yolo. 
``` 
deep-species-detection/yolov8/seg_coco_json_to_yolo.py #for segmentation
deep-species-detection/yolov8/preprocess_ann.py #For detection
``` 
# Available Models

The Ultralytics YOLOv8 model can be used to train multiple tasks including classification, detection, and segmentatio.
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

# Launching the API

To train the model, run:
```
deepaas-run --listen-ip 0.0.0.0
```
Then, open the Swagger interface, change the hyperparameters in the train section, and click on train.

><span style="color:Blue">**Note:**</span>  Please note that the model training process may take some time depending on the size of your dataset and the complexity of your custom backbone. Once the model is trained, you can use the API to perform inference on new images.

><span style="color:Blue">**Note:**</span> Augmentation Settings:
among the training arguments, there are options related to augmentation, such as flipping, scaling, etc. The default values are set to automatically activate some of these options during training. If you want to disable augmentation entirely or partially, please review the default values and adjust them accordingly to deactivate the desired augmentations.
# Inference Methods

You can utilize the Swagger interface to upload your images or videos and obtain the following outputs:

- For images:

    - An annotated image highlighting the object of interest with a bounding box.
    - A JSON string providing the coordinates of the bounding box, the object's name within the box, and the confidence score of the object detection.

- For videos:

    - A video with bounding boxes delineating objects of interest throughout.
    - A JSON string accompanying each frame, supplying bounding box coordinates, object names within the boxes, and confidence scores for the detected objects.

