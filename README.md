# Deep Species Detection

Provided by Ifremer, iMagine.

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/deep-species-detection/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/deep-species-detection/job/main/)

# Citizen science and data cleaning
In this repository, you will find a pipeline that cleans citizen science image datasets, and automatically trains a YoloV8 model on it.
You may also use this module to run inference on a pre trained YoloV8 model, specifically on 2 species : Buccinidae and Bythograeidae.
The pipeline converts bounding boxes from Deep Sea Spy format (lines, points, polygons) to regular bounding boxes (xmin, xmax, ymin, ymax). The conversion step is optional. It then unifies overlapping bounding boxes of each species, using the redundancy of citizen identifications as a 
There is 3 ways to use the pipeline :
    - DeepSeaLab.ipynb : step by step guide to clean the dataset and launch the Yolov8 training
    - Pipeline_txt.py : automatically cleans the dataset and launches the Yolov8 training with the arguments stored in config.txt
    - Deepaas API : easily visualize, customize and monitor the Yolov8 training. Very useful for inference.

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── deep-sea-lab            <- All of the data cleaning files
│   ├── DeepSeaLab.ipynb    <- Notebook pipeline for data cleaning & Yolov8 training
│   ├── Functions.py        <- Data processing file DeepSeaLab draws function from
│   ├── Pipeline_txt.py     <- Automatic pipeline to clean the data & train Yolov8
│   ├── config.txt          <- Configuration file for Pipeline.txt, which stores arguments to run the pipeline
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

# Running Deepaas for YoloV8 training and inference

First, install the package :

```
git clone https://github.com/ai4os-hub/deep-species-detection
cd  deep-species-detection
pip install -e .
```

You can then run the cleaning pipeline (see respective paragraphs for more information).

You can launch DeepaaS and run inference on a pre trained model by using :

```
deepaas-run --listen-ip 0.0.0.0
```

## Data requirements

The data required for the pipeline is to have a folder with your images, and a .csv file containing all of your annotations.

If your dataset is incomplete, it is better to remove incomplete rows rather than . Missing images will not cause problems with the pipeline.
For the conversion step, this pipeline converts data from Deep Sea Spy format :

|shapes  |x1 |y1 |x2 |y2 |polygon_values|name_img|species    |
|--------|---|---|---|---|--------------|--------|-----------|
|point   |59 |34 |NaN|NaN|NaN           |4366.jpg|Pycnogonid |
|lines   |761|451|859|364|NaN           |4366.jpg|Buccinidae |
|polygons|NaN|NaN|NaN|NaN|[{\x":282,"y":115},{"x":15,"y":538},{"x":50,"y":679},{"x":285,"y":497}]|4366.jpg|Mussels coverage|

To a regular format :

|xmin |xmax |ymin |ymax |name_img|species    |
|-----|-----|-----|-----|--------|-----------|
|69   |49   |24   |44   |4366.jpg|Pycnogonid |
|761  |859  |364  |451  |4366.jpg|Buccinidae |
|15   |285  |115  |679  |4366.jpg|Mussels coverage|

If your data is already in this format, you can skip the conversion steps (more details in the cleaning sections).

The pipeline expects image resolution of 1920x1080. You can input images with a different size by changing width_images and height_images in the beginning of Functions.py. If you have images of varying resolutions, you can modify the functions so that they take in the type of image you have.


# Cleaning from DeepSeaLab.ipynb
The **Notebook** is a ready-to-use, step by step cleaning file that you can change based on your needs/dataset. This option is better for a first use of the module since the notebook brings more context and guidance to the cleaning steps.
You can skip the conversion steps by skipping the cells that won't help your case. Arguments are pre filled to show you what's expected from the user.
You can launch the notebook by double clicking the file on the left, inside the deep-sea-lab folder.

# Cleaning from Pipeline_txt.py
Python script that automatically runs all the functions needed for the cleaning and analysis of citizen science datasets.
This option is more straight forward than using the python notebook. Detailed explanations of the functions can be found in the file in itself. You can modify them and use them as the basis for your work.
This file uses arguments in the config.txt file, which are pre filled to show you what's expected from the user.
You can modify arguments based on your needs.
### Arguments and usage
Paths to your data is required as the first arguments in the config.txt file
```
# Paths
# csv access :
path_csv=/storage/export.csv
# images :
path_imgs=/storage/Image_dsp/
# save
path_save=/storage/save
```
Those paths are by default, we are expecting you to have connected your Nextcloud account to the iMagine platform.
For the path_save, we recommend you to save on the deployment first, and then copying your training dataset/results afterwards. It is way faster this way.
path_imgs should refer to the folder containing all of your images.

In the config.txt file, if your dataset only contains annotated lines, it is expected :
```
# Dataset options
polygons=false
points=false
lines=True
```
If your dataset only contains annotated polygons and points, it is expected :
```
# Dataset options
polygons=True
points=True
lines=false
```
If your dataset is already in the regular format cited in the data requirements (therefore, you do not need the data conversion), you can put everything in false :
```
# Dataset options
polygons=false
points=false
lines=false
```
The pipeline will still create the training dataset from your data, and will train YoloV8 on it.

In the config.txt file, you can change the YoloV8 training parameters. They are set by their default value (from https://github.com/ultralytics/ultralytics).
You may also change the hyperparameters, but it is recommended to do so only if you know what each modified argument does to the training step.
If you wish to train the hyperparameters on a specific model, you can do so by running the last cell in the DeepSeaLab Notebook.

# Adding DeepaaS API into the existing codebase
In this repository, we have integrated a DeepaaS API into the  Ultralytics YOLOv8, enabling the seamless utilization of this pipeline. The inclusion of the DeepaaS API enhances the functionality and accessibility of the code, making it easier for users to leverage and interact with the pipeline efficiently.

# Environment variables settings
"In `./api/config.py` you can configure several environment variables:

- `DATA_PATH`: Path definition for the data folder; the default is './data'.
- `MODELS_PATH`: Path definition for saving trained models; the default is './models'.
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

