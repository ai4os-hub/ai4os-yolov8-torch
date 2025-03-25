# 🚀 YOLOv8 Hyperparameter Optimization with Hydra, Optuna & MLflow

This repository provides a framework to **train and optimize YOLOv8 models** using **Optuna**, **Hydra**, and **MLflow**. It is designed to streamline the process of configuring, training, validating, and evaluating object detection models with automated tracking and logging.

## 📁 Project Structure
```bash
├── configs 
│   └── basic_train_params_pretrained.yaml  # Hydra config with model, training, and logging params 
├── hpo.py                                  # Main script for training and hyperparameter optimization 
├── requirements.txt                        # Python dependencies

```
## ⚙️ Features

- 🔧 Hyperparameter optimization with **Optuna**
- 🧠 Config management via **Hydra**
- 📈 Metrics & artifact logging using **MLflow**
- 📦 Supports training/validation with **Ultralytics YOLOv8**

---
## 🧰 Installation
```bash
pip install -r requirements.txt
```
## 🧾 Configuration
In `config/basic_train_params_pretrained.yaml` you can
### 🔧 Defaults & Sweeper:
You do not need to change this part.
```bash
defaults:
  - override hydra/sweeper: optuna
```
### 📁 MLflow Settings
You can change the name of your MLflow experiment here.

```bash

mlflow_project: hpo_yolov8_kitti  # Name of the MLflow experiment
mlflow_parent: basic_train_params_loss_30epochs_640imgsz  # The name of the parent run in MLflow
```
### ⚙️ Hydra + Optuna Sweep Configuration
Here you can change the Hydra configuration.


```bash

hydra:
  sweep:
    dir: tmp_multirun  # Directory where multi-run outputs are stored
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper  # Optuna sweeper plugin
    sampler:
      seed: 815  # Seed for reproducibility
    direction: maximize  # Objective direction (e.g. maximize validation fitness)
    n_trials: 30  # Number of trials to run
    n_jobs: 1  # Number of jobs to run in parallel
    params:  # Hyperparameters to optimize. you can add any hyperparameters you want here
      train_params.optimizer: choice(SGD, Adam, AdamW)
      train_params.lr0: interval(0.001, 0.2)
      train_params.momentum: interval(0.6, 0.999)
      train_params.weight_decay: interval(0.00001, 0.001)
      train_params.box: interval(0.0, 10.0)
      train_params.cls: interval(0.0, 10.0)
      train_params.dfl: interval(0.0, 10.0)
```

### 🏋️‍♂️ Training Parameters
Here you can select the version of the YOLO model you want and the path to the `dataset.yaml` configuration..
```bash

train_params:
  model: yolov8m.pt        # Pretrained model to use (YOLOv8-m)
  epochs: 40               # Number of epochs to train
  patience: 10             # Early stopping patience
  box: 7.5                 # Box loss gain
  cls: 0.5                 # Class loss gain
  dfl: 1.5                 # Distribution Focal Loss gain
  optimizer: 'auto'        # Optimizer (overwritten by sweep)
  cos_lr: False            # Use cosine learning rate schedule
  lr0: 0.01                # Initial learning rate
  momentum: 0.937          # Momentum (used with SGD)
  weight_decay: 0.0005     # Weight decay
  data: &data path/to/data/kitti.yaml  # Path to dataset config
  batch: &batch 16         # Batch size
  imgsz: &imgsz 640        # Image size
  save: True               # Save model checkpoints
  cache: True              # Cache images for faster training
  device: &device 0        # GPU device
  workers: 8               # Number of data loading workers
  rect: &rect True         # Use rectangular training batches
  plots: &plots True       # Save training plots
```
### 🧪 Validation Parameters
```bash

val_test_params:
  data: *data              # Use same dataset as training
  imgsz: *imgsz            # Same image size
  batch: *batch            # Same batch size
  device: *device          # Same GPU
  plots: *plots            # Generate plots
  rect: *rect              # Use rectangular validation batches
```
## How to run
To launch training and start the Optuna hyperparameter optimization:

```bash
python3 hpo.py --multirun
```
✅ This will:

- Run training and validation across **30 trials**
- Optimize the defined hyperparameters using **Optuna**
- Log all **metrics**, **configs**, and **artifacts** to **MLflow**
- Store outputs in the **`tmp_multirun/`** directory

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Hydra – Elegant Configuration Management](https://hydra.cc/)
- [Hydra Optuna Sweeper Plugin](https://github.com/facebookresearch/hydra/tree/main/plugins/hydra_optuna_sweeper)
- [Optuna – Hyperparameter Optimization Framework](https://optuna.org/)
- [MLflow – Open Source Experiment Tracking](https://mlflow.org/)
s