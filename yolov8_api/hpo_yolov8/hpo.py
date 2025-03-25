from ultralytics import YOLO, settings
import hydra
from omegaconf import DictConfig
import mlflow
import torch
import os
import datetime
import re
import yaml
import shutil
import pynvml
import numpy
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

HYDRA_OPTUNA_CONFIG_NAME: str = "basic_train_params_pretrained.yaml"

def drop_None(original_dict: dict[str]):
    return {k: v for k, v in original_dict.items() if v not in [None, 'None']}


def drop_project_and_name(original_dict: dict[str]):
    return {k: v for k, v in original_dict.items() if k not in ['project', 'name']}
    

def get_yolo_base_path(cgf_dict: dict[str]):
    # return f"{cgf_dict['project']}/{cgf_dict['name']}"
    return "runs/detect"
    
    
def clean_string(input_string: str):
    # Use regex to keep only alphanumeric characters, _, -, and /
    return re.sub(r'[^a-zA-Z0-9._\-/]', '', input_string)
    

def get_logging_key(key: str, prefix1: str, prefix2: str = None):
    separator = ":"
    prefix = prefix1
    if prefix2:
        prefix += separator + prefix2
    return prefix + separator + clean_string(key)


def log_param(key: str, value):
    mlflow.log_param(key, value)
    print(f"logged param {key}: {value}") 


def log_metric(key: str, value):
    mlflow.log_metric(key, value)
    print(f"logged metric {key}: {value}") 


def log_artifact(parent_dir: str, artifact_name: str, artifact_type: str, prefix2: str = None):
    key = get_logging_key(artifact_name, artifact_type, prefix2)
    try:
        mlflow.log_artifact(local_path=f"{parent_dir}/{artifact_name}", artifact_path=key)
        print(f"logged artifact {key}")
    except:
        print(f"could not log artifact {key}")

    
def log_model_args(model):
    print("logging model arguments ...")
    for arg, value in model.args.items():
        key = get_logging_key(arg, "model.args")
        log_param(key, value)
    print("logged model arguments")


def log_speed(result_type: str, results):
    print(f"logging {result_type} speed ...")
    for speed_metric, value in results.speed.items():
        key = get_logging_key(f"{speed_metric}-speed", result_type)
        log_metric(key, value)
    print(f"logged {result_type} speed")


def verify_val_test(t: str):
    type_options = ["val", "test"]
    if t not in type_options:
        raise ValueError(f"Invalid type {t} for options {type_options}")


def log_val_test_results(result_type: str, results):
    verify_val_test(result_type)

    # Log results for each class
    print(f"logging {result_type} results for each class ...")
    for ind, class_name in results.names.items():
        print(f"logging {result_type} results class {class_name} with index {ind} ...")
        box_results = results.box

        # Fix index if there are no results for class
        if ind not in box_results.ap_class_index:
            print(f"no results")
            continue
        ind = list(box_results.ap_class_index).index(ind)
        
        # print(f"class_result: {box_results.class_result(ind)}")

        log_metric(
            key=get_logging_key("Precision", result_type, class_name),
            value=box_results.p[ind]
        )
        log_metric(
            key=get_logging_key("Recall", result_type, class_name),
            value=box_results.r[ind]
        )
        log_metric(
            key=get_logging_key("F1-score", result_type, class_name),
            value=box_results.f1[ind]
        )
        for all_ap_ind, iou_threshold in enumerate(range(50, 100, 5)):
            log_metric(
                key=get_logging_key(f"AP{iou_threshold}", result_type, class_name),
                value=box_results.all_ap[ind][all_ap_ind]
            )
        log_metric(
            key=get_logging_key("AP", result_type, class_name),
            value=box_results.ap[ind]
        )
        log_metric(
            key=get_logging_key("mAP", result_type, class_name),
            value=box_results.maps[ind]
        )
        print(f"logged {result_type} results class {class_name} with index {ind}")
    print(f"logged {result_type} results for each class")
    
    # Log results for all classes
    print(f"logging {result_type} results for all classes ...")

    # print(f"mean_results: {box_results.mean_results()}")

    log_metric(
        key=get_logging_key("Mean Precision", result_type),
        value=box_results.mp
    )
    log_metric(
        key=get_logging_key("Mean Recall", result_type),
        value=box_results.mr
    )
    log_metric(
        key=get_logging_key("mAP50", result_type),
        value=box_results.map50
    )
    log_metric(
        key=get_logging_key("mAP75", result_type),
        value=box_results.map75
    )
    log_metric(
        key=get_logging_key("mAP", result_type),
        value=box_results.map
    )
    
    # print(f"fitness: {box_results.fitness}")

    for metric, value in results.results_dict.items():
        log_metric(
            key=get_logging_key(metric, result_type),
            value=value
        )
        mlflow.log_metric(get_logging_key(metric, result_type), value)

    print(f"logged {result_type} results for all classes")


def log_val_test_plots(plot_type: str, parent_dir: str):
    verify_val_test(plot_type)

    try:
        print(f"logging {plot_type} plots ...")
        
        log_artifact(
            parent_dir=f"{parent_dir}/{plot_type}",
            artifact_name="confusion_matrix_normalized.png",
            artifact_type=plot_type
        )
    
        print(f"logged {plot_type} plots")
    except:
        print(f"no {plot_type} plots to log (enable plots)")


def log_train_artifacts(parent_dir: str):
    print(f"logging train plots ...")

    parent_dir = f"{parent_dir}/train"
    artifact_type = "train"

    try:
        log_artifact(
            parent_dir=f"{parent_dir}/weights",
            artifact_name="best.pt",
            artifact_type=artifact_type
        )
    except:
        print(f"no weights to log (enable save)")
        
    log_artifact(
        parent_dir=parent_dir,
        artifact_name="results.csv",
        artifact_type=artifact_type
    )
    
    try:
        log_artifact(
            parent_dir=parent_dir,
            artifact_name="results.png",
            artifact_type=artifact_type
        )
        log_artifact(
            parent_dir=parent_dir,
            artifact_name="labels.jpg",
            artifact_type=artifact_type
        )
        log_artifact(
            parent_dir=parent_dir,
            artifact_name="labels_correlogram.jpg",
            artifact_type=artifact_type
        )
    except:
        print(f"no train plots to log (enable plots)")

    print(f"logged train plots")  


@hydra.main(version_base=None, config_path='configs', config_name=HYDRA_OPTUNA_CONFIG_NAME)
def train_val_model(cfg: DictConfig):
   # HydraConfig.get().job.num = 8
    global best_objective
    settings.update(
        {
            "mlflow": False,
          #  "datasets_dir": "/storage/prepared",
            # "model_dir": config.MODELS_PATH,
        }
    )

    # Get clear dirs of testing, training and validation
    base_dir = get_yolo_base_path(cfg)   
    shutil.rmtree(base_dir)
    print(f"removed directory {base_dir}")

    cfg.train_params = drop_project_and_name(drop_None(cfg.train_params))
    cfg.val_test_params = drop_project_and_name(drop_None(cfg.val_test_params))
    
    # Initialize the YOLO model with the given configuration
    model = YOLO(cfg.train_params.model)  # Load the model (from file or pretrained)

    # Train, validate and test the model using the parameters from the config
    train_results = model.train(project=base_dir, name="train", exist_ok=True, **cfg.train_params)
    val_results = model.val(split="val", project=base_dir, name="val", exist_ok=True, **cfg.val_test_params)
   # test_results = model.val(split="test", project=base_dir, name="test", exist_ok=True, **cfg.val_test_params)
    
    # Start an MLflow run
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'mlflow_child' in cfg.keys():
        run_name = cfg.mlflow_child
    print(f"logging to {run_name} ...")
    with mlflow.start_run(run_name=run_name, nested=True):
        log_model_args(model)

        log_train_artifacts(base_dir)

       # log_speed("train", train_results)
       # log_speed("val", val_results)
       # log_speed("test", test_results)
        
        log_val_test_results("val", val_results)
        #log_val_test_results("test", test_results)

        log_val_test_plots("val", base_dir)
        #log_val_test_plots("test", base_dir)

    return val_results.fitness


if __name__ == "__main__":    
    print(f"GPU available: {torch.cuda.is_available()}")



    with open(f"configs/{HYDRA_OPTUNA_CONFIG_NAME}", "r") as file:
        config = yaml.safe_load(file)
    
    mlflow.set_experiment(config['mlflow_project'])
   # mlflow.enable_system_metrics_logging()
    settings.update(
        {
            "mlflow": False,
            "datasets_dir": "/storage/prepared",
            # "model_dir": config.MODELS_PATH,
        }
    )
    # pick up existing parent run
    run_id = None
    if 'mlflow_parent_run_id' in config.keys():
        run_id = config['mlflow_parent_run_id']
        print(f"logging to existing parent run with id {run_id} ...")
    child_run = None
    # if artifacts changed, log new version
    if 'mlflow_child' in config.keys():
        child_run = config['mlflow_child']
        print(f"logging child artifacts with additional prefix {child_run} ...")
    with mlflow.start_run(run_name=config['mlflow_parent'], run_id=run_id, nested=False) as parent_run:
        log_artifact(
            parent_dir=".",
            artifact_name=str(Path(__file__).name),
            artifact_type="hpo",
            prefix2=child_run
        )
        log_artifact(
            parent_dir="configs",
            artifact_name=HYDRA_OPTUNA_CONFIG_NAME,
            artifact_type="hpo",
            prefix2=child_run
        )
        data = Path(config['train_params']['data'])
        log_artifact(
            parent_dir=str(data.parent),
            artifact_name=str(data.name),
            artifact_type="hpo",
            prefix2=child_run
        )

        train_val_model()
        
        log_artifact(
            parent_dir=config['hydra']['sweep']['dir'],
            artifact_name="optimization_results.yaml",
            artifact_type="hpo",
            prefix2=child_run
        )
