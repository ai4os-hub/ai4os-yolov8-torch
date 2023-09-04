# AI code
In this directory, there are several scripts related to development of the AI model that require for functionality of the deepaas API. 

# Directory structure
```bash
project-directory/
│                 # Other cached files
│
├── README.md                     # Project README file
│
├── __init__.py                   # Initialization script with script for prediction
│
├── config.py                     # Configuration file (initial commit)
│
├── preprocess_ann.py             # Annotation preprocessing script to convert COCO JSON and Pascal XML annotations into YOLO .txt files.
│
├── seg_coco_json_to_yolo.py      # Script for converting COCO JSON to YOLO format 
│
└── utils.py                      # Utility functions script (update sty)
```

```bash
$ python -m yolov8_api.dataset.make_dataset {your_arguments}
$ python -m yolov8_api.models..make_model {your_arguments}
$ python -m yolov8_api.visualization.compare_models {your_arguments}
```
