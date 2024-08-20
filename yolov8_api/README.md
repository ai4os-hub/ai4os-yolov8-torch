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
# Convert COCO JSON and Pascal XML annotations to YOLO format
If you want to convert coco JSON and Pascal XML annotations for detection purposes, you can use the following command:
```bash
$ python -m yolov8.preprocess_ann.py -f {xml or json} -ann {PATH/To/ANNOTATION/FILES} 
```
For converting COCO JSON segmetation format to YOLO format, you can use the following command:

```bash
$ python -m yolov8.seg_coco_json_to_yolo.py  -ann {PATH/To/ANNOTATION/FILES} 
```
