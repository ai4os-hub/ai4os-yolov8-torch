# This script converts the Pascal VOC dataset into the yolo format.
#This script is copied from https://blog.paperspace.com/train-yolov5-custom-data/ AND https://haobin-tan.netlify.app/ai/computer-vision/object-detection/coco-json-to-yolo-txt/
import xml.etree.ElementTree as ET
import argparse
import json
import yaml
from tqdm import tqdm
import os
def parse_opt():
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO format")
    parser.add_argument( "-f","--format", required=True, choices=["json", "xml"], help="Annotation format (json or xml)")
    parser.add_argument("-ann", "--annotation_path", default="annotations", help="Path to annotation files")
    parser.add_argument("-m", "--class_mapping_file", default="class_mapping.yaml", help="Path to the YAML file containing class mapping")
    args =  parser.parse_args()
    return args

 
import json

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def convert_coco_json_to_yolo_txt(json_file, output_path):


    with open(json_file) as f:
        json_data = json.load(f)
    category_names = []
    # write _darknet.labels, which holds names of all classes (one class per line)
    for category in tqdm(json_data["categories"], desc="Categories"):
        category_name = category["name"]
        category_names.append(category_name)

    # Define the path for the YAML file
    yaml_file = os.path.join(output_path, "COCO_label.yaml")

    # Write category names to the YAML file
    with open(yaml_file, "w") as f:
        yaml.dump(category_names, f)

    for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):

    """
    Extracts information from an XML annotation file and converts it to an info dictionary.

    Parameters:
    - xml_file (str): Path to the XML annotation file.

    Returns:
    - info_dict (dict): A dictionary containing extracted information, including filename, image size, and bounding boxes.
    """
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolo_format(info_dict, class_name_to_id_mapping, annotation_path):
    """
    class_name_to_id_mapping: dict mapping from class name to class id: example
    class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}
    """
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    base_filename, _ = os.path.splitext(info_dict["filename"])
    save_file_name = os.path.join(annotation_path, base_filename+".txt")
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))    

def main(**args):
    annotation_path = args["annotation_path"]
    format = args["format"]
    class_name_to_id_mapping = yaml.safe_load(open(args["class_mapping_file"]))
    annotations = [os.path.join(annotation_path, x) for x in os.listdir(annotation_path) if x.endswith(format)]
    annotations.sort()
    if format == "json":
            convert_coco_json_to_yolo_txt(annotations[0], annotation_path)
            
    elif format == "xml":
        for ann in tqdm(annotations):
                info_dict = extract_info_from_xml(ann)
                convert_to_yolo_format(info_dict, class_name_to_id_mapping, annotation_path)

    
        
def plot_random_bounding_box(annotations_path):
    annotations = [os.path.join(annotations_path, x) for x in os.listdir(annotations_path) if x.endswith(".txt")]
    
    annotation_file = random.choice(annotations)
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x] for x in annotation_list]

    image_file = annotation_file.replace("annotations", "images").replace("txt", "png")
    assert os.path.exists(image_file)

    image = Image.open(image_file)

    plt.imshow(image)
    ax = plt.gca()

    for annotation in annotation_list:
        class_id, center_x, center_y, width, height = annotation
        x = (center_x - width / 2) * image.width
        y = (center_y - height / 2) * image.height
        rect = plt.Rectangle((x, y), width * image.width, height * image.height,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    plt.show()

if __name__ == "__main__":
   # args = parse_opt()
    args={"format": "json",
    "annotation_path": "/srv/yolov8_api/tests/data/processed",
    "class_mapping_file": "/srv/yolov8_api/tests/data/processed/class_map.yaml"}
    main(**args)
