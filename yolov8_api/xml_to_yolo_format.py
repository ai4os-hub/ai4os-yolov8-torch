# This script converts the Pascal VOC dataset into the yolo format.
#This script is copied from https://blog.paperspace.com/train-yolov5-custom-data/
import xml.etree.ElementTree as ET


def  extract_info_from_json (coco_annotations):
    info_dict = {}
    info_dict['bboxes'] = []
    
    # Extract relevant information from COCO annotations
    info_dict['filename'] = coco_annotations['images'][0]['file_name']
    info_dict['image_size'] = (coco_annotations['images'][0]['width'], coco_annotations['images'][0]['height'])
    
    for annotation in coco_annotations['annotations']:
        bbox = {}
        bbox['class'] = annotation['category_id']  # Assuming category_id corresponds to class name/index
        bbox['xmin'] = annotation['bbox'][0]
        bbox['ymin'] = annotation['bbox'][1]
        bbox['width'] = annotation['bbox'][2]
        bbox['height'] = annotation['bbox'][3]
        
        info_dict['bboxes'].append(bbox)
    
    return info_dict

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
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
    save_file_name = os.path.join(annotation_path, info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))    

def convert_annotations_to_yolo(format="json", annotation_path="annotations", class_name_to_id_mapping=None):
    annotations = [os.path.join(annotation_path, x) for x in os.listdir(annotation_path) if x.endswith(format)]
    annotations.sort()

    for ann in tqdm(annotations):
        if format == "json":
            with open(ann, 'r') as f:
                coco_annotations = json.load(f)
            info_dict = convert_coco_to_info_dict(coco_annotations)
        elif format == "xml":
            info_dict = extract_info_from_xml(ann)
        convert_to_yolo_format(info_dict, class_name_to_id_mapping, annotation_path)

    
        

if __name__ == "__main__":
    # Get the annotations
    annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]
    annotations.sort()

    # Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolo_format(info_dict)
    annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]    