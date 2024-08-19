# This script converts the either Pascal VOC dataset or COCO json
# annotation
# format pascal voc xml and coco jsoninto the yolo format.
# This script is copied from
# https://blog.paperspace.com/train-yolov5-custom-data/
#  AND https://haobin-tan.netlify.app/ai/computer-vision/object-detection/
# coco-json-to-yolo-txt/
# import xml.etree.ElementTree as ET
import defusedxml.ElementTree as ET
import argparse
import json
import yaml
from tqdm import (
    tqdm,
)
import os


def parse_opt():
    parser = argparse.ArgumentParser(
        description="Convert annotations to YOLO format"
    )
    parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=[
            "json",
            "xml",
        ],
        help="Annotation format (json or xml)",
    )
    parser.add_argument(
        "-ann",
        "--annotation_path",
        default="annotations",
        help="Path to annotation files",
    )
    args = parser.parse_args()
    return args


def convert_bbox_coco2yolo(
    img_width,
    img_height,
    bbox,
):
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
    (
        x_tl,
        y_tl,
        w,
        h,
    ) = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [
        x,
        y,
        w,
        h,
    ]


def convert_coco_json_to_yolo_txt(
    json_file,
    output_path,
):
    with open(json_file) as f:
        json_data = json.load(f)
    category_names = []
    # write _darknet.labels, which holds names of all classes
    # (one class per line)
    for category in tqdm(
        json_data["categories"],
        desc="Categories",
    ):
        category_name = category["name"]
        category_names.append(category_name)

    # Define the path for the YAML file
    yaml_file = os.path.join(
        output_path,
        "COCO_label.yaml",
    )

    # Write category names to the YAML file
    with open(
        yaml_file,
        "w",
    ) as f:
        yaml.dump(
            category_names,
            f,
        )

    for image in tqdm(
        json_data["images"],
        desc="Annotation txt for each image",
    ):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [
            anno
            for anno in json_data["annotations"]
            if anno["image_id"] == img_id
        ]
        anno_txt = os.path.join(
            output_path,
            img_name.split(".")[0] + ".txt",
        )
        with open(
            anno_txt,
            "w",
        ) as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                (
                    x,
                    y,
                    w,
                    h,
                ) = convert_bbox_coco2yolo(
                    img_width,
                    img_height,
                    bbox_COCO,
                )
                f.write(
                    f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                )

    print("Converting COCO Json to YOLO txt finished!")


def create_class_mapping(
    xml_file,
    class_name_to_id_mapping,
    current_class_id,
):
    root = ET.parse(xml_file).getroot()
    for elem in root:
        if elem.tag == "object":
            for subelem in elem:
                if subelem.tag == "name":
                    class_name = subelem.text
                    if (
                        class_name
                        not in class_name_to_id_mapping.keys()
                    ):
                        class_name_to_id_mapping[
                            class_name
                        ] = current_class_id
                        current_class_id += 1


# Function to get the data from XML Annotation
def extract_info_from_xml(
    xml_file,
    class_name_to_id_mapping,
    current_class_id,
):
    """
    Extracts information from an XML annotation file and converts
    it to an info dictionary.

    Parameters:
    - xml_file (str): Path to the XML annotation file.
    - class_name_to_id_mapping (dict): A dictionary mapping from
    class name to class id: example
    -current_id (int): The current class id.

    Returns:
    - info_dict (dict): A dictionary containing extracted information,
    including filename, image size, and bounding boxes.
    """
    root = ET.parse(xml_file).getroot()

    # Initialise the info dict
    info_dict = {}
    info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    class_name = subelem.text
                    bbox["class"] = class_name
                    if (
                        class_name
                        not in class_name_to_id_mapping.keys()
                    ):
                        class_name_to_id_mapping[
                            class_name
                        ] = current_class_id
                        current_class_id += 1

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict["bboxes"].append(bbox)

    return (
        info_dict,
        class_name_to_id_mapping,
        current_class_id,
    )


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolo_format(
    info_dict,
    class_name_to_id_mapping,
    annotation_path,
):
    """
    class_name_to_id_mapping: dict mapping from class name to class
    id: example
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
            print(
                "Invalid Class. Must be one from ",
                class_name_to_id_mapping.keys(),
            )

        # Transform the bbox co-ordinates as per the format required
        # by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = b["xmax"] - b["xmin"]
        b_height = b["ymax"] - b["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        (
            image_w,
            image_h,
            image_c,
        ) = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id,
                b_center_x,
                b_center_y,
                b_width,
                b_height,
            )
        )

    # Name of the file which we have to save
    (
        base_filename,
        _,
    ) = os.path.splitext(info_dict["filename"])
    save_file_name = os.path.join(
        annotation_path,
        base_filename + ".txt",
    )

    # Save the annotation to disk
    print(
        "\n".join(print_buffer),
        file=open(
            save_file_name,
            "w",
        ),
    )


def main(
    **args,
):
    annotation_path = args["annotation_path"]
    format = args["format"]
    annotations = [
        os.path.join(
            annotation_path,
            x,
        )
        for x in os.listdir(annotation_path)
        if x.endswith(format)
    ]
    annotations.sort()
    if format == "json":
        convert_coco_json_to_yolo_txt(
            annotations[0],
            annotation_path,
        )

    elif format == "xml":
        class_name_to_id_mapping = {}
        current_class_id = 0

        for ann in tqdm(annotations):
            (
                info_dict,
                class_name_to_id_mapping,
                current_class_id,
            ) = extract_info_from_xml(
                ann,
                class_name_to_id_mapping,
                current_class_id,
            )
            convert_to_yolo_format(
                info_dict,
                class_name_to_id_mapping,
                annotation_path,
            )


if __name__ == "__main__":
    args = parse_opt()
    main(**args)
