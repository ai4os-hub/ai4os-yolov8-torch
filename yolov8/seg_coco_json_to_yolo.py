# Ultralytics YOLO 🚀, AGPL-3.0 license
# This script is used to convert a custom dataset with COCO JSON
# annotation format into YOLOv8 annotation format (.txt).
# It is based on the modified version of Ultralytics YOLO,
# tailored for converting   COCO dataset JSON annotations
# to .txt format.


import json
from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)
import yaml
import os
import numpy as np
from tqdm import (
    tqdm,
)
import argparse


def parse_opt():
    parser = argparse.ArgumentParser(
        description="Convert annotations to YOLO format"
    )
    parser.add_argument(
        "-ann",
        "--labels_dir",
        default="annotations",
        help="Path to annotation files",
    )
    args = parser.parse_args()
    return args


def convert_coco(
    use_segments=True,
    use_keypoints=False,
    **args,
):
    """Converts COCO dataset annotations to a format suitable for
    training YOLOv5 models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO
        dataset annotation files.
        use_segments (bool, optional): Whether to include segmentation
        masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint
        annotations in the output.

    Output:
        Generates output files in the specified output directory.
    """

    labels_dir = Path(args["labels_dir"])

    # Create dataset directory
    save_dir = labels_dir

    for p in (
        save_dir / "labels",
        save_dir / "images",
    ):
        p.mkdir(
            parents=True,
            exist_ok=True,
        )  # make dir

    # Import json
    category_names = {}
    for json_file in sorted(
        Path(labels_dir).resolve().glob("*.json")
    ):
        fn = (
            Path(save_dir)
            / "labels"
            / json_file.stem.replace(
                "instances_",
                "",
            )
        )  # folder name
        fn.mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(json_file) as f:
            data = json.load(f)

        # write _darknet.labels, which holds names of all classes
        # (one class per line)
        for category in tqdm(
            data["categories"],
            desc="Categories",
        ):
            category_name = category["name"]
            category_id = category["id"]
            if category_name not in category_names.keys():
                category_names[category_name] = category_id
        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
        # Write labels file
        for (
            img_id,
            anns,
        ) in tqdm(
            imgToAnns.items(),
            desc=f"Annotations {json_file}",
        ):
            img = images[f"{img_id:d}"]
            (
                h,
                w,
                f,
            ) = (
                img["height"],
                img["width"],
                img["file_name"],
            )

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y,
                # width, height]
                box = np.array(
                    ann["bbox"],
                    dtype=np.float64,
                )
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[
                    [
                        0,
                        2,
                    ]
                ] /= w  # normalize x
                box[
                    [
                        1,
                        3,
                    ]
                ] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"]
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                if (
                    use_segments
                    and ann.get("segmentation") is not None
                ):
                    if len(ann["segmentation"]) == 0:
                        segments.append([])
                        continue
                    elif len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (
                            (
                                np.concatenate(
                                    s,
                                    axis=0,
                                )
                                / np.array(
                                    [
                                        w,
                                        h,
                                    ]
                                )
                            )
                            .reshape(-1)
                            .tolist()
                        )
                    else:
                        s = [
                            j for i in ann["segmentation"] for j in i
                        ]  # all segments concatenated
                        s = (
                            (
                                np.array(s).reshape(
                                    -1,
                                    2,
                                )
                                / np.array(
                                    [
                                        w,
                                        h,
                                    ]
                                )
                            )
                            .reshape(-1)
                            .tolist()
                        )
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)
                if use_keypoints and ann.get("keypoints") is not None:
                    keypoints.append(
                        box
                        + (
                            np.array(ann["keypoints"]).reshape(
                                -1,
                                3,
                            )
                            / np.array(
                                [
                                    w,
                                    h,
                                    1,
                                ]
                            )
                        )
                        .reshape(-1)
                        .tolist()
                    )

            # Write
            with open(
                (fn / f).with_suffix(".txt"),
                "a",
            ) as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (
                            *(keypoints[i]),
                        )  # cls, box, keypoints
                    else:
                        line = (
                            *(
                                segments[i]
                                if use_segments
                                and len(segments[i]) > 0
                                else bboxes[i]
                            ),
                        )  # cls, box or segments

                    file.write(
                        ("%g " * len(line)).rstrip() % line + "\n"
                    )

    yaml_file = os.path.join(
        save_dir,
        "label.yaml",
    )
    with open(
        yaml_file,
        "w",
    ) as f:
        yaml.dump(
            category_names,
            f,
        )


def min_index(
    arr1,
    arr2,
):
    """
    Find a pair of indexes with the shortest distance between two
    arrays of 2D points.

    Args:
        arr1 (np.array): A NumPy array of shape (N, 2) representing
        N 2D points.
        arr2 (np.array): A NumPy array of shape (M, 2) representing
         M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with
        the shortest distance in arr1 and arr2 respectively.
    """
    dis = (
        (
            arr1[
                :,
                None,
                :,
            ]
            - arr2[
                None,
                :,
                :,
            ]
        )
        ** 2
    ).sum(-1)
    return np.unravel_index(
        np.argmin(
            dis,
            axis=None,
        ),
        dis.shape,
    )


def merge_multi_segment(
    segments,
):
    """
    Merge multiple segments into one list by connecting the coordinates
     with the minimum distance between each segment.
    This function connects these coordinates with a thin line to
     merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
        Each element is a list of coordinates, like [segmentation1,
         segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments
        represented as NumPy arrays.
    """
    s = []
    segments = [
        np.array(i).reshape(
            -1,
            2,
        )
        for i in segments
    ]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(
        1,
        len(segments),
    ):
        (
            idx1,
            idx2,
        ) = min_index(
            segments[i - 1],
            segments[i],
        )
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for (
                i,
                idx,
            ) in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][
                        ::-1,
                        :,
                    ]

                segments[i] = np.roll(
                    segments[i],
                    -idx[0],
                    axis=0,
                )
                segments[i] = np.concatenate(
                    [
                        segments[i],
                        segments[i][:1],
                    ]
                )
                # deal with the first segment and the last one
                if i in [
                    0,
                    len(idx_list) - 1,
                ]:
                    s.append(segments[i])
                else:
                    idx = [
                        0,
                        idx[1] - idx[0],
                    ]
                    s.append(segments[i][idx[0]: idx[1] + 1])

        else:
            for i in range(
                len(idx_list) - 1,
                -1,
                -1,
            ):
                if i not in [
                    0,
                    len(idx_list) - 1,
                ]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


if __name__ == "__main__":
    args = parse_opt()
    convert_coco(
        use_segments=True,
        use_keypoints=False,
        **vars(args),
    )
