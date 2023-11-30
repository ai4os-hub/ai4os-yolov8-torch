"""
Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the
requested method into the desired format.
"""
import logging
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from . import config
import tempfile
from PyPDF3 import PdfFileMerger
import os

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def json_response(results, **options):
    """Converts the prediction or training results into json return
    format.

    Arguments:
        result -- Result value from call, expected dict
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into json dictionary format.
    """
    result = []
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    try:
        if options["task_type"] in ["seg", "det"]:
            for element in results[0]:
                result.append(
                    element.tojson()
                )  # This method converts the result into JSON

        elif options["task_type"] == "cls":
            result = {}
            for element in results[0]:
                result["file_name"] = os.path.basename(element.path)
                top5conf = [
                    conf.item() for conf in element.probs.top5conf
                ]
                class_names = [
                    element.names[i] for i in element.probs.top5
                ]
                result["top5_prediction"] = {
                    class_names[i]: top5conf[i]
                    for i in range(len(class_names))
                }
        else:
            raise ValueError("The task type is not supported.")
        if isinstance(result, (dict, list, str)):
            return result
        if isinstance(result, np.ndarray):
            return result.tolist()

    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to json: %s", err)
        raise RuntimeError("Unsupported response type") from err


def pdf_response(results, **options):
    """Converts the prediction or training results into pdf return format.

    Arguments:
        result -- Result value from call, expected dict
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into pdf buffer format.
    """
    logger.debug("Response result type: %d", type(results))
    logger.debug("Response result: %d", results)
    logger.debug("Response options: %d", options)

    try:
        merger = PdfFileMerger()
        for element in results[0]:
            # result.append(element.plot())
            im = Image.fromarray(
                element.plot(
                    labels=options["show_labels"],
                    conf=options["show_conf"],
                    boxes=options["boxes"],
                )
            )
            im = im.convert("RGB")
            buffer = BytesIO()
            buffer.name = "output.pdf"
            im.save(buffer)
            merger.append(buffer)
            buffer.seek(0)
        buffer_out = BytesIO()
        merger.write(buffer_out)
        buffer_out.name = "output.pdf"
        buffer_out.seek(0)
        return buffer_out
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to pdf: %s", err)
        raise RuntimeError("Unsupported response type") from err


def png_response(results, **options):
    logger.debug("Response result type: %d", type(results))
    logger.debug("Response result: %d", results)
    logger.debug("Response options: %d", options)
    try:
        for result in results[0]:
            # this will return a numpy array with the labels
            result = result.plot(
                labels=options["show_labels"],
                conf=options["show_conf"],
                boxes=options["boxes"],
                font_size= 6.0
            )
            success, buffer = cv2.imencode(".png", result)
            if not success:
                return "Error encoding image", 500

            # Create a BytesIO object and write the buffer into it
            image_buffer = BytesIO(buffer)

        return image_buffer
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to png: %s", err)
        raise RuntimeError("Unsupported response type") from err


def create_video_in_buffer(frame_arrays, output_format="mp4"):
    height, width, _ = frame_arrays[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.NamedTemporaryFile(
        suffix="." + output_format, delete=False
    ) as temp_file:
        temp_filename = temp_file.name
        out = cv2.VideoWriter(
            temp_filename, fourcc, 20.0, (width, height)
        )

        for frame in frame_arrays:
            out.write(frame)

        out.release()

    final_filename = "output.mp4"
    os.rename(temp_filename, final_filename)
    # Open the renamed file for reading
    message = open(final_filename, "rb")
    return message


def mp4_response(results, **options):
    """Converts the prediction or training results into
    mp4 return format.

    Arguments:
        result -- Result value from call, expected either dict or str
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into mp4 buffer format.
    """
    # Process MP4 video response
    logger.debug("Response result type: %d", type(results))
    logger.debug("Response result: %d", results)
    logger.debug("Response options: %d", options)
    new_results = []
    for result in results[0]:
        # this will return a numpy array with the labels
        new_results.append(
            result.plot(
                labels=options["show_labels"],
                conf=options["show_conf"],
                boxes=options["boxes"],
            )
        )
    message = create_video_in_buffer(new_results)
    return message


response_parsers = {
    "application/json": json_response,
    "application/pdf": pdf_response,
    "image/png": png_response,
    "video/mp4": mp4_response,
}
content_types = list(response_parsers)
