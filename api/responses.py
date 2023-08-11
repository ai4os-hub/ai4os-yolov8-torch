"""Module for defining custom API response parsers and content types.
This module is used by the API server to convert the output of the requested
method into the desired format. 

The module shows simple but efficient example functions. However, you may
need to modify them for your needs.
"""

import logging
from PIL import Image
import numpy as np
from fpdf import FPDF
import cv2
from io import BytesIO
from reportlab.pdfgen import canvas
from . import config
import tempfile

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


# EXAMPLE of json_response parser function
# = HAVE TO MODIFY FOR YOUR NEEDS =
def json_response(result, **options):
    """Converts the prediction or training results into json return format.

    Arguments:
        result -- Result value from call, expected either dict or str
          (see https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/user/v2-api.html).
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into json dictionary format.
    """
    result= result.tojson()  #this method converts the result into json
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    try:
        if isinstance(result, (dict, list, str)):
            return result
        if isinstance(result, np.ndarray):
            return result.tolist()
        return dict(result)
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to json: %s", err)
        raise RuntimeError("Unsupported response type") from err


# EXAMPLE of pdf_response parser function
# = HAVE TO MODIFY FOR YOUR NEEDS =
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def pdf_response(result, **options):
    """Converts the prediction or training results into pdf return format.

    Arguments:
        result -- Result value from call, expected either dict or str
          (see https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/user/v2-api.html).
        options -- Not used, added for illustration purpose.

    Raises:
        RuntimeError: Unsupported response type.

    Returns:
        Converted result into pdf buffer format.
    """
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    
    try:
        # 1. create BytesIO object
        result=result.plot()

        fig, ax = plt.subplots()

        # Plot the NumPy array as an image
        image = ax.imshow(result, cmap =None)
        ax.axis('off')
        # Create a PDF file
        pdf_filename = 'numpy_array_image.pdf'
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
            plt.close()
        buffer = BytesIO()
        fig.savefig(buffer, format='pdf')
        buffer.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmpfile:
                tmpfile.write(buffer.read())
                tmpfile.flush()
                message = open( tmpfile.name , 'rb')
                return message
      
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to pdf: %s", err)
        raise RuntimeError("Unsupported response type") from err

def png_response(result, **options):
    result= result.plot()
    logger.debug("Response result type: %d", type(result))
    logger.debug("Response result: %d", result)
    logger.debug("Response options: %d", options)
    try:
        success, buffer = cv2.imencode(".png", result)
        if not success:
            return "Error encoding image", 500

         # Create a BytesIO object and write the buffer into it
        image_buffer = BytesIO(buffer)
        
        return image_buffer
    except Exception as err:  # TODO: Fix to specific exception
        logger.warning("Error converting result to png: %s", err)
        raise RuntimeError("Unsupported response type") from err
    

response_parsers = {
    'application/json': json_response,
    'application/pdf': pdf_response,
    'image/png': png_response
}
content_types = list(response_parsers )

 
 
