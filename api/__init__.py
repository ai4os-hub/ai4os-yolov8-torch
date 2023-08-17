"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import os
import logging
import yaml
import datetime
import tempfile
import shutil
 

from ultralytics import YOLO
from aiohttp.web import HTTPException

import  yolov8_api as aimodel

from  yolov8_api.api import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.MODEL_NAME)
        metadata = {
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get("author-emails"),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            "datasets": utils.ls_dirs(config.DATA_PATH / "processed"),
            "models": utils.ls_dirs(config.MODELS_PATH),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


def warm():
    """Function to run preparation phase before anything else can start.

    Raises:
        RuntimeError: Unexpected errors aim to stop model loading.
    """
    try:  # Call your AI model warm() method
        logger.info("Warming up the model.api...")
        aimodel.warm()
    except Exception as err:
        raise RuntimeError(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict( **args):
    """Performs model prediction from given input data and parameters.

        Arguments:
            model_name -- Model name from registry to use for prediction values.
            input_file -- File with data to perform predictions from model.
            accept -- Response parser type, default is json.
            **args -- Arbitrary keyword arguments from PredArgsSchema.

        Raises:
            HTTPException: Unexpected errors aim to return 50X

        Returns:
            The predicted model values (dict or str) or files.
    """
    try:  # Call your AI model predict() method
        logger.debug("Predict with args: %s", args)

        if args['model'] is  None:
            args['model'] = utils.modify_model_name('yolov8n.pt', args['task_type'])      
        else:     
            args['model'] =  os.path.join(config.MODELS_PATH, args['model'],'weights/best.pt')
       
        with tempfile.TemporaryDirectory() as tmpdir: 
            for f in [args['input']]:
                shutil.copy(f.filename, tmpdir + '/' + f.original_filename )
           
            args['input'] = [os.path.join(tmpdir, t) for t in os.listdir(tmpdir)]
            result = aimodel.predict(**args)
            logger.debug("Predict result: %s", result)
            logger.info("Returning content_type for: %s", args['accept'])
            return responses.response_parsers[args['accept']](result, **args)

    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**args):
    """Performs model training from given input data and parameters.

        Arguments:

        Returns:
            Parsed history/summary of the training process.
    """
    try:  
        logger.info("Training model...")  
        logger.debug("Train with args: %s", args)
        #modified model name for seqmentation and classification tasks
        args['model']= utils.modify_model_name(args['model'], args['task_type'])
        args['data']=os.path.join(config.DATA_PATH, 'raw',  args['data'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        #TODO: should  the project name
        args['project'] = config.MODEL_NAME
        #TODO: point to the model directory without root directory
        args['name'] = os.path.join('models', timestamp)
        model = YOLO(args['model'])
        os.environ['WANDB_DISABLED'] = str(args['disable_wandb'])
        args.pop('disable_wandb', None)
        args.pop('task_type', None)

        model.train(**args)
        return {f'The model was trained successfully and was saved to: {os.path.join(args["project"], args["name"])}'}


    except Exception as err:
        raise HTTPException(reason=err) from err        


if __name__=='__main__':
 
    fields = schemas.TrainArgsSchema().fields

    args={}
    for key,  value in fields.items():
            print(key, value)
            if value.missing:
               args[key]=value.missing
    args['model'] = 'yolov8s.pt'
    args['data']='/srv/yolov8_api/data/raw/PlantDoc.v1-resize-416x416.yolov8/data.yaml'
    
    args['epochs']=5
    args['resume']=False#FIXME

   # train(**args)
    fields = schemas.PredArgsSchema().fields
    from deepaas.model.v2.wrapper import UploadedFile
    args={}
    from  api import  schemas
    for key,  value in fields.items():
            print(key, value)
            if value.missing:
               args[key]=value.missing

    input ='/srv/yolov8_api/data/mixkit-white-cat-lying-among-the-grasses-seen-up-close-22732-large.mp4' 
    args['input']=UploadedFile('input', input, 'application/octet-stream', 'input.mp4')  
    args['model']= None
    args['accept']= 'video/mp4'
    args['task_type'] = 'seg'
    predict(**args)
