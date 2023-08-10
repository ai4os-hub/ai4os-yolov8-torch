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
 
#from yolov8_api.yolov8_api import trainer
from ultralytics import YOLO
from yolov8_api.yolov8_api import trainer



from aiohttp.web import HTTPException

import yolov8_api as aimodel

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
def predict(model_name, input_file, accept='application/json', **options):
    """Performs model prediction from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for prediction values.
        input_file -- File with data to perform predictions from model.
        accept -- Response parser type, default is json.
        **options -- Arbitrary keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """
    try:  # Call your AI model predict() method
        logger.info("Using model %s for predictions", model_name)
        logger.debug("Loading data from input_file: %s", input_file.filename)
        logger.debug("Predict with options: %s", options)
        result = aimodel.predict(input_file.filename, model_name, **options)
        logger.debug("Predict result: %s", result)
        logger.info("Returning content_type for: %s", accept)
        return responses.content_types[accept](result, **options)
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**args):
    """Performs model training from given input data and parameters.

        Arguments:

        Returns:
            Parsed history/summary of the training process.
    """
    try:  # Call your AI model train() method #    
    
        
        logger.info("Training model...")  
        logger.debug("Train with options: %s", args)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      #  args['save_dir'] = os.path.join(config.MODELS_PATH, timestamp)
        args['mode']= 'train'
        args['name']=   None
       # os.makedirs(args['save_dir'], exist_ok=True)
        #filename = 'config.yaml'
        #cfg_path = os.path.join(args['save_dir'], filename)
        # Write the args dictionary to the config file
        #with open(cfg_path, 'w') as yaml_file:
        #    yaml.dump(args, yaml_file, default_flow_style=False)
       # args=utils.DotDict(args)    
        #3CFG=utils.load_config(cfg_path)
       # trainer.train(cfg=args)   

  
        # Define the command and arguments as a list
        args['name'] = os.path.join('models', timestamp)#Fixme:should be from root dir
        args['project'] = 'yolov8_api' #FIXME:should be root dir
        model = YOLO(args['model'])
        os.environ['WANDB_DISABLED'] = 'true'
        results = model.train(**args)
           

        

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

    train(**args)