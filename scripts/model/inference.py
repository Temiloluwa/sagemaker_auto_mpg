import sys
import os
import logging
import json
import joblib
from sagemaker_containers.beta.framework import worker

# configure logger to standard output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s"))
logger.addHandler(stream_handler)

def model_fn(model_dir):
    """Deserialize fitted model
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def output_fn(prediction, accept):
    """
    Preprocess numpy array to return JSON output
    """
    pred = []
    for i, row in enumerate(prediction.tolist()):
        pred.append({"id": i, "prediction": row})

    return worker.Response(json.dumps(pred))
