import argparse
import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Hyperparameters are described here.
    parser.add_argument('--n_estimators', type=int, default=5)
    
    args = parser.parse_args()
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    

    def input_fn(input_data, content_type):
        """Parse input data payload
        """
        target, df = prepare_data(input_data)

        return df       
        
    def model_fn(model_dir):
        """Deserialize fitted model
        """
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        return model