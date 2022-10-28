import os
import joblib
import argparse
import pandas as pd
from io import StringIO
from custom_preprocessor import original_features, CustomFeaturePreprocessor

def input_fn(input_data, content_type):
    """ Preprocess input data """
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == len(original_features) + 1:
            df = df.iloc[:, 1:]
            
        if list(df.columns) != original_features:
            df.columns = original_features
            
        return df
    else:
        raise ValueError(f"Unsupported content type: {type(content_type)}")
        

def predict_fn(input_data, model):
    predictions = model.transform(input_data)
    return predictions


def model_fn(model_dir):
    
    model_path = os.path.join(model_dir, "model.joblib")
    loaded_model = joblib.load(model_path)
    return loaded_model


#https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#deploy-a-scikit-learn-model
# by default output content type is application/x-npy
