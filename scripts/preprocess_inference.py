import os
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="pl.joblib")
args = parser.parse_args()


def predict_fn(input_object, model):
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, args.model_path))
    return loaded_model
