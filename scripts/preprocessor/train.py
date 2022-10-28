import os
import joblib
import argparse
import numpy as np
import pandas as pd
import tarfile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from custom_preprocessor import original_features, CustomFeaturePreprocessor

def save_numpy(np_array, path):
    """ save np array """
    with open(path, 'wb') as f:
        np.save(f, np_array)

if __name__ == '__main__':
    CONTAINER_TRAIN_INPUT_PATH = "/opt/ml/processing/input/train"
    CONTAINER_TEST_INPUT_PATH = "/opt/ml/processing/input/test"
    CONTAINER_TRAIN_OUTPUT_PATH = "/opt/ml/processing/train"
    CONTAINER_TEST_OUTPUT_PATH = "/opt/ml/processing/test"
    CONTAINER_OUTPUT_PATH = "/opt/ml/processing/output"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-filename', type=str, default='train.csv')
    parser.add_argument('--test-filename', type=str, default='test.csv')
    parser.add_argument('--model-filename', type=str, default='model.tar.gz')
    parser.add_argument('--train-feats-filename', type=str, default='train_feats.npy')
    parser.add_argument('--test-feats-filename', type=str, default='test_feats.npy')
    
    args = parser.parse_args()

    # one hot categorical features
    # apply standard scaler to numerical features
    ct = ColumnTransformer([("categorical-feats", OneHotEncoder(), make_column_selector(dtype_include="category")),
                            ("numerical-feats", StandardScaler(), make_column_selector(dtype_exclude="category"))])

    # apply custom preprocessing
    pl = Pipeline([("custom-preprocessing", CustomFeaturePreprocessor()), ("column-preprocessing", ct)])
    
    train_data_path = os.path.join(CONTAINER_TRAIN_INPUT_PATH, args.train_filename)
    test_data_path = os.path.join(CONTAINER_TEST_INPUT_PATH, args.test_filename)
    
    # preprocess features, first column is target and the rest are features
    train_df = pd.read_csv(train_data_path)
    train_data = train_df.iloc[:, 1:]
    train_target = train_df["mpg"].values.reshape(-1, 1)
    train_features = pl.fit_transform(train_data)
    
    test_df = pd.read_csv(test_data_path)
    test_data = test_df.iloc[:, 1:]
    test_target = test_df["mpg"].values.reshape(-1, 1)
    test_features = pl.transform(test_data)
    
    # save features in output path with the container
    train_features = np.concatenate([train_target, train_features], axis=1)
    test_features = np.concatenate([test_target, test_features], axis=1)
    train_features_save_path = os.path.join(CONTAINER_TRAIN_OUTPUT_PATH, args.train_feats_filename)
    test_features_save_path = os.path.join(CONTAINER_TEST_OUTPUT_PATH, args.test_feats_filename)
    save_numpy(train_features, train_features_save_path)
    save_numpy(test_features, test_features_save_path)
    
    # save preprocessor model
    model_save_path = os.path.join(CONTAINER_OUTPUT_PATH, args.model_filename)
    
    # save model should be a tarfile so it can be loaded in the future
    joblib.dump(pl, "model.joblib")
    with tarfile.open(model_save_path, "w:gz") as tar_handle:
        tar_handle.add("model.joblib")
    