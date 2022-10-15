import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


original_features = ['cylinders',
                     'displacement',
                     'horsepower',
                     'weight',
                     'acceleration',
                     'model year',
                     'origin']


class CustomFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    This is a custom transformer class that does the following
    
        1. converts model year to age
        2. converts data type of categorical columns
    """
    
    feat = original_features
    new_datatypes = {'cylinders': 'category', 'origin': 'category'}
    
    def fit(self, X, y=None):
        """ Fit function"""
        return self

    def transform(self, X, y=None):
        """ Transform Dataset """
        assert set(list(X.columns)) - set(list(self.feat))\
                    ==  set([]), "input does have the right features"
        
        # conver model year to age
        X["model year"] = 82 - X["model year"]
        
        # change data types of cylinders and origin 
        X = X.astype(self.new_datatypes)
        
        return X
    
    def fit_transform(self, X, y=None):
        """ Fit transform function """
        x = self.fit(X)
        x = self.transform(X)
        return x
    
    
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
    train_features_save_path = os.path.join(CONTAINER_TRAIN_OUTPUT_PATH, "train_feats.npy")
    test_features_save_path = os.path.join(CONTAINER_TEST_OUTPUT_PATH, "test_feats.npy")
    save_numpy(train_features, train_features_save_path)
    save_numpy(test_features, test_features_save_path)
    
    # save preprocessor model
    model_save_path = os.path.join(CONTAINER_OUTPUT_PATH, "pl.joblib")
    joblib.dump(pl, model_save_path)
    