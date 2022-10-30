import logging
import sys
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 

# configure logger to standard output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s"))
logger.addHandler(stream_handler)


def get_metrics(train_y_true, train_y_pred, test_y_true, test_y_pred):
    """
    Return train and test metrics
    """
    
    # mae
    t_mae = mean_absolute_error(train_y_true, train_y_pred)
    ts_mae = mean_absolute_error(test_y_true, test_y_pred)
    
    # mse
    t_mse = mean_squared_error(train_y_true, train_y_pred, squared=False)
    ts_mse = mean_squared_error(test_y_true, test_y_pred, squared=False)
    
    # rmse
    t_rmse = mean_squared_error(train_y_true, train_y_pred, squared=True)
    ts_rmse = mean_squared_error(test_y_true, test_y_pred, squared=True)
    
    return t_mae, ts_mae, t_mse, ts_mse, t_rmse, ts_rmse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # location in container: '/opt/ml/model'
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # location in container: '/opt/ml/input/data/train'
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # location in container: '/opt/ml/input/data/test'
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    # model filename
    parser.add_argument('--model-filename', type=str, default="model.joblib")
    
    # Hyperparameters are described here.
    parser.add_argument('--n_estimators', type=int)
    args = parser.parse_args()
    
    logger.debug(f"Number of Estimators: {args.n_estimators}")
    
    # Load numpy features saved in S3 
    # Targets are the first column, features are other columns
    train_data = np.load(os.path.join(args.train, "train_feats.npy"))
    train_feats = train_data[:, 1:]
    train_target = train_data[:, 0]
    
    test_data = np.load(os.path.join(args.test, "test_feats.npy"))
    test_feats = test_data[:, 1:]
    test_target = test_data[:, 0]
    
    # Train random forest model
    model = RandomForestRegressor(max_depth=args.n_estimators, random_state=0)
    model.fit(train_feats, train_target)
    logger.info("Model Trained ")
    
    train_pred = model.predict(train_feats)
    test_pred = model.predict(test_feats)
    
    # Evaluate Model
    train_mae, test_mae, train_mse, test_mse, train_rmse, test_rmse = \
        get_metrics(train_target, train_pred, test_target, test_pred)
    
    logger.info(f"train_mae={train_mae};  test_mae={test_mae};")
    logger.info(f"train_mse={train_mse};  test_mse={test_mse};")
    logger.info(f"train_rmse={train_rmse}; test_rmse={test_rmse};")
    
    # Save the Model
    joblib.dump(model, os.path.join(args.model_dir, args.model_filename))
    