# A Practical Introduction to Amazon SageMaker Python SDK

## Introduction
On the 12th of October, 2022, I presented a Knowledge Share to my colleagues at [Machine Learning Reply GmBH](https://www.reply.com/machine-learning-reply/de/) titled, ["Developing Solutions with Sagemaker"](https://www.slideshare.net/TemiReply/mldevelopmentwithsagemakerpptx). Knowledge Sharing is a tradition we observe weekly at [Machine Learning Reply GmBH](https://www.reply.com/machine-learning-reply/de/) that helps us as consultants to develop a broad range of skill sets. On the day, there was little time to go delve into the Sagemaker Python SDK. There was little time to go delve into the Sagemaker Python SDK on the day. With this follow-up blog post, I would like to explore the Estimator API, Model API, Preprocessor API, and Predictor API  using the AWS Sagemaker Python SDK.

## AWS Sagemaker Python SDK

The Amazon SageMaker Python SDK is the recommended library for developing solutions is Sagemaker. The other ways of interacting with Sagemaker are the AWS CLI, Boto3 and the AWS web console.
In theory, the SDK should offer the best developer experience, but I discovered a learning curve exists to hit the ground running with it.

This post walks through a simple regression task that showcases the important APIs in the SDK.
I also highlight "gotchas" encountered while developing this solution. The entire codebase is found [here](https://github.com/Temiloluwa/sagemaker_auto_mpg).

## Regression Task: Fuel Consumption Prediction
I selected a regression task I tackled as a budding Data scientist  ([notebook link](https://github.com/Temiloluwa/ML-database-auto-mpg-prediction/blob/master/solution.ipynb)): to predict fuel consumption of vehicles in MPG ([problem definition](https://archive.ics.uci.edu/ml/datasets/auto+mpg)). I broke down the problem into three stages:

1. A preprocessing stage for feature engineering
2. A model training and evaluation stage
3. A model inferencing  stage

Each of these stages produces reusable model artifacts that are stored in S 

## Sagemaker Preprocessing and Training

Two things are king in Sagemaker: S3 and Docker containers. S3 is the primary location for storing training data and the destination for exporting training artifacts like models. The SDK provides [Preprocessors](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html) and [Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) as the fundamental interfaces for data preprocessing and model training. These two APIs are simply wrappers for Sagemaker Docker containers. This is what happens under the hood when a preprocessing job is created with a Preprocessor or training job with an Estimator:

1. Data is transferred from S3 into the Sagemaker Docker container
2. The Job (training or preprocessing) is executed in the container that runs on the compute instance you have specified for the job
3. Output artifacts (models, preprocessed features) are exported to S3 when the job is concluded

<br>
<figure>
  <img src="https://sagemaker.readthedocs.io/en/stable/_images/amazon_sagemaker_processing_image1.png" alt="Sagemaker Preprocessing Container">
  <figcaption>This image depicts data transfer into and out of a preprocessing container from S3.</figcaption>
</figure>

### Sagemaker Containers
It is very important to get familiar with the environmental variables and pre-configured path locations in Sagemaker containers. More information is found on the Sagemaker Containers' [Github page](https://github.com/aws/sagemaker-containers). For example, Preprocessors, receive data from S3 into `/opt/ml/preprocessing/input` while Estimators store training data in `/opt/ml/input/data/train`. Some environmental variables include `SM_MODEL_DIR` for exporting models, `SM_NUM_CPUS`, and `SM_HP_{hyperparameter_name}`.

## Project Folder Structure
The diagram below shows the project's folder structure. The main script is the python notebook `auto_mpg_prediction.ipynb` whose cells are executed in Sagemaker Studio. Training and preprocessing scripts are located in the `scripts` folder.


``` bash
├── Blog.md
├── LICENSE
├── README.md
├── auto_mpg_prediction.ipynb
└── scripts
    ├── model
    │   ├── inference.py
    │   └── train.py
    └── preprocessor
        ├── custom_preprocessor.py
        ├── inference.py
        └── train.py
```

## Preliminary Steps

Let's start with initializing a Sagemaker session followed by boilerplate steps of getting the region, execution role, and default bucket. I create prefixes to key s3 locations for data storage, and the export of preprocessed features and models. 

``` python
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import boto3
import sagemaker
from sagemaker import get_execution_role
from io import StringIO

# initialize sagemaker session 
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket() 
role = get_execution_role()

# boto3 client
sm_client = boto3.client('sagemaker')

prefix = "auto_mpg"

# raw data path
raw_train_prefix = f"{prefix}/data/bronze/train"
raw_val_prefix = f"{prefix}/data/bronze/val"
raw_test_prefix = f"{prefix}/data/bronze/test"

# preprocessed features path
pp_train_prefix = f"{prefix}/data/gold/train"
pp_val_prefix = f"{prefix}/data/gold/val"
pp_test_prefix = f"{prefix}/data/gold/test"

# preprocessor and ml models
pp_model_prefix = f"{prefix}/models/preprocessor"
ml_model_prefix = f"{prefix}/models/ml"


def get_s3_path(prefix, bucket=bucket):
    """ get full path in s3 """
    return f"s3://{bucket}/{prefix}"

```

## Raw Data Transfer to S3

Next, we have to transfer our raw data to S3. In a production setting, an ETL job sets an S3 bucket as the final data destination. I have implemented a function that downloads the raw data, splits it into the train, validation, and test sets then uploads them all to their respective s3 paths in the default bucket based on pre-defined prefixes.

```python

def upload_raw_data_to_s3(sess,
                          raw_train_prefix=raw_train_prefix,
                          raw_val_prefix=raw_val_prefix,
                          raw_test_prefix=raw_test_prefix, 
                          split=0.8):
    """
    Read MPG dataset, perform train test split, then upload to s3
    """
    # filenames
    train_fn = "train.csv"
    val_fn = "val.csv"
    test_fn = "test.csv"
    
    # download data
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    res = requests.get(data_url)
    file = StringIO(res.text)
    
    # read data
    data = pd.read_csv(file, header = None, delimiter = '\s+', low_memory = False, na_values = "?")
    data_frame = data.drop(columns = 8)
    data_frame = data_frame.fillna(data_frame.mean())
    data_frame = data_frame.rename(index = int, columns = {0: "mpg", 1:"cylinders", 2: "displacement",3: "horsepower", 4: "weight", 5:"acceleration",6:"model year",7:"origin"})
    
    # train - test - split
    train_df = data_frame.sample(frac=split)
    test_df = data_frame.drop(train_df.index)
    
    # take the last 10 rows of test_df as the test data and the 
    val_df = test_df[:-10]
    test_df = test_df[-10:]
    
    assert set(list(train_df.index)).intersection(list(test_df.index)) == set([]), "overlap between train and test"
    
    # save data locally and upload data to s3
    train_df.to_csv(train_fn, index=False, sep=',', encoding='utf-8')
    train_path = sess.upload_data(path=train_fn, bucket=bucket, key_prefix=raw_train_prefix)
    
    val_df.to_csv(val_fn, index=False, sep=',', encoding='utf-8')
    val_path = sess.upload_data(path=val_fn, bucket=bucket, key_prefix=raw_val_prefix)
    
    test_df.to_csv(test_fn, index=False, sep=',', encoding='utf-8')
    test_path = sess.upload_data(path=test_fn, bucket=bucket, key_prefix=raw_test_prefix)
    
    # delete local versions of the data
    os.remove(train_fn)
    os.remove(val_fn)
    os.remove(test_fn)
    
    print("Path to raw train data:", train_path)
    print("Path to raw val data:", val_path)
    print("Path to raw test data:", test_path)
    
    return train_path, val_path, test_path

train_path, val_path, test_path = upload_raw_data_to_s3(sess)

```

## Stage 1: Feature Engineering
The preprocessing steps are implemented using the Sklearn python library. These are the goals of this stage:

1. Preprocess the raw train and validation `.csv` data into features and export them to s3 in [`.npy`](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) format
2. Save the preprocessing model using [`joblib`](https://scikit-learn.org/stable/model_persistence.html) and export it to s3. This saved model will be deployed as the first step of our inference pipeline.  During inferencing, its task will be to generate features (.npy) for raw test data.

The Sagemaker Python SDK offers [Sklearn Preprocessors](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#sagemaker.sklearn.processing.SKLearnProcessor) and [PySpark Preprocessors](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.spark.processing.PySparkProcessor). These are preprocessors that already come with Sklearn and Pyspark pre-installed. Unfortunately, I discovered it is not possible to use custom scripts or dependencies with both. Therefore, I had to use the [Framework Preprocessor](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.FrameworkProcessor). 

To instantiate the Framework Preprocessor with the Sklearn library,  I supplied [`SKlearn estimator`](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html) Class to the `estimator_cls` parameter. The `.run` method of the preprocessor comes with a `code` parameter for specifying the entry point script and  `source_dir` parameter for indicating the directory that contains all custom scripts.

Pay close attention to how data is transferred into and exported out of the preprocessing container using [ProcessingInput](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingInput.html) and [ProcessingOutput](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingOutput.html) APIs. You will see how container (`/opt/ml/*`) and S3 paths for data transfer are specified. Note that unlike Estimators that are executed using a `.fit` method, Preprocessors use a `.run` method.

``` python

from datetime import datetime
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import FrameworkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

current_time = datetime.now().strftime("%d-%b-%Y-%H:%M:%S").replace(":", "-")
TRAIN_FN = 'train.csv'
VAL_FN = 'val.csv'
TRAIN_FEATS_FN = 'train_feats.npy'
VAL_FEATS_FN = 'val_feats.npy'


sklearn_processor = FrameworkProcessor(
    base_job_name=f"auto-mpg-feature-eng-{current_time}",
    framework_version="1.0-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    estimator_cls=SKLearn
)

sklearn_processor.run(
    code="train.py",
    source_dir="scripts/preprocessor/",
    inputs=[
        ProcessingInput(source=get_s3_path(raw_train_prefix), destination="/opt/ml/processing/input/train"),
        ProcessingInput(source=get_s3_path(raw_val_prefix), destination="/opt/ml/processing/input/test")
    ],
    outputs=[
        ProcessingOutput(output_name="train_features", source="/opt/ml/processing/train", destination=get_s3_path(pp_train_prefix)),
        ProcessingOutput(output_name="val_features", source="/opt/ml/processing/test", destination=get_s3_path(pp_val_prefix)),
        ProcessingOutput(output_name="preprocessor_model", source="/opt/ml/processing/output", destination=get_s3_path(pp_model_prefix)),
    ],
    arguments=["--train-filename", TRAIN_FN,
               "--test-filename", VAL_FN,
               "--train-feats-filename", TRAIN_FEATS_FN,
               "--test-feats-filename", VAL_FEATS_FN],
)

```

### Custom Preprocessor

I wanted to implement some custom preprocessing logic so I created a custom preprocessor class and  configured it to follow the popular Sklearn `.fit` and `.transform` interfaces by extending `BaseEstimator` and `TransformerMixin`. The preprocessor engineers the `Model Year` Feature into `Age` and makes features `Origin` and `Cylinders` categorical. It is vital that this custom transformer be stored in a separate file and imported by the main preprocessing script. The reason for this will be explained during the inferencing step.

``` python 
%%writefile scripts/preprocessor/custom_preprocessor.py

from sklearn.base import BaseEstimator, TransformerMixin

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
        
        # convert model year to age
        X["model year"] = 82 - X["model year"]
        
        # change data types of cylinders and origin 
        X = X.astype(self.new_datatypes)
        
        return X
    
    def fit_transform(self, X, y=None):
        """ Fit transform function """
        x = self.fit(X)
        x = self.transform(X)
        return x
```
### Preprocessing Job

The preprocessing script at `scripts/preprocessor/train.py` is executed in the preprocessing container to perform the feature engineering. I create a [Sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) model is using my `CustomFeaturePreprocessor` class as its first step, followed by a `OneHotEncoder` for transforming categorical columns and finally `StandardScaler` for numerical columns. A Sklearn pipeline is an easy way to chain multiple Sklearn transformers together.

As a good ML Engineer, you should avoid [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)) during feature engineering. Since the same transformer is applied to the train and validation, I excluded the first columns of the pandas dataframes which is the target. I also fitted the model on only the train set.

After the model is saved using joblib, it is imperative that it be compressed into a [`tar`](https://docs.python.org/3/library/tarfile.html) file. Sagemaker models are archived as tarfiles else,  an error will be thrown when importing the model during inferencing.

``` python
%%writefile scripts/preprocessor/train.py

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
    CONTAINER_VAL_INPUT_PATH = "/opt/ml/processing/input/test"
    CONTAINER_TRAIN_OUTPUT_PATH = "/opt/ml/processing/train"
    CONTAINER_VAL_OUTPUT_PATH = "/opt/ml/processing/test"
    CONTAINER_OUTPUT_PATH = "/opt/ml/processing/output"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-filename', type=str, default='train.csv')
    parser.add_argument('--val-filename', type=str, default='val.csv')
    parser.add_argument('--model-filename', type=str, default='model.tar.gz')
    parser.add_argument('--train-feats-filename', type=str, default='train_feats.npy')
    parser.add_argument('--val-feats-filename', type=str, default='val_feats.npy')
    
    args = parser.parse_args()

    # one hot categorical features
    # apply standard scaler to numerical features
    ct = ColumnTransformer([("categorical-feats", OneHotEncoder(), make_column_selector(dtype_include="category")),
                            ("numerical-feats", StandardScaler(), make_column_selector(dtype_exclude="category"))])

    # apply custom preprocessing
    pl = Pipeline([("custom-preprocessing", CustomFeaturePreprocessor()), ("column-preprocessing", ct)])
    
    train_data_path = os.path.join(CONTAINER_TRAIN_INPUT_PATH, args.train_filename)
    val_data_path = os.path.join(CONTAINER_VAL_INPUT_PATH, args.val_filename)
    
    # preprocess features, first column is target and the rest are features
    train_df = pd.read_csv(train_data_path)
    train_data = train_df.iloc[:, 1:]
    train_target = train_df["mpg"].values.reshape(-1, 1)
    train_features = pl.fit_transform(train_data)
    
    val_df = pd.read_csv(val_data_path)
    val_data = val_df.iloc[:, 1:]
    val_target = val_df["mpg"].values.reshape(-1, 1)
    val_features = pl.transform(val_data)
    
    # save features in output path with the container
    train_features = np.concatenate([train_target, train_features], axis=1)
    val_features = np.concatenate([val_target, val_features], axis=1)
    train_features_save_path = os.path.join(CONTAINER_TRAIN_OUTPUT_PATH, args.train_feats_filename)
    val_features_save_path = os.path.join(CONTAINER_VAL_OUTPUT_PATH, args.val_feats_filename)
    save_numpy(train_features, train_features_save_path)
    save_numpy(val_features, val_features_save_path)
    
    # save preprocessor model
    model_save_path = os.path.join(CONTAINER_OUTPUT_PATH, args.model_filename)
    
    # save model should be a tarfile so it can be loaded in the future
    joblib.dump(pl, "model.joblib")
    with tarfile.open(model_save_path, "w:gz") as tar_handle:
        tar_handle.add("model.joblib")
```

## Stage 2: Model Training and Evaluation
The different python libary `smexperiments` is used for Experment tracking in Sagemaker. A Trial in Sagemaker is synonymous to an MLFlow run. A trial could consist of multiple ML workflow stages, depending on the complexity of the solution.  For example, model training and evaluation or just a single hyperparameter optimization step could be considered a trial. What's important is that metrics are logged for each trial run to enable the comparison of different trials. 

I created a trial for just the training step and attributed it to the created experiment using the `experiment_name` parameter in the `Trial.create` call. 

``` python

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

current_time = datetime.now().strftime("%d-%b-%Y-%H:%M:%S").replace(":", "-")
experiment_name = "auto-mg-experiment"
# create a new experiment a load one if it exists
try:
    auto_experiment = Experiment.load(experiment_name=experiment_name)
    print(f'experiment {experiment_name} was loaded')
except Exception as ex:
    if "ResourceNotFound" in str(ex):
        auto_experiment = Experiment.create(experiment_name = experiment_name,
                                            description = "Regression on Auto MPG dataset",
                                            tags = [{'Key': 'Name', 'Value': f"auto-mg-experiment-{current_time}"},
                                                    {'Key': 'MLEngineer', 'Value': f"Temiloluwa Adeoti"},
                                                   ])
        print(f'experiment {experiment_name} was created')

from sagemaker.sklearn.estimator import SKLearn

current_time = datetime.now().strftime("%d-%b-%Y-%H:%M:%S").replace(":", "-")
n_estimators = 10
trail_name = f"auto-mg-{n_estimators}-estimators"
# create a trial for the training job
training_job_trial = Trial.create(trial_name = f"{trail_name}-{current_time}",
                              experiment_name = auto_experiment.experiment_name,
                              sagemaker_boto_client=sm_client,
                              tags = [{'Key': 'Name', 'Value': f"auto-mg-{current_time}"},
                                       {'Key': 'MLEngineer', 'Value': f"Temiloluwa Adeoti"}])

# configure the estimator
model = SKLearn(
    entry_point="train.py",
    source_dir="./scripts/model",
    framework_version="1.0-1", 
    instance_type="ml.m5.xlarge", 
    role=role,
    output_path = get_s3_path(ml_model_prefix), # model output path
    hyperparameters = {
        "n_estimators": n_estimators
    },
    metric_definitions=[
            {"Name": "train:mae", "Regex": "train_mae=(.*?);"},
            {"Name": "test:mae", "Regex": "test_mae=(.*?);"},
            {"Name": "train:mse", "Regex": "train_mse=(.*?);"},
            {"Name": "test:mse", "Regex": "test_mse=(.*?);"},
            {"Name": "train:rmse", "Regex": "train_rmse=(.*?);"},
            {"Name": "test:rmse", "Regex": "test_rmse=(.*?);"},
        ],
    enable_sagemaker_metrics=True
)

# fit the estimator
model.fit(job_name=f"auto-mpg-{current_time}",
          inputs = {"train": get_s3_path(pp_train_prefix), 
                    "test": get_s3_path(pp_val_prefix)
                   }, 
          experiment_config={
            "TrialName": training_job_trial.trial_name,
            "TrialComponentDisplayName": f"Training-auto-mg-run-{current_time}",
          },
          logs="All")

```

I reckon logging of metrics to be a bit involved in Sagemaker in comparision to other frameworks. This is how custom training metrics are captured:

1. Create a logger that streams to standard output `(logging.StreamHandler(sys.stdout))`. The streamed logs are automatically captured by AWS Cloudwatch.
2. Log metrics based on your predetermined format e.g metric-name=metric-value.
3. When creating the estimator that runs the training script, a regex pattern that matches your metric logging format must be given to the `metric_definition` parameter.

The training job is executed by running the `scripts/model/train.py` file within in an Sklearn Estimator container on a compute instance (ml.m5.xlarge in this case). Pay attention to how inputs are supplied to estimators using the `inputs` parameter and how the training job is  assigned to the created trial using the `experiment_config` parameter. 

The script trains a `RandomForestRegressor` on the `.npy` preprocessed train features and the model is evaluated on validation features. I will explain in what follows why I did not save the model as a tarfile.

``` python
%%writefile scripts/model/train.py
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


def get_metrics(train_y_true, train_y_pred, val_y_true, val_y_pred):
    """
    Return train and val metrics
    """
    
    # mae
    t_mae = mean_absolute_error(train_y_true, train_y_pred)
    ts_mae = mean_absolute_error(val_y_true, val_y_pred)
    
    # mse
    t_mse = mean_squared_error(train_y_true, train_y_pred, squared=False)
    ts_mse = mean_squared_error(val_y_true, val_y_pred, squared=False)
    
    # rmse
    t_rmse = mean_squared_error(train_y_true, train_y_pred, squared=True)
    ts_rmse = mean_squared_error(val_y_true, val_y_pred, squared=True)
    
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
    
    val_data = np.load(os.path.join(args.test, "val_feats.npy"))
    val_feats = val_data[:, 1:]
    val_target = val_data[:, 0]
    
    # Train random forest model
    model = RandomForestRegressor(max_depth=args.n_estimators, random_state=0)
    model.fit(train_feats, train_target)
    logger.info("Model Trained ")
    
    train_pred = model.predict(train_feats)
    val_pred = model.predict(val_feats)
    
    # Evaluate Model
    train_mae, val_mae, train_mse, val_mse, train_rmse, val_rmse = \
        get_metrics(train_target, train_pred, val_target, val_pred)
    
    logger.info(f"train_mae={train_mae};  val_mae={val_mae};")
    logger.info(f"train_mse={train_mse};  val_mse={val_mse};")
    logger.info(f"train_rmse={train_rmse}; val_rmse={val_rmse};")
    
    # Save the Model
    joblib.dump(model, os.path.join(args.model_dir, args.model_filename))
        
```

## Stage 3: Model Inferencing

Sagemaker provides the [Model](https://sagemaker.readthedocs.io/en/stable/api/inference/model.html) API for model deployement to an endpoint and the [Predictor](https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html) API for making predictions with the endpoint. Since we have two models, the preprocessor and regressor models, we require a Sagemaker pipeline model to chain both and make a deployment. 

I selected the `SKLearnModel` (Model with Sklearn dependencies pre-installed) for the preprocessor. To prepare the Model, I supplied as arguments paths to the saved tar model in S3, the inference script which is its entry point and the `custom_preprocessor.py`.

Repeating the `CustomFeaturePreprocessor` in both the preprocessor model training (`scripts/preprocessor/train.py`) and inference (`scripts/preprocessor/inference.py`) scripts did not work. I needed to import the class from a seperate file (`scripts/preprocessor/custom_preprocessor.py.py`) for inferencing.  This was the error I encountered:
<code>Can't get attribute 'CustomFeaturePreprocessor' on <module '__main__' from '/miniconda3/bin/gunicorn'> </code>. It is a common problem faced during model deployment. You can read more on this type of problem at this [Stackoverflow post](https://stackoverflow.com/questions/49621169/joblib-load-main-attributeerror)

I did not save the regressor as a tarfile because I chose not to import the regressor model from S3. Instead, I created a Sagemaker Model directly from the trained Estimator with the `.create_model` method (another way to create Models).

When the two models are provided in a list in the pipeline model definition as shown below, Sagemaker automatically serves the output of the preprocessor model as input to the regressor model. I deployed the model `ml.c4.xlarge` instance and used a [`CSVSerializer`](https://sagemaker.readthedocs.io/en/stable/api/inference/serializers.html) for input requests. 


``` python
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.pipeline import PipelineModel
from sagemaker.serializers import CSVSerializer
from datetime import datetime

current_time = datetime.now().strftime("%d-%b-%Y-%H:%M:%S").replace(":", "-")
model_name = f"inference-pipeline-{current_time}"
endpoint_name = f"inference-pipeline-{current_time}"
pp_model_path = get_s3_path(pp_model_prefix) + "/model.tar.gz"

print("preprocessor model path ", pp_model_path)

# preprocessor
sklearn_processor_model = SKLearnModel(
                             model_data=pp_model_path,
                             role=role,
                             entry_point="scripts/preprocessor/inference.py",
                             dependencies=["scripts/preprocessor/custom_preprocessor.py"],
                             framework_version="1.0-1",
                             sagemaker_session=sess)

# regression model
reg_model = model.create_model(entry_point="inference.py",
                               source_dir="./scripts/model")

# create a pipeline model with the two models    
inference_pipeline = PipelineModel(
    name=model_name, role=role, models=[sklearn_processor_model, reg_model],
    sagemaker_session=sess
)

# deploy model
predictor = inference_pipeline.deploy(initial_instance_count=1, 
                                      instance_type="ml.c4.xlarge", 
                                      endpoint_name=endpoint_name,
                                      serializer=CSVSerializer() # to ensure input is csv
                                     )
```

### Sagemaker Inference Script Structure

In Sagemaker, the model server requires four functions to successfully deploy a model.

1. The `model_fn`  loads the model file e.g `.joblib`. 
2. The `input_fn` for parses the input request to model. Data deserialization and transformation can occur to prepare the request for the model.
3. The `predict_fn` makes the actual prediction with the model e.g calls `model.predict`.
4. The `output_fn` processes the response and delivers it in the desired format to the caller.

Our Scikit-learn model server already has default implementations of these functions which can be overriden in our inference script.

#### Preprocessor Inference Script

In my inference script for the preprocessor, I imported the custom dependencies to avoid the error earlier mentioned. The `model_fn` loads the `.joblib` model, while the `input_fn` ensures the request is in `text` or `csv` format and transforms the data into a `Pandas dataframe`. By default, the `predict_fn` should make a `.predict` call on the model, but since the preprocessor is a transform, the `.transform` method is used. 

``` python
%%writefile scripts/preprocessor/inference.py
import os
import joblib
import argparse
import pandas as pd
from io import StringIO
# import the custom dependencies
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
    """ Call the transform method instead of the default predict method"""
    predictions = model.transform(input_data)
    return predictions


def model_fn(model_dir):
    """Load the model"""
    model_path = os.path.join(model_dir, "model.joblib")
    loaded_model = joblib.load(model_path)
    return loaded_model
```

#### Regressor Inference Script

The regressor uses a much simpler inference script. Here the `output_fn` is implemented to provide a JSON response to the input request.

``` python
%%writefile scripts/model/inference.py
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
```

### Making Predictions

All the major work is done and making predictions is a very simple process with the [`Predictor API`](https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html). 

In the code snippet below, I download raw test set from S3 and store each line of the csv,expect the header, as a string in the list called `test_data`. After instantiating the `Predictor` with the `endpoint name` and `sagemaker session`, I make predictions by calling the  `.predict` method on the `predictor` instance.
``` python
from pprint import pprint

# download raw test data and read text files
sess.download_data(path=".", bucket=bucket, key_prefix=raw_test_prefix)
with open("test.csv", "r") as f:
    test_data = f.readlines()[1:]

# make predictions with the deployed endpoint
from sagemaker.predictor import Predictor
from sagemaker.deserializers import JSONLinesDeserializer

predictor = Predictor(
    endpoint_name=endpoint_name, sagemaker_session=sess, serializer=CSVSerializer(), deserializer=JSONLinesDeserializer()
)

num_of_samples = 1
response = predictor.predict(test_data[:num_of_samples])
```

## Conclusion
In this post, we have explored the Estimator API, Model API, Preprocessor API, and Predictor API using the AWS Sagemaker Python SDK by training and deploying a regression model. These are the fundamental APIs that are required for developing solutions in Sagemaker.

Sagemaker comes with over 20 features that cover most of the ML life cycle from data annotation and preprocessing to model deployment. A full feature list is found here. Some of these features like we saw with Sagemaker Experiments have separate python libraries. This could make Sagemaker challenging to learn. From my experience, I have found it helpful to study Sagemaker examples when working with unfamiliar Sagemaker features. 

Since most Sagemaker jobs run within a docker container, it's vital to have logging activated to debug errors. Logs streamed to standard output and default Sagemaker metrics are captured by Cloudwatch. I guarantee that this would save you a lot of pain when working with Sagemaker.