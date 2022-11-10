# A Practical Introduction to Amazon SageMaker Python SDK

The Amazon SageMaker Python SDK is the recommended library for developing solutions is Sagemaker. The other ways of interacting with Sagemaker are the AWS CLI V2, Boto3 and the AWS web console.
Although the SDK should offer a faster development experience, I discovered a learning curve exists to hit the ground running with it.

This post walks through a simple regression task that showcases the important APIs in the SDK.
I also highlight "gotchas" encountered while developing this solution. The entire codebase is found [here](https://github.com/Temiloluwa/sagemaker_auto_mpg).

## Regression Task: Fuel Consumption Prediction
I selected a regression task I tackled as budding Data scientist ([notebook link](https://github.com/Temiloluwa/ML-database-auto-mpg-prediction/blob/master/solution.ipynb)): to predict fuel consumption in MPG of vehicles ([problem definition](https://archive.ics.uci.edu/ml/datasets/auto+mpg)). The problem is broken down into three stages:

1. A preprocessing stage for feature engineering
2. A model training and evaluation stage
3. An inference stage

Each of these stages produce resuable model artifacts that are stored in S3. 

## Sagemaker Preprocessing and Training

Two things are king in Sagemaker: S3 and Docker containers. S3 is the primary location for storing training data and exporting training artifacts like models. The SDK provides [Preprocessors](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html) and [Estimators](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) as the fundamental interfaces for data preprocessing and model training. These two APIs are simply wrappers for Sagemaker Docker containers. This is what happens under the hood when a preprocessing job is created with a Preprocessor or training job with an Estimator:

1. Data Transfer from S3 into the Sagemaker Docker container
2. Job execution, training or preprocessing, in the container
3. Export of output artifacts (models, preprocessed features) to S3

<br>
<figure>
  <img src="https://sagemaker.readthedocs.io/en/stable/_images/amazon_sagemaker_processing_image1.png" alt="Sagemaker Preprocessing Container">
  <figcaption>This image depicts data transfer into and out of a preprocessing container from S3.</figcaption>
</figure>

### Sagemaker Containers
It is very important to get familar with the enviromental variables and pre-configured path locations in Sagemaker containers. More information is found at the Sagemaker Containers' [Github page](https://github.com/aws/sagemaker-containers). For example Preprocessors receive data from S3 into `/opt/ml/preprocessing/input` while Estimators store training data in `/opt/ml/input/data/train`. Important environmental variables include `SM_MODEL_DIR` for exporting models, `SM_NUM_CPUS`, and `SM_HP_{hyperparameter_name}` 

## Preliminary Steps

Development in Sagemaker begins with initializing a Sagemaker session followed by boilerplate steps of getting the region, execution role and default bucket. I create prefixes to key s3 locations for storing data, preprocessed features and model artifacts. 

``` python

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

This function downloads the raw data, splits it into train, validation and test sets and uploads them to their respective s3 destinations in the default bucket based on the prefixes I defined.

```python

def upload_raw_data_to_s3(sess,
                          raw_train_prefix=raw_train_prefix,
                          raw_val_prefix=raw_val_prefix,
                          raw_test_prefix=raw_test_prefix, 
                          split=0.8):
    """
    Read MPG dataset, peform train test split, then upload to s3
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

## Feature Engineering
All preprocessing scripts are stored in a directory named `scripts/preprocessor`. The preprocessing steps are implemented using Sklearn python library. These are the goals of this step:

1. Preprocess the raw train and validation `.csv` data into features and export them to s3 in `.npy` format
2. Save the trained preprocessing model using `joblib` and export it to s3. This saved model will serve the first stage of the inference pipeline, generating features given input test data.

The Sagemaker Python SDK offers [Sklearn Preprocessors](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#sagemaker.sklearn.processing.SKLearnProcessor) and [PySpark Preprocessors](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.spark.processing.PySparkProcessor). Unfortunately, I discovered it is not possible to use custom scripts or dependencies with both. I had to use the [Framework Preprocessor](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.FrameworkProcessor). 

During instantiation, I configured the preprocessor with the `SKlearn estimator` using the `estimator_cls` parameter. The `.run` method of the preprocessor comes with a `code` parameter for specifying the entry point script and  `source_dir` parameter for defining the directory containing custom scripts.

Data is transferred into and exported out of the preprocessing containiner using [ProcessingInput](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingInput.html) and [ProcessingOutput](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProcessingOutput.html) APIs. Note that unlike Estimators that are executed using a `.fit` method, Preprocessors use a `.run` method.

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

I create a custom preprocessor class that obeys the `.fit` and `.transform` interface by extending `BaseEstimator` and `TransformerMixin`. The preprocessor engineers the `Model Year` Feature into `Age` and makes features `Origin` and `Cylinders` categorical. It is vital that this custom transformer be stored in a separate file and imported by the main preprocessing script. The reason for this will be explained during the training step.

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
### Preprocessor Train Script

The preprocessing script at `preprocessor/train.py` is executed in the Preprocessing container to perform the feature engineering. A [Sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) model is created with the `CustomFeaturePreprocessor` as its first step, followed by a `OneHotEncoder` for categorical columns and `StandardScaler` for numerical columns. The first columns of the pandas dataframes are excluded during transformation because they contain the target. 

After the model is saved using joblib, it is imperative that it be compressed into a `tar` file so that it can successfully be imported during inferencing.

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
```

## Model Training