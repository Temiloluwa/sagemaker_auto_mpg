# A Practical Introduction to Amazon SageMaker Python SDK

As a Machine Learning Engineer working with Sagemaker, I use the Amazon SageMaker Python SDK mostly in conjuction with the AWS CLI V2 and Boto3.
Although the SDK should offer a faster development experience over the others, I found a learning curve exists to hit the ground running with it.
This post walks through a simple regression task in a bid to showcase some of the common APIs in the SDK.
I also highlight "gotchas" I encountered while developing this solution.

## Regression Task
I chose a regression task I tackled some years back as budding Data scientist ([notebook link](https://github.com/Temiloluwa/ML-database-auto-mpg-prediction/blob/master/solution.ipynb)).
Given features of some vehicles, the task is to predict fuel consumption in MPG  ([problem definition](https://archive.ics.uci.edu/ml/datasets/auto+mpg)). Let's use Sagemaker to develop this solution with the following stages:

1. A preprocessing stage for feature engineering
2. A model training and evaluation stage
3. An inference stage

Each of these stages will produce resuable model artifacts that will be stored stored in S3. 

## First Steps

Development in Sagemaker begins with creating a Sagemaker session and it is followed by boilerplate steps: getting the region, execution role and default bucket. I create prefixes to key s3 locations for storing data, preprocessed features and model artifacts. 

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

## Getting the Raw Data

Two things are king in Sagemaker: S3 and containers (more on this later). The goal is to get the raw data to s3 to serve our preprocessing and training containers. This function downloads the raw data, splits it into train, validation and test sets and uploads them to their s3 destinations.

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
