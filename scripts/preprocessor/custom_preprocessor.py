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