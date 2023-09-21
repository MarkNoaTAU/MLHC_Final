import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class StandardScalerCustomWrapper(BaseEstimator, TransformerMixin):
    """
    A dummy custom wrapper to handle working with AgeGenderImputer and ColumnTransformer,
    without scaling gender.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_names = None

    def fit(self, X, y=None):
        X_ = X.drop(columns='gender') if 'gender' in X.columns else X
        self.scaler.fit(X_, y)
        return self

    def transform(self, X):
        self.columns_names = X.columns
        gender = X['gender']
        X_ = X.drop(columns='gender') if 'gender' in X.columns else X
        X_t = self.scaler.transform(X_)
        return pd.concat([pd.DataFrame(X_t, columns=X_.columns, index=X_.index), gender], axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.columns_names


class SimpleImputerWrapper(BaseEstimator, TransformerMixin):
    """
    A dummy custom wrapper to save the columns names info
  """

    def __init__(self):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.columns_names = None

    def fit(self, X, y=None):
        self.imputer.fit(X, y)
        return self

    def transform(self, X):
        self.columns_names = X.columns
        X_ = self.imputer.transform(X)
        return pd.DataFrame(X_, columns=X.columns, index=X.index)


class AgeGenderImputer(BaseEstimator, TransformerMixin):
    """
      Impute missing value based on the mean value of the age-gender group.
    """
    def __init__(self):
        self.train_mean = None
        self.columns_names = None

    def fit(self, seen_data, y=None):
        # calculate the mean of each of the age-gender group numerical features on the seen-data.
        if not pd.Index(['age', 'gender']).isin(seen_data.columns).all():
            raise ValueError("AgeGenderImputer fit data must contain age and gender information. ")
        seen_data['age_group'] = AgeGenderImputer.split_age_group(seen_data)
        self.train_mean = seen_data.drop('age', axis=1).groupby(['age_group', 'gender']).mean()

        return self

    def transform(self, X, y=None):
        self.columns_names = X.columns
        X['age_group'] = AgeGenderImputer.split_age_group(X)
        X = X.groupby(['age_group', 'gender'], group_keys=False).apply(
            lambda df: df.fillna(value=self.train_mean.loc[df.name])).drop(['age_group'], axis=1)
        return X

    @staticmethod
    def split_age_group(X):
        return pd.cut(X['age'], [18, 30, 40, 50, 60, 70, 80, 200],
                      labels=['18_29', '30_39', '40_41', '50_59', '60_69', '70_79', '80p'], right=False)

    def get_feature_names_out(self, input_features=None):
        return self.columns_names
