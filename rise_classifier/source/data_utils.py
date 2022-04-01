
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List


class DataUtils:
    """
        utils for processing dataframes
    """

    @staticmethod
    def is_numeric(series: pd.Series) -> bool:
        """
            method which check if series are numeric or no
        """
        return is_numeric_dtype(series)

    @staticmethod
    def is_categorical(series: pd.Series) -> bool:
        """
            method which checks if the series are categorical
        """
        return not DataUtils.is_numeric(series)

    @staticmethod
    def encode(series: pd.Series) -> Tuple[np.ndarray, preprocessing.OrdinalEncoder]:
        """
            apply ordinal encoding to series and return the encoded array with 
            encoder 
        """
        encoder = preprocessing.OrdinalEncoder()
        arr = series.to_numpy().reshape(-1, 1)
        return encoder.fit_transform(arr).astype(int), encoder

    @staticmethod
    def normalize(series: pd.Series) -> Tuple[np.ndarray, preprocessing.MinMaxScaler]:
        """
            apply normalization to series and return scalled array with scaller
        """
        scaler = preprocessing.MinMaxScaler()
        arr = series.to_numpy().reshape(-1, 1)
        return scaler.fit_transform(arr), scaler

    @staticmethod
    def split_into_folds(df: pd.DataFrame, fold_num=5) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
           split the dataset into @folds_num based 
        """
        fold_generator = StratifiedKFold(n_splits=fold_num, shuffle=True)
        class_column = df.columns[-1]

        y = df[class_column]
        del df[class_column]

        x_train, y_train, x_test, y_test = list(), list(), list(), list()
        for train_indices, test_indices in fold_generator.split(df, y):

            x_train.append(df.iloc[train_indices])
            y_train.append(pd.DataFrame(y.iloc[train_indices]))
            x_test.append(df.iloc[test_indices])
            y_test.append(pd.DataFrame(y.iloc[test_indices]))

        return x_train, y_train, x_test, y_test
