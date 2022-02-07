"""Conditional mode regression.

"""
from typing import Iterable

import numpy as np
import pandas as pd


class ConditionalMode:
    """
    Compute empirical mode of `target_columns` conditional on `regressor_columns`. These estimates
    can be used to predict unknown `target_columns` values if corresponding `regressor_columns` values
    of a record are known.

    Target and regressor can be multidimensional.
    """

    def __init__(self, target_columns: Iterable[str], regressor_columns: Iterable[str]):
        self._target_columns = list(target_columns)
        self._regressor_columns = list(regressor_columns)
        self._conditional_modes = None

    def fit(self, data_train: pd.DataFrame) -> None:
        """
        Computes empirical modes of `target_columns` conditional on `predictor_columns` on a training set.
        """
        df = data_train[self._regressor_columns + self._target_columns].copy().dropna()
        if len(self._target_columns) > 1:
            df['_target'] = df[self._target_columns].apply(tuple, axis=1)
        else:
            df['_target'] = df[self._target_columns[0]]
        self._conditional_modes = (df.groupby(self._regressor_columns)['_target']
                                   .agg(lambda x: x.mode().head(1)).reset_index()
                                   .dropna())

    def predict(self, data_predict, expand=False) -> np.array:
        """
        Returns predictions of `target_columns` conditional on the input's `regressor_columns`.

        If target columns are more than one dimension and if expand is True, then a 2-dim array is returned instead
        of a 1-dim array of tuples.
        """
        if len(self._target_columns) == 1 or not expand:
            return data_predict.merge(self._conditional_modes, on=self._regressor_columns, how='left')['_target'].values
        else:
            conditional_modes = self._conditional_modes.copy()
            conditional_modes[self._target_columns] = pd.DataFrame([*conditional_modes['_target']],
                                                                   index=conditional_modes.index)
            return data_predict.merge(conditional_modes.drop('_target', axis=1),
                                      on=self._regressor_columns, how='left')[self._target_columns].values
