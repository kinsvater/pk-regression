import numpy as np
import pandas as pd


class FractionalDifferencing:
    def __init__(self, tolerance: float = 1e-2):
        self._tolerance = tolerance

    def get_diff(self, s: pd.Series, degree: float) -> pd.Series:
        """
        Compute fractional differences of `s`.

        :param s: pandas series of floats.
        :param degree: degree of differencing.
        :return: fractional differences of the input series.
        """
        weights = self._get_weights(degree=degree)
        return s.rolling(window=len(weights)).apply(lambda x: np.dot(weights.T, x), raw=False)

    def _get_weights(self, degree: float) -> np.array:
        """
        Fractional differencing is just another weighted average. This method returns these weights.
        In theory the number of non-trivial weights can be infinitely long. Here we always return a finite number
        by dropping weights with magnitude below a given tolerance level.

        :param degree: degree of fractional differencing.
        :return: array of significant weights in shape (-1, 1).
        """
        weights, k = [1.], 1
        while True:
            weight = -weights[-1] / k * (degree - k + 1)
            if abs(weight) < self._tolerance:
                break
            weights.append(weight)
            k += 1
        return np.array(weights[::-1]).reshape(-1, 1)
