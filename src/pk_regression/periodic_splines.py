import datetime
from typing import Optional

import pandas as pd
from patsy import dmatrix


def get_periodic_splines_regressor(s: pd.Series, degree: int = 4) -> pd.DataFrame:
    """
    Construct a `degree`-dimensional regressor for fitting periodic signals.

    :param s: series of floats between 0 and 1.
    :param degree: number of splines.
    :return: data frame of spline regressors for the input series.
    """
    scores_matrix = dmatrix("cc(val, df={}, constraints='center') - 1".format(degree),
                            data=s.to_frame(name='val'), return_type='dataframe')

    # rename columns for convenience
    base_name = str(s.name) + '_spline_dim_{}' if s.name else 'spline_dim_{}'
    scores_matrix.columns = [base_name.format(idx + 1) for idx in range(scores_matrix.shape[1])]
    return scores_matrix


def transform_dates_to_period_scores(dates: pd.Series, period_length: datetime.timedelta,
                                     period_start: Optional[datetime.datetime] = None) -> pd.Series:
    """

    :param dates:
    :param period_length: period length as a timedelta.
    :param period_start:
    :return: series of period scores between 0 and 1.
    """
    reference_date = period_start if period_start else dates.min()
    return (dates - reference_date) / period_length % 1
