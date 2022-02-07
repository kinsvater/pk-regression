import numpy as np
import pandas as pd

from pk_regression.periodic_splines import get_periodic_splines_regressor, transform_dates_to_period_scores


class TestPeriodicSplines:

    def test_get_periodic_splines_regressor(self):
        name = 'hello'
        scores = pd.Series(np.arange(0, 1, 0.1), name=name)
        degree = 3
        regressor = get_periodic_splines_regressor(s=scores, degree=degree)

        assert regressor.shape == (scores.size, degree)
        assert regressor.columns.str.startswith(name + '_spline_dim').all()

    def test_transform_dates_to_period_scores(self):
        dates = pd.date_range(start='2018-01-01', end='2019-12-31', freq='7 D').to_series()
        scores = transform_dates_to_period_scores(dates=dates, period_length=pd.Timedelta('1 Y'))

        assert scores.transform(lambda s: 0 <= s <= 1).all()
        assert scores.size == dates.size
