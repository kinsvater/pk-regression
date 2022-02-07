import numpy as np
import pandas as pd

from pk_regression.conditional_mode import ConditionalMode


class TestConditionalMode:

    def test_fit_and_predict(self):
        df_train = pd.DataFrame({'y1': ['a', 'a', 'b', 'c', 'a'],
                                 'y2': ['c', 'c', 'd', 'e', 'c'],
                                 'x1': ['v', 'v', 'v', 'w', 'v'],
                                 'x2': ['s', 's', 's', 't', np.nan]})

        regressor_columns = ['x1', 'x2']
        target_columns = ['y1', 'y2']
        model = ConditionalMode(target_columns=target_columns, regressor_columns=regressor_columns)

        model.fit(data_train=df_train)
        assert model._conditional_modes.shape[1] == len(regressor_columns) + 1
        for value in model._conditional_modes['_target']:
            assert len(value) == len(target_columns)

        df_predict = pd.DataFrame({'x1': ['v', 'w', 'w'], 'x2': ['s', 's', 't']})
        pred = model.predict(data_predict=df_predict)
        assert pred[0] == ('a', 'c')
        assert np.isnan(pred[1])
        assert pred[2] == ('c', 'e')

        pred_expanded = model.predict(df_predict, expand=True)
        assert pred_expanded.shape == (3, 2)
