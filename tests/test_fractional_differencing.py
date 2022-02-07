import numpy as np

from pk_regression.fractional_differencing import FractionalDifferencing


class TestFractionalDifferencing:

    def test_get_weights(self):
        tolerance = 1e-3
        fd = FractionalDifferencing(tolerance=tolerance)
        weights = fd._get_weights(degree=1)
        assert np.all(weights == [[-1],
                                  [1]])

        weights = fd._get_weights(degree=2)
        assert np.all(weights == [[1],
                                  [-2],
                                  [1]])

        weights = fd._get_weights(degree=0.7)

        for weight in weights:
            assert abs(weight[0]) >= tolerance
