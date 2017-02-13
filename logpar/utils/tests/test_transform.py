''' Tests utils.transform '''

import numpy
from .. import transform

def test_logit_expit_transform():
    ''' Tests that the functions applied are inversibles '''

    matrix = numpy.random.random((1000, 1000))

    matrix_copy = matrix.copy()

    log_matrix = transform.to_logodds(matrix_copy, aprox_zero=0, aprox_one=1)
    exp_matrix = transform.from_logodds(log_matrix, aprox_zero=0, aprox_one=1)

    numpy.testing.assert_almost_equal(matrix, exp_matrix)

    matrix_copy = matrix.copy()
    mat_min, mat_max = matrix.min(), matrix.max()
    aprox_zero = mat_min/2.
    aprox_one = (1-mat_max)/2. + mat_max

    log_matrix = transform.to_logodds(matrix_copy, traslate=True,
                                      aprox_zero=aprox_zero,
                                      aprox_one=aprox_one)
    exp_matrix = transform.from_logodds(log_matrix, traslate=True,
                                        aprox_zero=aprox_zero,
                                        aprox_one=aprox_one)

    numpy.testing.assert_almost_equal(matrix, exp_matrix)


def test_logit_expit_transform_32bit():
    ''' Tests that the functions applied are inversibles in 32bit'''

    matrix = numpy.random.random((1000, 1000)).astype(dtype=numpy.float32)

    matrix_copy = matrix.copy()

    log_matrix = transform.to_logodds(matrix_copy, aprox_zero=0, aprox_one=1)
    exp_matrix = transform.from_logodds(log_matrix, aprox_zero=0, aprox_one=1)

    numpy.testing.assert_almost_equal(matrix, exp_matrix)

    matrix_copy = matrix.copy()
    mat_min, mat_max = matrix.min(), matrix.max()
    aprox_zero = mat_min/2.
    aprox_one = (1-mat_max)/2. + mat_max

    log_matrix = transform.to_logodds(matrix_copy, traslate=True,
                                      aprox_zero=aprox_zero,
                                      aprox_one=aprox_one)
    exp_matrix = transform.from_logodds(log_matrix, traslate=True,
                                        aprox_zero=aprox_zero,
                                        aprox_one=aprox_one)

    numpy.testing.assert_almost_equal(matrix, exp_matrix, 6)
