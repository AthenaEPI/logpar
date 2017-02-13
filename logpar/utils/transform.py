''' Functions to transform from and into LogOdds '''
import numpy

def to_logodds(mat, traslate=False, aprox_zero=0.00011, aprox_one=0.99998999):
    ''' Transforms matrix to logodds (applies logit)

        Parameters
        ----------
        mat: array_like
            Matrix to send to transform
        traslate: bool
            If true, the matrix is linealy traslated. This helps to mantain the
            sparsity of the matrix. If the elements of the matrix are going to
            be compared with a metric invariant to traslations, this is a good
            practice.
        aprox_zero: float
            The value 0 is going to be changed to aprox_zero, since logit is
            -inf for zero
        aprox_one: float
            The value 1 is going to be changed to aprox_one, since
            logit(1) = undef '''
    matrix = numpy.array(mat, dtype=mat.dtype)

    matrix[matrix <= aprox_zero] = aprox_zero
    matrix[matrix >= aprox_one] = aprox_one

    complement = numpy.ones_like(matrix, dtype=mat.dtype)
    complement -= matrix

    matrix = numpy.log(matrix)
    complement = numpy.log(complement)

    matrix -= complement

    if traslate:
        matrix -= numpy.log(aprox_zero) - numpy.log(1. - aprox_zero)

    return matrix


def from_logodds(mat, traslate=False, aprox_zero=0.00011, aprox_one=0.99998999):
    ''' Transforms matrix from logodds (applies expit)

        Parameters
        ----------
        mat: array_like
            Matrix to send to transform
        traslate: bool
            If true, the matrix is linealy traslated. This shoulb be used
            if the matrix was traslated during logit.
        aprox_zero: float
            The value 0 is going to be changed to aprox_zero, since logit is
            -inf for zero
        aprox_one: float
            The value 1 is going to be changed to aprox_one, since
            logit(1) = undef
        '''
    matrix = numpy.array(mat, dtype=mat.dtype)

    if traslate:
        matrix += numpy.log(aprox_zero) - numpy.log(1. - aprox_zero)

    ones = numpy.ones_like(matrix, dtype=mat.dtype)
    matrix *= -1
    matrix = numpy.exp(matrix)
    matrix += 1

    matrix = numpy.divide(ones, matrix)

    matrix[matrix <= aprox_zero] = 0.
    matrix[matrix >= aprox_one] = 1.

    return matrix
