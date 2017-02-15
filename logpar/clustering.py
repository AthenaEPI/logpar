''' Prepares features and constraints to be clustered by the cython module  '''

import numpy
import scipy.spatial.distance as distance

from skimage.graph.heap import FastUpdateBinaryHeap as heap
from .constrained_ahc import agglomerative, cond2mat_index


def check_input(method, constraints, min_size):
    ''' Basic input checking '''
    if method not in ['centroid', 'ward']:
        raise ValueError("Method MUST be either 'centroid' or 'ward'")

    if min_size < 0:
        raise ValueError("min_size MUST be greater or equal to zero")


def clustering(features, method='ward', constraints=None, min_size=0,
               copy=True, verbose=False):
    ''' Clusters features using the selected method. If indicated, it
        constraints the clustering between neighbors UNTIL clusters reach a
        minimum size. After, the clustering continues WITHOUT constraints.
        
        Parameters
        ----------
        features: array_like
            Matrix where each row is a vector of features
        method: string
            Clustering method to use (ward, centroid)
        constraints: array_like
            Binary matrix of #features x #features, each entry
            represents if two features are neighbors
        min_size: int
            Minimum amount of features in each resulting cluster
        copy: bool
            If FALSE then the features are clustered IN PLACE,
            modifying the INPUT features matrix.
        verbose: bool
            if TRUE displays debbuging information
        
        Returns
        -------
        dendrogram
            Returns a dendrogram represeting the clustering of features
        '''
    check_input(method, constraints, min_size)

    n = features.shape[0]

    if constraints is None:
        constraints = numpy.ones(n*(n-1)/2, dtype=numpy.int8)
    else:
        if len(constraints.shape) > 1:
            constraints = distance.squareform(constraints)

        if len(constraints) != (n*(n-1)/2):
            raise ValueError('Wrong shape in constraint matrix')

        constraints = constraints.astype(numpy.int8)

    min_heap = heap(n*(n+1)/2)

    nonzero_constraints = constraints.nonzero()[0]

    # Creating heap with the constraints
    for nzr in nonzero_constraints:
        ii, jj = cond2mat_index(n, nzr)
        dist = numpy.linalg.norm(features[ii]-features[jj])
        min_heap.push(dist, nzr)

    Z = numpy.zeros((n-1, 4))

    method_as_int = int(method == 'centroid')

    # Starts clustering
    agglomerative(features, Z, n, min_heap, constraints, min_size,
                  method_as_int, verbose, copy)

    return Z
