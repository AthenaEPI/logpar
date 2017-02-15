''' Tests for tracpy/clustering.py '''
from collections import Counter

import nibabel
import numpy
import scipy.cluster.hierarchy as sci_hie
import logpar.clustering as our_hie


def random_feat_matrix(n, m):
    ''' Returns a random feature matrix '''
    return numpy.random.random((n, m))


def try_function(f):
    ''' Tests a function and returns if it failed or not '''
    failed = False
    try:
        f()
    except:
        failed = True
    return failed


def test_invalid_cases_fails():
    ''' Tests different cases of wrong configurations which SHOULD fail
        TODO: Improve THIS test'''
    features = random_feat_matrix(600, 750)
    n = features.shape[0]
    wrong_const = numpy.ones(n-1)
    right_const = numpy.ones((n,n)) - numpy.eye(n)

    # 1. Wrong method - 2. Wrong constraints - 3. Wrong clustering size
    failing_functions = [lambda: our_hie.clustering(features, method='none'),
                         lambda: our_hie.clustering(features, 
                                                    constraints=wrong_const),
                         lambda: our_hie.clustering(features, 
                                                    constraints=right_const,
                                                    min_size=-3)]

    for i, f in enumerate(failing_functions):
        if not try_function(f):
            raise Exception('Function number {} did not fail!'.format(i))


def test_clustering_is_same_as_scipy():
    ''' Basic clustering returns the same as scipy  '''
    features = random_feat_matrix(600, 705)

    scipy_ward = sci_hie.ward(features)
    our_ward = our_hie.clustering(features)

    scipy_centroid = sci_hie.centroid(features)
    our_centroid = our_hie.clustering(features, method='centroid')

    numpy.testing.assert_almost_equal(scipy_ward, our_ward)
    numpy.testing.assert_almost_equal(scipy_centroid, our_centroid)
    return True


def test_clustering_is_same_as_scipy_2():
    ''' Basic clustering returns the same as scipy 2 '''
    cifti = nibabel.load('./logpar/cli/tests/data/test.dconn.nii')
    features = cifti.get_data()[0, 0, 0, 0]

    scipy_ward = sci_hie.ward(features)
    our_ward = our_hie.clustering(features)

    scipy_centroid = sci_hie.centroid(features)
    our_centroid = our_hie.clustering(features, method='centroid')

    numpy.testing.assert_almost_equal(scipy_ward[:, :2], our_ward[:, :2])
    numpy.testing.assert_almost_equal(scipy_centroid[:, :2],
                                      our_centroid[:, :2])
    return True


def test_all_neighbors_is_same_as_scipy():
    ''' Clustering without constraints returns the same as scipy  '''
    features = random_feat_matrix(200, 100)
    n = features.shape[0]
    all_neighbors = numpy.ones((n, n)) - numpy.eye(n)

    scipy_ward = sci_hie.ward(features)
    our_ward = our_hie.clustering(features, method='ward',
                                  constraints=all_neighbors)

    scipy_centroid = sci_hie.centroid(features)
    our_centroid = our_hie.clustering(features, method='centroid',
                                      constraints=all_neighbors)

    numpy.testing.assert_almost_equal(scipy_ward, our_ward)
    numpy.testing.assert_almost_equal(scipy_centroid, our_centroid)


def test_minimum_size():
    ''' Minimum size is respected '''
    features = random_feat_matrix(200, 100)
    n = features.shape[0]
    all_neighbors = numpy.ones((n, n)) - numpy.eye(n)

    s = 25
    Z = our_hie.clustering(features, constraints=all_neighbors, min_size=s)

    # Take the finest granularity
    parcels = sci_hie.fcluster(Z, Z[:, 2].min(), criterion='distance')

    # Each resulting parcel could be as big as (2s-2), since in the worst case
    # two clusters of size (s-1) would be merged
    count = Counter(parcels)
    assert(all([c <= 2*s for c in count]))


def test_copy_flag():
    ''' Tests copy flag '''
    features = random_feat_matrix(200, 100)
    features_copy = features.copy()
    n = features.shape[0]
    all_neighbors = numpy.ones((n, n)) - numpy.eye(n)

    s = 10
    Z = our_hie.clustering(features, constraints=all_neighbors, min_size=s)

    numpy.testing.assert_equal(features, features_copy)

    Z = our_hie.clustering(features, constraints=all_neighbors, min_size=s,
                           copy=False)
    failed = False
    try:
        numpy.testing.assert_equal(features, features_copy)
    except AssertionError:
        failed = True

    assert(failed)


def test_patches_of_neighbors():
    ''' If the features are organized as patches then the resulting clustering
        has contiguous patches '''
    features = random_feat_matrix(200, 100)
    n = features.shape[0]

    # The features are organized in space as disjoint patches
    patches = numpy.zeros((n, n))
    s = 10
    for i in xrange(0, n, s):
        patches[i:i+s, i:i+s] = numpy.ones((s, s)) - numpy.eye(s)

    # Cluster using patches constraints
    Z = our_hie.clustering(features, constraints=patches, min_size=s)

    # Take the finest granularity
    parcels = sci_hie.fcluster(Z, Z[:, 2].min(), criterion='distance')

    # Check they're contiguous patches
    for i in xrange(0, n, s):
        assert(all([v == parcels[i] for v in parcels[i:i+s]]))
