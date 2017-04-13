import itertools
from collections import defaultdict

import numpy

from logpar.utils import streamline_utils


def connectivity_between_labels(streamlines, labeled_volume, affine=None,
                                starting_points=None):
    ''' Computes how many streamlines connect each pair of labels in the
        labeled volume.

        Parameters
        ----------
        streamlines: list of streamlines
        labeled_volume: volume with labels
        affine: transformation from rasmm to voxels in the labeled_volume,
                if affine is None, then the streamlines are already in vx space
        starting_points: list of points from where each streamline start

        Returns
        -------
        conn_matrix: array_like
            Returns a symmetrical connectivity matrix of labels*labels with
            the amount of streamlines that go from one label to another, this
            is: M[i,j] = number of streamlines going trough i and j 
        grouping: dictionary
            A dictionary grouping the streamlines by passing points. Given
            two labels i,j such that i < j, grouping[i,j] = streamlines
            that pass trough i and j. Note: the constraint of i < j is imposed
            to mimic the behaviour of connectivity_matrix from dipy. '''

    if starting_points is not None:
        raise NotImplementedError()

    # Make a connectivity matrix of max_labels+1 x max_labels+1
    max_label = labeled_volume.max()
    conn_matrix = numpy.zeros((max_label+1, max_label+1))
    grouping = defaultdict(list)

    if affine is not None:
        inv_affine = numpy.linalg.inv(affine)
        streamlines = [streamline_utils.transform_and_round(s, inv_affine)
                       for s in streamlines]

    for sl in streamlines:
        slx, sly, slz = numpy.transpose(sl)
        touched_labels = numpy.unique(labeled_volume[slx, sly, slz])
        conn_matrix[touched_labels[:, None], touched_labels] += 1
        for l1, l2 in itertools.combinations(touched_labels, 2):
            if l1 > l2:
                l1 , l2 = l2, l1
            grouping[l1, l2].append(sl)

    return conn_matrix, grouping
