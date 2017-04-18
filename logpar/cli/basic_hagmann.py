''' Computes a connectivity matrix as in Hagmann et al 2007 '''
import os
import nibabel
import numpy
import dipy.tracking.utils as dutils

import pandas as pd

from logpar.utils import streamline_utils, connectivity
from logpar.cli.seeds_from_labeled_volume import read_labels_file


def correct_name(name):
    starts = ['CTX-LH', 'WM-LH', 'CTX-RH', 'WM-RH']
    replace = ['LEFT', 'LEFT', 'RIGHT', 'RIGHT']
    for i in xrange(len(starts)):
        if name.startswith(starts[i]):
            name = '{}{}'.format(replace[i], name[len(starts[i]):])
    return name


def basic_hagmann(streamlines_file, labeled_volume_file, labels_file, outfile):
    labeled_volume_nifti = nibabel.load(labeled_volume_file)
    affine = labeled_volume_nifti.get_affine()
    labeled_volume = labeled_volume_nifti.get_data().astype(int)

    # Get the dictionary labels -> name
    l2n = read_labels_file(labels_file)
    n2l = sorted((correct_name(v),k) for k, v in l2n.iteritems())
    ordered_names = [name for name, _ in n2l]
    ordered_labels = [label for _, label in n2l]
    nlabels = len(ordered_names)

    # Load streamlines and Remove wrong ones
    print "Loading Streamlines"
    streamlines = streamline_utils.load_stream(streamlines_file)
    streamlines = [s for s in streamlines if len(s) > 1]

    # Now lets calculate the connectivity matrix
    print "Computing connectivity map"
    _, grouping = connectivity.connectivity_between_labels(streamlines,
                                                           labeled_volume,
                                                           affine=affine)

    # grouping is a dictionary with keys (label_i, label_j) SUCH THAT
    # lab_i < lab_j. The values are the streamlines between those regions.

    # Calculate area in voxels of each parcel
    areas = [int((labeled_volume == l).sum()) for l in ordered_labels]

    # reduce M to get only labels, excluding the background
    conn_matrix = numpy.zeros((nlabels, nlabels))

    r2M = {i: l for i, l in enumerate(ordered_labels)}

    for i in xrange(nlabels):
        for j in xrange(i+1, nlabels):
            mi, mj = r2M[i], r2M[j]
            if mj < mi:
                mi, mj = mj, mi  # requeriment for grouping indexing
            area_sum = areas[i] + areas[j]
            len_sum = sum([1./streamline_utils.nbr_visited_voxels(s, affine)
                           for s in grouping[mi, mj]])
            conn_matrix[j, i] = conn_matrix[i, j] = 2. * len_sum / area_sum

    df = pd.DataFrame(conn_matrix, index=ordered_names,
                      columns=ordered_names)

    df.to_csv(outfile)
