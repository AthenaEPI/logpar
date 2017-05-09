''' Testing CLI of basic_hagmann '''
import logging

import numpy
import pandas

from tempfile import NamedTemporaryFile

from .. import basic_hagmann
from ...utils import cifti_utils, streamline_utils


def test_basic_3d_volume():
    ''' Tests a basic configuration for hamming matrix '''
    labeled_volume = numpy.zeros((3, 3, 3))

    labeled_volume[:, 0, 0] = 1
    labeled_volume[0, 2, 0] = 2
    labeled_volume[0, 1, 0] = 2
    labeled_volume[0, 0, 2] = 3

    # 1 -> 2
    strm0 = numpy.array([[0, 0, 0], [0, 1, 0]])
    strm1 = numpy.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])

    # 3 -> 1
    strm2 = numpy.array([[0, 0, 2], [0, 1, 2], [0, 1, 1],
                         [0, 0, 1], [0, 0, 0]])
    strm3 = numpy.array([[0, 0, 2], [0, 0, 1], [0, 0, 0]])
    strm4 = numpy.array([[0, 0, 2], [1, 0, 1], [2, 0, 0]])

    connectivity = [[0, 0.3, 0.43],
                    [0.3, 0, 0],
                    [0.43, 0, 0]]

    # Save files
    strms = [strm0, strm1, strm2, strm3, strm4]

    streamfile = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name
    streamline_utils.save_stream(streamfile, strms)

    labvolfile = NamedTemporaryFile(mode='w', delete=True, suffix='.nii').name
    cifti_utils.save_nifti(labvolfile, labeled_volume, affine=numpy.eye(4))

    labelsfile = NamedTemporaryFile(mode='w', delete=True, suffix='.txt').name
    with open(labelsfile, 'w') as f:
        f.write('1 test1\n2 test2\n3 test3')

    outfile = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    # Call function
    basic_hagmann.basic_hagmann(streamfile, labvolfile, labelsfile, outfile)

    # Read result and compare
    retrieved = numpy.array(pandas.read_csv(outfile))[:, 1:]

    numpy.testing.assert_almost_equal(retrieved, connectivity, 2)
