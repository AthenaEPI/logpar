''' Tests for logpar/utils/streamline_utils '''
import numpy
import nibabel

from tempfile import NamedTemporaryFile
from .. import streamline_utils


def test_save_and_load():
    ''' Tests the streamlines saving function '''
    streamlines = numpy.random.random((3, 4, 3))
    streamlines[streamlines < 0] = 0

    affine = numpy.array([[1.5, 0, 0, 1],
                          [0, 1.2, 0, 1],
                          [0, 0, 1.3, 1],
                          [0, 0, 0, 1]])

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name

    # With transformation
    streamline_utils.save_stream(output, streamlines, affine)
    recovered = streamline_utils.load_stream(output, affine)
    numpy.testing.assert_almost_equal(streamlines, recovered)

    # Without transformation
    streamline_utils.save_stream(output, streamlines)
    recovered = streamline_utils.load_stream(output)
    numpy.testing.assert_almost_equal(streamlines, recovered)


def test_length_in_voxels():
    ''' Tests that the lenght of a streamline is correctly computed '''
    stream = numpy.array([[0, 0, 0],
                          [0.5, 0, 0],
                          [0.5, 1, 0],
                          [1, 1, 0],
                          [1, 2, 0],
                          [1, 2, 0],
                          [1, 2, 2]])

    affine = numpy.array([[1.5, 0, 0, 1],
                          [0, 1.2, 0, 1],
                          [0, 0, 1.3, 1],
                          [0, 0, 0, 1]])

    numpy.testing.assert_equal(5, streamline_utils.length(stream))

    stream_mm = nibabel.affines.apply_affine(affine, stream)

    numpy.testing.assert_equal(5, streamline_utils.length(stream_mm, affine))


def test_nbr_visited_voxels():
    ''' Tests that the number of visited voxels is correctly computed '''
    streamline0 = numpy.array([[0, 0, 0], [0, 0, 1]])
    streamline1 = numpy.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
    streamline2 = numpy.array([[1, 0, 0], [0, 0, 1], [0, 0, 2]])
    streamline3 = numpy.array([[0, 0, 0], [1, 1, 1], [1, 2, 3],
                               [1, 1, 1], [0, 0, 0]])

    numpy.testing.assert_equal(2,
                               streamline_utils.nbr_visited_voxels(streamline0))
    numpy.testing.assert_equal(4,
                               streamline_utils.nbr_visited_voxels(streamline1))
    numpy.testing.assert_equal(3,
                               streamline_utils.nbr_visited_voxels(streamline2))
    numpy.testing.assert_equal(3,
                               streamline_utils.nbr_visited_voxels(streamline3))

    affine = numpy.array([[1.5, 0, 0, 1],
                          [0, 1.2, 0, 1],
                          [0, 0, 1.3, 1],
                          [0, 0, 0, 1]])
    stream_mm = nibabel.affines.apply_affine(affine, streamline1)

    numpy.testing.assert_equal(4,
                               streamline_utils.nbr_visited_voxels(stream_mm,
                                                                   affine))

