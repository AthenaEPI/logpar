import numpy

from tempfile import NamedTemporaryFile
from .. import streamline_utils

def test_save_and_load():
    ''' Tests the streamlines saving function '''
    streamlines = numpy.random.random((3,4,3))
    affine = numpy.array([[1.5,  0,  0, 1],
                          [  0,1.2,  0, 1],
                          [  0,  0,1.3, 1],
                          [  0,  0,  0, 1]])
    
    print streamlines
    output = NamedTemporaryFile(mode='w', delete=True, suffix='.trk').name
    streamline_utils.save_stream(output, streamlines, affine)

    recovered = streamline_utils.load_stream(output, affine)

    numpy.testing.assert_almost_equal(streamlines, recovered)
