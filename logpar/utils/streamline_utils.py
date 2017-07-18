import numpy
import nibabel

from dipy.direction import ProbabilisticDirectionGetter, DeterministicMaximumDirectionGetter
from dipy.data import default_sphere

def save_stream(outfile, streamlist, affine_to_rasmm=None):
    ''' Function to save a set of streamlines. The streamlines can be in
        any space. If the streamlines are in rasmm space, then it's not
        necessary to specify a transformation. Otherwhise, the transformation
        from the current space should be specified.

        Parameters
        ----------
        streamlist: list of array-like
            List of streamlines to save
        affine_to_rasmm: array-like
            Transformation from the current space of the streamlines to the
            rasmm space

        Returns
        -------
        None '''
    if affine_to_rasmm is None:
        affine_to_rasmm = numpy.eye(4)

    tract = nibabel.streamlines.Tractogram(streamlist,
                                           affine_to_rasmm=affine_to_rasmm)
    nibabel.streamlines.save(tract, outfile)


def load_stream(streamfile, affine_to_rasmm=None):
    ''' Loads streamlines from a file. By default, nibabel.streamlines.load
        loads the streamlines in the ras+mm space, this function automatically
        transforms the loaded streamlines if an affine_to_rasmm is given.

        Parameters
        ----------
        streamfile: string
            Route to the file
        affine_to_rasm: array_like
            Transformation from some space to ras+mm. This matrix will
            be inverted and applied to the loaded streamlines
        Returns
        ------
        array_like
            List of streamlines
    '''
    # When loading, the streamlines are in RAS+ mm
    tractogram = nibabel.streamlines.load(streamfile)
    streamlines = tractogram.streamlines
    if affine_to_rasmm is not None:
        inv_affine = numpy.linalg.inv(affine_to_rasmm)
        streamlines = [nibabel.affines.apply_affine(inv_affine, s)
                       for s in streamlines]
    return streamlines


def direction_getter(shm, max_angle, algo='prob'):
    if algo == 'det':
        return DeterministicMaximumDirectionGetter.from_shcoeff(shm,
                                                                max_angle=max_angle,
                                                                sphere=default_sphere)
    else:
        return ProbabilisticDirectionGetter.from_shcoeff(shm, max_angle=30.,
                                                         sphere=default_sphere)

def length(streamline, transformation=None):
    ''' Computes the lenght of a streamline, if an transformation matrix is
        defined, then the streamline is transformed before computing its lenght

        Parameters
        ----------
        streamline: array-like
            Array of positions with shape (cant_positions, 3)
        transformation: array-like
            If present, then the streamline is in mm
            '''

    if transformation is not None:
        inv_trans = numpy.linalg.inv(transformation)
        streamline = nibabel.affines.apply_affine(inv_trans, streamline)

    linear_length = sum((numpy.linalg.norm(p)
                         for p in streamline[1:] - streamline[:-1]))
    return linear_length


def nbr_visited_voxels(streamline, affine=None):
    ''' Computes the number of voxels a streamline pass trought, if an
        affine is given, its inverse is computed and used to transform
        the streamline to voxel space

        Parameters
        ----------
        streamline: array-like
            Array of positions with shape (cant_positions, 3)
        affine: array-like
            If present, its inverse is used to transform the streamline
            '''
    if affine is not None:
        affine = numpy.linalg.inv(affine)
    return len(set(map(tuple, transform_and_round(streamline, affine))))


def transform_and_round(points, transformation=None):
    ''' Tansforms the points of a streamline to a new space, then it rounds
        down the resulting points. If no transformation is given, the points
        are only rounded down'''
    if transformation is not None:
        points = nibabel.affines.apply_affine(transformation, points)
    return numpy.floor(points).astype(int)
