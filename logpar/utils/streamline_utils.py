import numpy
import nibabel

from dipy.direction import ProbabilisticDirectionGetter, DeterministicMaximumDirectionGetter
from dipy.data import default_sphere

def save_stream(outfile, streamlist, affine):
    tractogram = nibabel.streamlines.Tractogram(streamlist,
                                                affine_to_rasmm=affine)
    nibabel.streamlines.save(tractogram, outfile)


def load_stream(streamfile, affine_to_rasmm=None):
    # When loading, the streamlines are in RAS+ mm
    tractogram = nibabel.streamlines.load(streamfile)
    streamlines = tractogram.streamlines
    if affine_to_rasmm is not None:
        inv_affine = numpy.linalg.inv(affine_to_rasmm)
        streamlines = [nibabel.affines.apply_affine(inv_affine, s)
                       for s in streamlines]
    return numpy.array(streamlines)


def direction_getter(shm, max_angle, algo='prob'):
    if algo == 'det':
        return DeterministicMaximumDirectionGetter.from_shcoeff(shm,
                                                                max_angle=max_angle,
                                                                sphere=default_sphere)
    else:
        return ProbabilisticDirectionGetter.from_shcoeff(shm, max_angle=30.,
                                                         sphere=default_sphere)

def longitud(streamline, vox2mm=None):
    
    if vox2mm is not None:
        inv_affine = numpy.linalg.inv(vox2mm)
        streamline = nibabel.affines.apply_affine(inv_affine, streamline)
    rounded = streamline.round().astype(int)
    as_set = set(map(tuple, rounded.astype(int)))

    return len(as_set)


def transform_and_round(points, affine):
    vx = nibabel.affines.apply_affine(affine, points)
    return numpy.round(vx).astype(int)
