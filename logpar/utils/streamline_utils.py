import numpy
import nibabel


def save_stream(outfile, streamlist, affine):
    tractogram = nibabel.streamlines.Tractogram(streamlist, affine_to_rasmm=affine)
    nibabel.streamlines.save(tractogram, outfile)

def load_stream(streamfile, affine_to_rasm):
    # When loading, the streamlines are in RAS+ mm
    tractogram = nibabel.streamlines.load(streamfile)
    streamlines = tractogram.streamlines
    Iaffine = numpy.linalg.inv(affine_to_rasm)
    streamlines = nibabel.affines.apply_affine(Iaffine, streamlines)
    return numpy.array(streamlines)
