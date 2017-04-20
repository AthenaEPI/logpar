''' Testing CLI of basic_hagmann '''
import logging

import numpy

from tempfile import NamedTemporaryFile

from .. import seeds_from_labeled_volume
from ...utils import cifti_utils, streamline_utils, seeds_utils


def test_one_random_seed():
    ''' Tests that the created seeds are inside the desired mask '''
    labeled_volume = numpy.zeros((10, 10, 10))
    x, y, z = numpy.random.randint(0, 3, 3)
    labeled_volume[x, y, z] = 1

    affine = numpy.eye(4)

    # Save files
    labvolfile = NamedTemporaryFile(mode='w', delete=True, suffix='.nii').name
    cifti_utils.save_cifti(labvolfile, labeled_volume, affine=numpy.eye(4))

    labelsfile = NamedTemporaryFile(mode='w', delete=True, suffix='.txt').name
    with open(labelsfile, 'w') as f:
        f.write('1 test1')

    outfile = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    seeds_per_voxel = 1
    # Call function
    seeds_from_labeled_volume.seeds_from_labeled_volume(labvolfile, labelsfile,
                                                        seeds_per_voxel,
                                                        outfile,
                                                        mask_file=None,
                                                        vx_expand=0,
                                                        style='complete')
    # Read result and compare
    cifti_info, seeds_pnts = seeds_utils.load_seeds(outfile)
    rounded = [streamline_utils.transform_and_round(pnt, transformation=affine)
               for pnt in seeds_pnts]
    numpy.testing.assert_almost_equal(rounded, [[[x, y, z]]*seeds_per_voxel])


    seeds_per_voxel = 100
    # Call function
    seeds_from_labeled_volume.seeds_from_labeled_volume(labvolfile, labelsfile,
                                                        seeds_per_voxel,
                                                        outfile,
                                                        mask_file=None,
                                                        vx_expand=0,
                                                        style='complete')
    # Read result and compare
    cifti_info, seeds_pnts = seeds_utils.load_seeds(outfile)
    rounded = [streamline_utils.transform_and_round(pnt, transformation=affine)
               for pnt in seeds_pnts]
    numpy.testing.assert_almost_equal(rounded, [[[x, y, z]]*seeds_per_voxel])


def test_one_random_seed_with_affine():
    ''' Tests that the created seeds are inside the desired mask '''
    labeled_volume = numpy.zeros((10, 10, 10))
    x, y, z = numpy.random.randint(0, 3, 3)
    labeled_volume[x, y, z] = 1

    affine = numpy.array([[1.5, 0, 0, 1],
                          [0, 1.2, 0, 1],
                          [0, 0, 1.3, 1],
                          [0, 0, 0, 1]])
    iaffine = numpy.linalg.inv(affine)

    # Save files
    labvolfile = NamedTemporaryFile(mode='w', delete=True, suffix='.nii').name
    cifti_utils.save_cifti(labvolfile, labeled_volume, affine=affine)

    labelsfile = NamedTemporaryFile(mode='w', delete=True, suffix='.txt').name
    with open(labelsfile, 'w') as f:
        f.write('1 test1')

    outfile = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    seeds_per_voxel = 1
    # Call function
    seeds_from_labeled_volume.seeds_from_labeled_volume(labvolfile, labelsfile,
                                                        seeds_per_voxel,
                                                        outfile,
                                                        mask_file=None,
                                                        vx_expand=0,
                                                        style='complete')
    # Read result and compare
    cifti_info, seeds_pnts = seeds_utils.load_seeds(outfile)
    rounded = [streamline_utils.transform_and_round(pnt, transformation=iaffine)
               for pnt in seeds_pnts]
    numpy.testing.assert_almost_equal(rounded, [[[x, y, z]]*seeds_per_voxel])


    seeds_per_voxel = 100
    # Call function
    seeds_from_labeled_volume.seeds_from_labeled_volume(labvolfile, labelsfile,
                                                        seeds_per_voxel,
                                                        outfile,
                                                        mask_file=None,
                                                        vx_expand=0,
                                                        style='complete')
    # Read result and compare
    cifti_info, seeds_pnts = seeds_utils.load_seeds(outfile)
    rounded = [streamline_utils.transform_and_round(pnt, transformation=iaffine)
               for pnt in seeds_pnts]
    numpy.testing.assert_almost_equal(rounded, [[[x, y, z]]*seeds_per_voxel])
