#!/usr/bin/env python
import logging

import nibabel
import numpy
from scipy.ndimage import morphology
from dipy.tracking import utils

from logpar.utils import cifti_utils


def read_labels_file(labels_file):
    ''' Returns a dictionary with the labels as keys and which structure
        they represent as value '''
    valid_cifti_structures = set(cifti_utils.CIFTI_STRUCTURES)

    label2structure = {}
    with open(labels_file) as f:
        for line in f:
            label, struc = line.split()[:2]
            label = int(label)

            #if struc not in valid_cifti_structures:
            #    print("{} is not a valid CIFTI structure".format(struc))
            label2structure[label] = struc

    return label2structure


def seeds_from_labeled_volume(labeled_volume_file, labels_file,
                              seeds_per_voxel, outfile, mask_file=None,
                              vx_expand=0, style='border', vol_out=None,
                              verbose=1):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    # Extract map labels -> structures
    label2structure = read_labels_file(labels_file)

    # Load volume with labels
    labels_nifti = nibabel.load(labeled_volume_file)
    labels_volume = labels_nifti.get_data()
    labels_affine = labels_nifti.affine

    # Load mask if any
    if mask_file:
        mask = nibabel.load(mask_file).get_data().astype(bool)
    else:
        mask = None

    # Create header of the output file
    txtheader = "#ModelType BrainStructure Vox_i Vox_j Vox_k Seeding_Points \n"
    with open(outfile, 'w') as f:
        f.write(txtheader)

    # Create seeds from voxels
    seed_volume = numpy.zeros_like(labels_volume)  # Visual confirmation
    text = ""

    for label in label2structure:
        logging.debug('Procesing label: {}'.format(label))
        label_mask = labels_volume==label

        # We dilate the structure *vx_expand* times, which could be zero
        if vx_expand:
            seed_structure = morphology.binary_dilation(label_mask, None,
                                                        vx_expand)
        else:
            seed_structure = label_mask

        if style == 'border':
          # erode the structure one time and substract to get the border
          eroded_structure = morphology.binary_erosion(label_mask)
          seed_structure = seed_structure - eroded_structure

        # Intersect with the mask
        if mask is not None:
            seed_structure = numpy.multiply(seed_structure, mask)
        nzr = seed_structure.nonzero()
        seed_volume[nzr] = label

        # Create seeds randomly distributed inside of each voxel
        nzr_positions = numpy.transpose(nzr)
        label_seeds = utils.random_seeds_from_mask(seed_structure,
                                                   seeds_per_voxel,
                                                   affine=labels_affine)
        # Add information to the output file
        structure_name = label2structure[label]
        for s, (i, j, k) in enumerate(nzr_positions):
            # Take the seed points for this voxel
            li, ls = s*seeds_per_voxel, s*seeds_per_voxel + seeds_per_voxel
            spoints = label_seeds[li:ls]
            ptxt = " ".join("{} {} {}".format(x, y, z) for x, y, z in spoints)
            # Add them to the text
            text += "CIFTI_MODEL_TYPE_VOXELS {} {} {} {} {}\n".format(
                structure_name, i, j, k, ptxt)

    with open(outfile, 'a') as f:
        f.write(text)
    # Save just for visual confirmation
    if vol_out:
        cifti_utils.save_cifti(vol_out, seed_volume,
                               header=labels_nifti.header,
                               affine=labels_nifti.affine, version=1)
