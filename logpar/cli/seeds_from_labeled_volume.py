#!/usr/bin/env python
import os
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
            label, struc = line.split()
            label = int(label)
            
            if struc not in valid_cifti_structures:
                raise ValueError("{} is not a valid structure".format(struc))
            label2structure[label] = struc
            
    return label2structure


def seeds_from_labeled_volume(labeled_volume_file, labels_file,
                              seeds_per_voxel, mask_file, outfile,
                              vx_expand=0, style='border', verbose=1):
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
        mask = nibabel.load(mask_file).get_data()
    else:
        mask = numpy.ones_like(labels_volume)
    
    # Create header of the output file
    txtheader = "#CIFTI_MODEL_TYPE_VOXELS\n#Pos_x Pos_y Pos_z BrainStructure Vox_i Vox_j Vox_k\n"
    with open(outfile, 'w') as f:
        f.write(txtheader)

    # Create seeds from voxels
    xyz_struc_ijk = None
    seed_volume = numpy.zeros_like(labels_volume)  # Visual confirmation
    for label in label2structure:
        logging.debug('Procesing label: {}'.format(label))
        sc_structure = labels_volume==label
        # We dilate the structure *vx_expand* times, which could be zero
        dilated_structure = sc_structure
        for _ in xrange(vx_expand):
            dilated_structure = morphology.binary_dilation(dilated_structure)
        
        if style=='complete':
            # We seed from the whole structure
            seed_structure = dilated_structure  
        elif style=='border':
            # erode *vx_expand-1* times and substract to get the border
            eroded_structure = dilated_structure
            for _ in xrange(vx_expand+1):
                eroded_structure = morphology.binary_erosion(eroded_structure)
            seed_structure = dilated_structure - eroded_structure
        # Intersect with the mask
        nzr = numpy.multiply(seed_structure, mask).nonzero()
        nzr_positions = numpy.transpose(nzr)
        seed_volume[nzr] = label  
        # Create seeds randomly distributed inside of each voxel
        label_seeds = utils.random_seeds_from_mask(seed_structure,
                                                   seeds_per_voxel,
                                                   affine=labels_affine)
        # Save txt file with seed's information
        seeds_voxel = numpy.repeat(nzr_positions, seeds_per_voxel, 0)
        structure = numpy.repeat(label2structure[label], seeds_voxel.shape[0])
        label_all_data = numpy.hstack([label_seeds, structure[:, None],
                                       seeds_voxel])
        with open(outfile, 'a') as f:
            f.writelines((' '.join(line) + '\n' for line in label_all_data)) 
    # Save just for visual confirmation
    cifti_utils.save_cifti(outfile+'.nii', seed_volume, version=1)
