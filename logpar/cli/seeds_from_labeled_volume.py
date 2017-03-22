#!/usr/bin/env python
import os

import nibabel
from nibabel import gifti
import numpy
from collections import Counter
import tracpy.utils as utils

from scipy.ndimage import morphology

from logpar.utils.cifti_utils import CIFTI_STRUCTURES


def read_labels_file(labels_file):
    ''' Returns a dictionary with the labels as keys and which structure
        they represent as value '''

    valid_cifti_structures = set(CIFTI_STRUCTURES)
    
    label2structure = {}
    with open(labels_file) as f:
        for line in f:
            label, struc = line.split()
            label = int(label)
            
            if label in label2structure:
                raise ValueError("Repited label in labels_file")
            if struc not in valid_cifti_structures:
                raise ValueError("{} is not a valid structure".format(struc))
            label2structure[label] = struc
            
    return label2structure


def seeds_from_labeled_volume(volume_file, labels_file, seeds_per_voxel,
                              mask_file, mm, outfile):
    label2structure = read_labels_file(labels_file)


def subcortical_seeds(labels, wmparc, wmmask, outdir):
    ''' Create subcort seeds by expanding the subcortical labels in wmm '''
    wmmask = nibabel.load(wmmask)
    wmmater = wmmask.get_data().astype(bool)
    affine = wmmask.get_affine()

    wmparc = utils.nii2mat(wmparc)

    # Take labels from wmparc
    sub_cor = numpy.zeros_like(wmparc, dtype=int)
    for l in map(int, labels):
        sub_cor += (wmparc == l)*l

    # Dilate wmmask to get the border of subc regions
    wmmater_dilated = morphology.binary_dilation(wmmater)
    sc_seeds = numpy.multiply(wmmater_dilated, sub_cor)  # Inner border of sc
    
    # Take the upper border from the brain-steam
    sub_cor_no_steam = (sub_cor>0)*(sub_cor != 16)
    sc_expanded = morphology.binary_dilation(sub_cor_no_steam)
    brain_steam_border = numpy.multiply((wmparc == 16), sc_expanded)
    nzr = brain_steam_border.nonzero()
    sc_seeds[nzr] = 16

    # Retrieve the seeds as positions in mm space
    nzr = sc_seeds.nonzero()
    seeds_pos = numpy.transpose(nzr)
    values = sc_seeds[nzr]
    cifti_info = subcortical_info(values)

    seeds_list = nibabel.affines.apply_affine(affine, seeds_pos)
    seeds_and_info = numpy.hstack((seeds_list, cifti_info))
    return seeds_and_info



        sc_info += subcortical_seeds(args.labels, args.wmparc,
                                     args.wmmask, outdir).tolist()


    with open(args.out_list, 'w') as f:
        f.write('#Pos_x,Pos_y,Pos_z,CIFTI_STRUCTURE,cIndx,cTotal\n')
        f.writelines(','.join(str(p) for p in s) + '\n' for s in info)
