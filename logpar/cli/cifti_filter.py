#!/usr/bin/env python
''' Function called by the cli cifti_parcellate '''
import logging
from collections import defaultdict, OrderedDict

import numpy
import nibabel

from ..utils import cifti_utils, cifti_header, seeds_utils


def check_input(cifti_file, direction):
    ''' Basic input checking '''
    not_dconn = cifti_file.split('.')[-2] not in ['dconn']
    not_dconn_gz = cifti_file.split('.')[-3] not in ['dconn']
    if not_dconn and not_dconn_gz:
        raise ValueError('Only dconn files supported so far')

    if direction not in ['ROW', 'COLUMN']:
        raise ValueError('Direction should be either ROW or COLUMN')


def cifti_filter(cifti_file, filter_file, outfile, direction="ROW", verbose=0):
    ''' Given a cifti_file, it creates another cifti_file which, among
        the chosen dimention, only posses the structures present in
        filter_file

        Parameters
        ----------
        cifti_file: string
            Connectivity file in CIFTI format
        filter_file: string
            File with the structures to keep, the structure of the file
            is the same as the one handled by the seeds submodule
        outfile: string
            File where to output the dendrogram. The file SHOULD have a csv
            extension, if not, the extension is appended to the file name.
        direction: string (optional)
            Direction to parcellate from the CIFTI file. Default: 'ROW'

        Returns
        -------
        None
            One or more files are created '''
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    check_input(cifti_file, direction)

    # Load cifti file
    cifti = nibabel.load(cifti_file)
    data = cifti.get_data()[0, 0, 0, 0]

    # Retrieve all the structures/indices that should be keeped
    all_structures = OrderedDict({})
    all_sizes = {}
    cifti_info, _ = seeds_utils.load_seeds(filter_file)
    for mtype, bstructure, idx, size in cifti_info:
        key = (mtype, bstructure)
        if key not in all_structures:
            all_structures[key] = []
        all_structures[(mtype, bstructure)].append(idx)
        all_sizes[(mtype, bstructure)] = size

    # Get indices to filter from matrix and gatter info for the xml header
    new_order = []
    filtered_structures = []
    offset = 0
    for (modeltype, structure), indices in all_structures.iteritems():
        findices = cifti_utils.cifti_filter_indices(cifti.header, direction,
                                                    modeltype, structure,
                                                    indices)
        findices = numpy.array(findices)
        indices = numpy.array(indices)[findices!=-1]
        size = all_sizes[(modeltype, structure)]
        filtered_structures.append(cifti_header.brain_model_xml(modeltype,
                                                                structure,
                                                                indices,
                                                                offset, size))
        new_order += list(findices[findices!=-1])
        offset += len(indices)
    new_order = numpy.array(new_order)
    
    # Filter matrix
    if direction == 'ROW':
        filtered_data = data[new_order]
        row_structures = filtered_structures
        col_structures = cifti_utils.extract_brainmodel(cifti.header, 'COLUMN')
    else:
        filtered_data = data[:, new_order]
        row_structures = cifti_utils.extract_brainmodel(cifti.header, 'ROW')
        col_structures = filtered_structures

    row_dim, row_aff = cifti_utils.volume_attributes(cifti.header, 'ROW')
    col_dim, col_aff = cifti_utils.volume_attributes(cifti.header, 'COLUMN')

    # Create header
    header = cifti_header.create_conn_header(row_structures, col_structures,
                                             row_dim, col_dim,
                                             row_aff, col_aff)
    # Save file
    cifti_utils.save_nifti(outfile, filtered_data[None, None, None, None],
                           header, cifti.affine)
