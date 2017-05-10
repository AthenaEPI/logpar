''' Tool to average cifti files '''
from collections import defaultdict
import nibabel
import numpy

from logpar.utils import cifti_header, cifti_utils, transform


def check_input(matrix_files, outfile):
    ''' Basic input check '''
    conn_type = matrix_files[0].split('.')[-2]

    if not all([name.split('.')[-2] == conn_type for name in matrix_files]):
        raise ValueError('All the input files MUST be of the same type')

    if conn_type != outfile.split('.')[-2]:
        raise ValueError('The output file MUST be of the same type as inputs')


def cifti_merge(matrix_files, direction, outfile):
    ''' Merges the matrix_files in a given direction

            Parameters
            ----------
            matrix_files: list(string)
                List of files to merge
            direction: string
                ROW or COLUMN
            outfile: string
                Output file

           Returns
           -------
           None    '''
    # Check input files are coherent
    check_input(matrix_files, outfile)

    # Count number of matrices and open them
    nbr_matrices = len(matrix_files)
    matrices = [nibabel.load(mat) for mat in matrix_files]
    headers = [mat.header for mat in matrices]

    # First, we retrieve the structures/indices of each brain model
    total = offset = 0
    all_structures = defaultdict(list)
    all_structures_meta = {}
    sidx2midx = {}
    for mat_idx, header in enumerate(headers):
        xml_entities = cifti_utils.extract_brainmodel(header, direction)
        offset = 0

        for xml_entity in xml_entities:
            structure = xml_entity.attrib['BrainStructure']
            model = xml_entity.attrib['ModelType']

            indices = cifti_utils.modeltext2indices(xml_entity[0].text, model)
            total += len(indices)

            # Gather the meta-data and the used indices of every structures
            if model == 'CIFTI_MODEL_TYPE_VOXELS':
                all_structures[(structure, model)] += indices
            else:
                current = all_structures[(structure, model)]
                all_structures[(structure, model)] = sorted(current + indices)
            all_structures_meta[(structure, model)] = xml_entity

            # Save a mapping from each cifti-structure idx to a matrix idx
            for i, idx in enumerate(indices):
                sidx2midx[(structure, model, idx)] = (mat_idx, i+offset)
            offset += len(indices)

    # Sort the structures by name
    all_structures = {k:all_structures[k]
                      for k in sorted(all_structures.keys())}

    # Matrix where we will store the merged result
    m = matrices[0].shape[4+(direction == 'ROW')]
    merged_matrix = numpy.zeros((total, m))

    # Create a map between each matrix and the merged_matrix in such a way
    # that we only need to read "once" from each matrix
    matrix_reorder = defaultdict(list)
    xml_rows = []
    offset = 0
    for (structure, model), indices in all_structures.iteritems():
        for i, idx in enumerate(indices):
            mat_idx, previous_i = sidx2midx[(structure, model, idx)]
            matrix_reorder[(mat_idx, 'from')].append(previous_i)
            matrix_reorder[(mat_idx, 'to')].append(i+offset)

        # Create xml_structure for the future cifti_header
        xml_entity = all_structures_meta[(structure, model)]
        xml_entity.attrib['IndexCount'] = str(len(indices))
        xml_entity.attrib['IndexOffset'] = str(offset)
        xml_entity[0].text = cifti_utils.indices2modeltext(indices, model)
        offset += len(indices)

        xml_rows.append(xml_entity)

    # Fill merged_matrix
    for i, cifti_matrix in enumerate(matrices):
        data = cifti_matrix.get_data()[0, 0, 0, 0]

        if direction == 'COLUMN':
            data = numpy.transpose(data)  # Check if this is performant enought
        to = numpy.array(matrix_reorder[(i, 'to')])
        fr = numpy.array(matrix_reorder[(i, 'from')])
        merged_matrix[to] = data[fr]

    # Create header : we already have the xml_entities for rows of our
    #  merged_matrix, now we need the xml_entities for the columns
    oposite_dir = 'COLUMN' if direction == 'ROW' else 'ROW'
    xml_cols = cifti_utils.extract_brainmodel(headers[0], oposite_dir)

    # Save merged matrix: So far we forced the merge to happen in the 0 dim,
    # this simplifies the process of handling indices. Now we need to put the
    # matrix in the right order. This is, if the merge was supposed to happen
    # in the COLUMNS, then we need to transpose the matrix
    rdim, raff = cifti_utils.volume_attributes(headers[0], 'ROW')
    cdim, caff = cifti_utils.volume_attributes(headers[0], 'COLUMN')

    if direction == 'COLUMN':
        merged_matrix = merged_matrix.T
        merged_header = cifti_header.create_conn_header(xml_cols, xml_rows,
                                                        rdim, cdim, raff, caff)
    else:
        merged_header = cifti_header.create_conn_header(xml_rows, xml_cols,
                                                        rdim, cdim, raff, caff)

    cifti_utils.save_nifti(outfile, merged_matrix[None, None, None, None],
                           header=merged_header, affine=matrices[0].affine)
