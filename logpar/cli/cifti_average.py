''' Tool to average cifti files '''
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


def cifti_average(matrix_files, outfile, in_logodds=False, only_common=False):
    check_input(matrix_files, outfile)

    nbr_matrices = len(matrix_files)
    matrices = [nibabel.load(mat) for mat in matrix_files]
    headers = [mat.header for mat in matrices]
    # First, we retrieve the strucutures/indices that all the subjects
    # share for both directions
    if only_common:
        header_function = cifti_header.header_intersection
    else:
        header_function = cifti_header.header_union

    average_header = headers[0]
    for header in headers[1:]:
        average_header = header_function(header, average_header)

    sizeR, sizeC = cifti_utils.matrix_size(average_header)

    average_connectivity = numpy.zeros((sizeR, sizeC), dtype=numpy.float32)

    matrices = [nibabel.load(mat) for mat in matrix_files] 
    for i, matrix in enumerate(matrices):
        subject_conn = cifti_utils.retrieve_head_data(average_header, matrix)
        if in_logodds:
            subject_conn = transform.to_logodds(subject_conn)

        average_connectivity += subject_conn

    average_connectivity /= nbr_matrices

    if in_logodds:
        average_connectivity = transform.from_logodds(average_connectivity)

    cifti_utils.save_nifti(outfile,
                           average_connectivity[None, None, None, None, ...],
                           header=average_header,
                           affine=matrices[0].affine)
