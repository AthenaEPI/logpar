''' Tool to average cifti files '''
import nibabel
import numpy

from logpar.utils import other_utils, cifti_utils


def check_input(matrix_files, outfile):
    ''' Basic input check '''
    
    conn_type = matrix_files[0].split('.')[-2]
    print conn_type

    if not all([name.split('.')[-2] == conn_type for name in matrix_files]):
        raise ValueError('All the input files MUST be of the same type')
    
    if conn_type != outfile.split('.')[-2]:
        raise ValueError('The output file MUST be of the same type as inputs')


def cifti_average(matrix_files, outfile):

    check_input(matrix_files, outfile)

    nbr_matrices = len(matrix_files)
    matrices = [nibabel.load(mat) for mat in matrix_files] 
    headers = [mat.header for mat in matrices]

    # First, we retrieve the strucutures/indices that all the subjects
    # share for both directions
    structures_manager = other_utils.CiftiMinimumCommonHeader()

    for header in headers:
        structures_manager.intersect_header(header)        

    sizeR, sizeC = structures_manager.get_matrix_size()

    average_connectivity = numpy.zeros((sizeR, sizeC), dtype=numpy.float32)

    matrices = [nibabel.load(mat) for mat in matrix_files] 
    for i, matrix in enumerate(matrices):
        print matrix_files[i]
        average_connectivity += structures_manager.extract_common_struc(matrix)

    average_connectivity /= nbr_matrices

    cifti_utils.save_cifti(outfile,
                           average_connectivity[None, None, None, None, ...],
                           header=structures_manager._header,
                           affine=matrices[0].affine)
