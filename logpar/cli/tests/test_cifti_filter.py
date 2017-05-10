''' Testing CLI of clustering '''
from tempfile import NamedTemporaryFile

import numpy
import nibabel

from logpar.cli import cifti_filter
from logpar.utils import cifti_utils


def test_cifti_filter_both_direction():
    ''' Testing cifti_filter for both direction'''
    cifti_file = './logpar/cli/tests/data/filter.dconn.nii'
    cifti_file_T = './logpar/cli/tests/data/filter.transpose.dconn.nii'
    outfile = NamedTemporaryFile(mode='w', delete=True, suffix='.dconn.nii')
    outfile = outfile.name

    filter_txt = NamedTemporaryFile(mode='w', delete=True, suffix='.txt').name

    # cifti_file has the following entries:
    #   - 50 rows representing VERTICES from the CORTEX_LEFT, with value = 1
    #   - 100 rows representing VERTICES from the CORTEX_RIGHT, value = 2
    #   - 5 rows representing VOXELS from the BRAIN_STEAM, value = 3
    mtype = ['CIFTI_MODEL_TYPE_' + m for m in  ['SURFACE', 'SURFACE', 'VOXELS']]
    bstru = ['CIFTI_STRUCTURE_' + m for m in ['CORTEX_LEFT', 'CORTEX_RIGHT',
                                              'BRAIN_STEM']]
    voxels = [(22, 32, 22), (19, 32, 23), (22, 34, 24),
              (24, 33, 22), (21, 27, 24)]
    indices = [range(50), range(100), voxels]


    for j, (mt, bs, idx) in enumerate(zip(mtype, bstru, indices), 1):
        # Write the file with structures to filter
        with open(filter_txt, 'w') as f:
            f.write('#header\n')
            for i in idx:
                if mt == 'CIFTI_MODEL_TYPE_VOXELS':
                    i = '{} {} {}'.format(i[0], i[1], i[2])
                    s = i
                else:
                    s = '{0} {1} {1} {1}'.format(32492, i)  #size + fake seed

                f.write('{} {} {} {}\n'.format(mt, bs, i, s))

        # TEST ROW
        cifti_filter.cifti_filter(cifti_file, filter_txt, outfile)
        filtered = nibabel.load(outfile)
        filtered_data = filtered.get_data()[0, 0, 0, 0]
        numpy.testing.assert_equal(numpy.unique(filtered_data)[0], j)


        # TEST COLUMN
        cifti_filter.cifti_filter(cifti_file_T, filter_txt, outfile,
                                  direction='COLUMN')
        filtered = nibabel.load(outfile)
        filtered_data = filtered.get_data()[0, 0, 0, 0]
        numpy.testing.assert_equal(numpy.unique(filtered_data)[0], j)


if __name__ == "__main__":
    test_cifti_filter_both_direction()
