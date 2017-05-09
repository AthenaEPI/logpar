''' Testing CLI of clustering '''
from tempfile import NamedTemporaryFile

import numpy
import nibabel

from logpar.cli import cifti_merge
from logpar.utils import cifti_utils


def test_cifti_merge_row_direction():
    ''' Testing cifti_merge for ROW direction'''
    # This TEST is TOO CLOSE to the IMPLEMENTATION, please IMPROVE
    matrix_files = ['./logpar/cli/tests/data/merge1.dconn.nii',
                    './logpar/cli/tests/data/merge2.dconn.nii',
                    './logpar/cli/tests/data/merge3.dconn.nii']
    outfile = NamedTemporaryFile(mode='w', delete=True, suffix='.dconn.nii')
    outfile = outfile.name

    cifti_merge.cifti_merge(matrix_files, 'ROW', outfile)
    
    # The resulting matrix should be:
    #   - 15 rows representing VOXELS from the BRAIN_STEAM, every 5 rows the
    #     value MUST change linearly, starting from 1
    #   - 150 rows representing VERTICES from the CORTEX_LEFT, every 50 rows
    #     the value MUST change linearly, starting from 1
    #   - 300 rows representing VERTICES from the CORTEX_RIGHT, every 100 rows
    #     the value MUST change linerly, starting from 1
    expected = numpy.ones((465, 150))
    rows = [5, 50, 100]
    o = 0
    for r in rows:
        for i in xrange(3):
            expected[r*i + o: r*i + r + o] = i+1
        o += r*3

    merged = nibabel.load(outfile)
    merged_data = merged.get_data()[0, 0, 0, 0]
    
    numpy.testing.assert_almost_equal(merged_data, expected)
    
    expected_structure = ['BRAIN_STEM', 'CORTEX_LEFT', 'CORTEX_RIGHT']
    expected_structure = ['CIFTI_STRUCTURE_' + e for e in expected_structure]
    row_structures = cifti_utils.extract_brainmodel(merged.header, 'ALL', 'ROW')
    
    for i, structure in enumerate(row_structures):
        numpy.testing.assert_equal(structure.attrib['BrainStructure'],
                                   expected_structure[i])


def test_cifti_merge_col_direction():
    ''' Testing cifti_merge for COLUMN direction'''
    # This TEST is TOO CLOSE to the IMPLEMENTATION, please IMPROVE
    matrix_files = ['./logpar/cli/tests/data/merge1.transpose.dconn.nii',
                    './logpar/cli/tests/data/merge2.transpose.dconn.nii',
                    './logpar/cli/tests/data/merge3.transpose.dconn.nii']
    outfile = NamedTemporaryFile(mode='w', delete=True, suffix='.dconn.nii')
    outfile = outfile.name

    cifti_merge.cifti_merge(matrix_files, 'COLUMN', outfile)
    
    # The resulting matrix should be:
    #   - 15 cols representing VOXELS from the BRAIN_STEAM, every 5 rows the
    #     value MUST change linearly, starting from 1
    #   - 150 cols representing VERTICES from the CORTEX_LEFT, every 50 rows
    #     the value MUST change linearly, starting from 1
    #   - 300 cols representing VERTICES from the CORTEX_RIGHT, every 100 rows
    #     the value MUST change linerly, starting from 1
    expected = numpy.ones((465, 150))
    rows = [5, 50, 100]
    o = 0
    for r in rows:
        for i in xrange(3):
            expected[r*i + o: r*i + r + o] = i+1
        o += r*3
    expected = expected.T

    merged = nibabel.load(outfile)
    merged_data = merged.get_data()[0, 0, 0, 0]
    
    numpy.testing.assert_almost_equal(merged_data, expected)
    
    expected_structure = ['BRAIN_STEM', 'CORTEX_LEFT', 'CORTEX_RIGHT']
    expected_structure = ['CIFTI_STRUCTURE_' + e for e in expected_structure]
    row_structures = cifti_utils.extract_brainmodel(merged.header, 'ALL', 'COLUMN')
    
    for i, structure in enumerate(row_structures):
        numpy.testing.assert_equal(structure.attrib['BrainStructure'],
                                   expected_structure[i])


if __name__ == "__main__":
    test_cifti_merge_row_direction()
    test_cifti_merge_col_direction()
