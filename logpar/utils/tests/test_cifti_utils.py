''' Test cifti_utils.py '''
import xml.etree.ElementTree as xml

from nibabel import gifti
import nibabel
import numpy

from scipy.spatial.distance import squareform

from .. import cifti_utils, transform


def test_constraint_structure():
    ''' Tests if we recover the right structure from a gifti surface '''
    l_surf = gifti.read('./logpar/cli/tests/data/L.white.surf.gii')
    r_surf = gifti.read('./logpar/cli/tests/data/R.pial.surf.gii')

    l_struct = cifti_utils.principal_structure(l_surf)
    r_struct = cifti_utils.principal_structure(r_surf)

    assert(l_struct == 'CIFTI_STRUCTURE_CORTEX_LEFT')
    assert(r_struct == 'CIFTI_STRUCTURE_CORTEX_RIGHT')


def test_surface_attributes():
    ''' Tests if the offset is correct. We should add testing of indices '''
    cifti_test = nibabel.load('./logpar/cli/tests/data/test.dconn.nii')

    header = cifti_test.header

    offset_l, indices_l = cifti_utils.surface_attributes(
        header, 'CIFTI_STRUCTURE_CORTEX_LEFT', "ROW"
    )
    assert(offset_l == 0)
    assert(len(indices_l) == 200)

    offset_r, indices_r = cifti_utils.surface_attributes(
        header, 'CIFTI_STRUCTURE_CORTEX_RIGHT', "ROW"
    )
    assert(offset_r == len(indices_l))
    assert(len(indices_r) == 400)

    offset_l, indices_l = cifti_utils.surface_attributes(
        header, 'CIFTI_STRUCTURE_CORTEX_LEFT', "COLUMN"
    )
    assert(offset_l == 455)
    assert(len(indices_l) == 250)

    offset_r, indices_r = cifti_utils.surface_attributes(
        header, 'CIFTI_STRUCTURE_CORTEX_RIGHT', "COLUMN"
    )
    assert(offset_r == 0)
    assert(len(indices_r) == 450)


def test_extract_all_xml_structure():
    ''' Tests that we extract the right xml structures '''
    cifti_test = nibabel.load('./logpar/cli/tests/data/test.dconn.nii')
    header = cifti_test.header
   
    xml_structures = cifti_utils.extract_brainmodel(header, 'ALL', 'COLUMN')
    
    assert(len(xml_structures) == 3)

    for i, bmodel in enumerate(['CIFTI_STRUCTURE_CORTEX_RIGHT',
                                'CIFTI_STRUCTURE_BRAIN_STEM',
                                'CIFTI_STRUCTURE_CORTEX_LEFT']):
        numpy.testing.assert_equal(xml_structures[i].attrib['BrainStructure'],
                                   bmodel)


def test_constraint_matrix():
    ''' Test that the constraint matrix is correctly generated '''
    surf_left = gifti.read('./logpar/cli/tests/data/L.white.surf.gii')

    triangles = surf_left.darrays[1].data
    N = len(surf_left.darrays[0].data)

    ady_matrix = numpy.zeros((N, N), dtype=numpy.int8)

    for (edg1, edg2, edg3) in triangles:
        ady_matrix[edg1, edg2] = ady_matrix[edg2, edg1] = 1
        ady_matrix[edg1, edg3] = ady_matrix[edg3, edg1] = 1
        ady_matrix[edg2, edg3] = ady_matrix[edg3, edg2] = 1

    vertices = numpy.arange(0, N, 2)

    sub_ady_matrix = cifti_utils.constraint_from_surface(surf_left, vertices)

    base = squareform(ady_matrix[vertices[:, None], vertices])

    numpy.testing.assert_equal(sub_ady_matrix, base)
