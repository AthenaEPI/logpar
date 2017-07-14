''' Testing CLI of clustering '''
import logging

import numpy
import nibabel
from scipy.cluster.hierarchy import ward as sci_ward
import xml.etree.ElementTree as xml
from nibabel import gifti
from tempfile import NamedTemporaryFile

from ... import clustering as our_hie
from .. import cifti_parcellate
from ...utils import cifti_utils, transform
from ...utils import dendrogram_utils


def test_correctly_configured():
    ''' Tests the clustering with many different parameters
        TODO: Improve this '''

    cifti_test = './logpar/cli/tests/data/test.dconn.nii'
    l_surf = './logpar/cli/tests/data/L.white.surf.gii'
    r_surf = './logpar/cli/tests/data/R.pial.surf.gii'

    cifti = nibabel.load(cifti_test)

    header = cifti.header
    data = cifti.get_data()[0, 0, 0, 0]

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    for direction in ["COLUMN", "ROW"]:
        for constraint in [None, l_surf, r_surf]:
            for to_logodds in [True, False]:
                logging.info('--- Starting: {} {} {} ---'.format(direction,
                                                                 constraint,
                                                                 to_logodds))
                # Result from cifti_parcellate CLI
                cifti_parcellate.cifti_parcellate(cifti_test, output,
                                                  direction=direction,
                                                  constraint=constraint,
                                                  to_logodds=to_logodds)
                dendro, sxml = dendrogram_utils.load(output)

                # Result from clustering function
                test_data = data.copy()
                if direction == 'COLUMN':
                    test_data = numpy.transpose(test_data)

                ady_matrix = None
                struc, model = None, None
                if constraint is not None:
                    surf = gifti.read(constraint)
                    struc = cifti_utils.principal_structure(surf)
                    model = 'CIFTI_MODEL_TYPE_SURFACE'
                    off, ind = cifti_utils.offset_and_indices(
                        header, model, struc, direction
                    )
                    test_data = test_data[off:off+len(ind)]

                    ady_matrix = cifti_utils.constraint_from_surface(surf, ind)

                if to_logodds:
                    test_data = transform.to_logodds(test_data, True)

                # Parcellate directly with the parcelling function
                if constraint is None:
                    dendro2 = sci_ward(test_data)
                else:
                    dendro2 = our_hie.clustering(test_data,
                                                 constraints=ady_matrix)
                # The results should be the same
                numpy.testing.assert_almost_equal(dendro, dendro2, 5)

                cifti_xml = cifti_utils.extract_brainmodel(header, direction,
                                                           model, struc)
                new_offset = 0
                vstruct = False
                for structure in cifti_xml:
                    vstruct += structure.attrib['ModelType'] == 'CIFTI_MODEL_TYPE_VOXELS'
                    structure.attrib['IndexOffset'] = str(new_offset)
                    new_offset += int(structure.attrib['IndexCount'])

                if vstruct > 0:
                    # There's a voxel structure: ,add volume information
                    volume_xml = cifti_utils.extract_volume(header, direction)
                    cifti_xml += volume_xml
                
                numpy.testing.assert_equal(len(sxml), len(cifti_xml))

                for original, retrieved in zip(cifti_xml, sxml):
                    numpy.testing.assert_equal(xml.tostring(retrieved),
                                               xml.tostring(original))


def test_correctly_configured_32bit():
    ''' Tests the clustering with many different parameters using 32bit
        TODO: Improve this '''

    cifti_test = './logpar/cli/tests/data/test.dconn.nii'
    l_surf = './logpar/cli/tests/data/L.white.surf.gii'
    r_surf = './logpar/cli/tests/data/R.pial.surf.gii'

    cifti = nibabel.load(cifti_test)

    header = cifti.header
    data = cifti.get_data()[0, 0, 0, 0].astype(numpy.float32)

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    for direction in ["ROW", "COLUMN"]:
        for constraint in [None, l_surf, r_surf]:
            for to_logodds in [True, False]:
                # Result from cifti_parcellate CLI
                cifti_parcellate.cifti_parcellate(cifti_test, output,
                                                  direction=direction,
                                                  constraint=constraint,
                                                  to_logodds=to_logodds)
                dendro1 = numpy.loadtxt(output, delimiter=',')

                # Result from clustering function
                test_data = data.copy()
                if direction == 'COLUMN':
                    test_data = test_data.T

                ady_matrix = None
                if constraint is not None:
                    surf = gifti.read(constraint)
                    struc = cifti_utils.principal_structure(surf)
                    model = 'CIFTI_MODEL_TYPE_SURFACE'
                    off, ind = cifti_utils.offset_and_indices(header, model,
                                                              struc, direction)
                    test_data = test_data[off:off+len(ind)]

                    ady_matrix = cifti_utils.constraint_from_surface(surf, ind)

                if to_logodds:
                    test_data = transform.to_logodds(test_data, True)

                dendro2 = our_hie.clustering(test_data, constraints=ady_matrix)

                # They should be the same
                try:
                    numpy.testing.assert_almost_equal(dendro1, dendro2)
                except AssertionError:
                    print direction, constraint, to_logodds, ady_matrix
                    raise
