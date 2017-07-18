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


def test_not_constrained():
    """Test the clustering without applying constraints"""
    cifti_test = './logpar/cli/tests/data/test.dconn.nii'
    cifti = nibabel.load(cifti_test)

    header = cifti.header
    data = cifti.get_data()[0, 0, 0, 0]

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    for direction in ["COLUMN", "ROW"]:
        for to_logodds in [True, False]:
            logging.info('--- Starting: {} {} ---'.format(direction,
                                                          to_logodds))
            # Result from cifti_parcellate CLI
            cifti_parcellate.cifti_parcellate(cifti_test, output,
                                              direction=direction,
                                              constrained=False,
                                              to_logodds=to_logodds)
            dendro, sxml = dendrogram_utils.load(output)

            # Result from clustering function
            test_data = data.copy()
            if direction == 'COLUMN':
                test_data = numpy.transpose(test_data)

            if to_logodds:
                test_data = transform.to_logodds(test_data, True)

            # Parcellate directly with scipy ward
            dendro2 = sci_ward(test_data)

            # The results should be the same
            numpy.testing.assert_almost_equal(dendro, dendro2, 5)

            cifti_xml = cifti_utils.extract_brainmodel(header, direction)

            new_offset = 0
            vstruct = False
            for structure in cifti_xml:
                vstruct += structure.attrib['ModelType'] == cifti_utils.VOXEL
                structure.attrib['IndexOffset'] = str(new_offset)
                new_offset += int(structure.attrib['IndexCount'])

            if vstruct > 0:
                # There's a voxel structure: ,add volume information
                volume_xml = cifti_utils.extract_volume(header, direction)
                cifti_xml += volume_xml
            
            assert(len(cifti_xml) == len(sxml))
            for original, retrieved in zip(cifti_xml, sxml):
                numpy.testing.assert_equal(xml.tostring(retrieved),
                                           xml.tostring(original))

def test_not_constrained_32bit():
    """Test the clustering without applying constraints using 32 bit data"""
    cifti_test = './logpar/cli/tests/data/test.dconn.nii'
    cifti = nibabel.load(cifti_test)

    header = cifti.header
    data = cifti.get_data()[0, 0, 0, 0].astype(numpy.float32)

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    for direction in ["COLUMN", "ROW"]:
        for to_logodds in [True, False]:
            logging.info('--- Starting: {} {} ---'.format(direction,
                                                          to_logodds))
            # Result from cifti_parcellate CLI
            cifti_parcellate.cifti_parcellate(cifti_test, output,
                                              direction=direction,
                                              constrained=False,
                                              to_logodds=to_logodds)
            dendro, sxml = dendrogram_utils.load(output)

            # Result from clustering function
            test_data = data.copy()
            if direction == 'COLUMN':
                test_data = numpy.transpose(test_data)

            if to_logodds:
                test_data = transform.to_logodds(test_data, True)

            # Parcellate directly with scipy ward
            dendro2 = sci_ward(test_data)

            # The results should be the same
            numpy.testing.assert_almost_equal(dendro, dendro2, 5)

            cifti_xml = cifti_utils.extract_brainmodel(header, direction)

            new_offset = 0
            vstruct = False
            for structure in cifti_xml:
                vstruct += structure.attrib['ModelType'] == cifti_utils.VOXEL
                structure.attrib['IndexOffset'] = str(new_offset)
                new_offset += int(structure.attrib['IndexCount'])

            if vstruct > 0:
                # There's a voxel structure: ,add volume information
                volume_xml = cifti_utils.extract_volume(header, direction)
                cifti_xml += volume_xml

            assert(len(cifti_xml) == len(sxml))

            for original, retrieved in zip(cifti_xml, sxml):
                numpy.testing.assert_equal(xml.tostring(retrieved),
                                           xml.tostring(original))


def test_surface_constrained_clustering():
    """Tests the clustering when constraining by surface"""

    cifti_test = './logpar/cli/tests/data/test.dconn.nii'
    l_surf = './logpar/cli/tests/data/L.white.surf.gii'
    r_surf = './logpar/cli/tests/data/R.pial.surf.gii'

    cifti = nibabel.load(cifti_test)

    header = cifti.header
    data = cifti.get_data()[0, 0, 0, 0]

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    for direction in ["COLUMN", "ROW"]:
        for constraint in [l_surf, r_surf]:
            for to_logodds in [True, False]:
                logging.info('--- Starting: {} {} {} ---'.format(direction,
                                                                 constraint,
                                                                 to_logodds))
                # Result from cifti_parcellate CLI
                cifti_parcellate.cifti_parcellate(cifti_test, output,
                                                  direction=direction,
                                                  constrained=True,
                                                  surface=constraint,
                                                  to_logodds=to_logodds)
                dendro, sxml = dendrogram_utils.load(output)

                # Result from clustering function
                test_data = data.copy()
                if direction == 'COLUMN':
                    test_data = numpy.transpose(test_data)

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
                dendro2 = our_hie.clustering(test_data, constraints=ady_matrix)

                # The results should be the same
                numpy.testing.assert_almost_equal(dendro, dendro2, 5)

                cifti_xml = cifti_utils.extract_brainmodel(header, direction,
                                                           model, struc)
                new_offset = 0
                for structure in cifti_xml:
                    structure.attrib['IndexOffset'] = str(new_offset)
                    new_offset += int(structure.attrib['IndexCount'])

                assert(len(cifti_xml) == len(sxml))
                for original, retrieved in zip(cifti_xml, sxml):
                    numpy.testing.assert_equal(xml.tostring(retrieved),
                                               xml.tostring(original))


def test_voxel_constrained_clustering():
    """Tests the clustering when constraining by voxels"""

    cifti_test = './logpar/cli/tests/data/test.dconn.nii'
    cifti = nibabel.load(cifti_test)

    header = cifti.header
    data = cifti.get_data()[0, 0, 0, 0]

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name

    direction = "COLUMN"
    for to_logodds in [True, False]:
        logging.info('--- Starting: {} ---'.format(direction))

        # Result from cifti_parcellate CLI
        cifti_parcellate.cifti_parcellate(cifti_test, output,
                                          direction=direction,
                                          constrained=True,
                                          to_logodds=to_logodds)
        dendro, sxml = dendrogram_utils.load(output)

        # Result from clustering function
        test_data = data.copy()
        if direction == 'COLUMN':
            test_data = numpy.transpose(test_data)

        model = 'CIFTI_MODEL_TYPE_VOXELS'

        bmodels = cifti_utils.extract_brainmodel(header, direction, model)

        off_idx = [cifti_utils.offset_and_indices(header,
                                                  b.attrib['ModelType'],
                                                  b.attrib['BrainStructure'],
                                                  direction)
                   for b in bmodels]
        offset, indices = zip(*off_idx)
        indices = [v for i in indices for v in i]
        filtered_data = numpy.zeros((len(indices), test_data.shape[1]))

        off = 0
        for offset, index in off_idx:
            lidx = len(index)
            filtered_data[off:off+lidx] = test_data[offset:offset+lidx]
        test_data = filtered_data

        ady_matrix = cifti_utils.constraint_from_voxels(header, direction,
                                                        indices)

        if to_logodds:
            test_data = transform.to_logodds(test_data, True)

        # Parcellate directly with the parcelling function
        dendro2 = our_hie.clustering(test_data, constraints=ady_matrix)

        # The results should be the same
        numpy.testing.assert_almost_equal(dendro, dendro2, 5)

        cifti_xml = cifti_utils.extract_brainmodel(header, direction,
                                                   model)
        new_offset = 0
        for structure in cifti_xml:
            structure.attrib['IndexOffset'] = str(new_offset)
            new_offset += int(structure.attrib['IndexCount'])

        # There's a voxel structure: ,add volume information
        volume_xml = cifti_utils.extract_volume(header, direction)
        cifti_xml += volume_xml

        assert(len(cifti_xml) == len(sxml))

        for original, retrieved in zip(cifti_xml, sxml):
            numpy.testing.assert_equal(xml.tostring(retrieved),
                                       xml.tostring(original))
