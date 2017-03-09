''' Test cifti_utils.py '''
import xml.etree.ElementTree as xml

import nibabel
import numpy

from .. import cifti_header, cifti_utils


def test_label_color():
    ''' Tests that a correct LabelTable is created '''

    cifti_test = nibabel.load('./logpar/cli/tests/data/test.dconn.nii')
    nlabels = 10

    header = cifti_test.header

    xml_structures = cifti_utils.extract_brainmodel(header, 'ALL', 'COLUMN')

    lt_header = cifti_header.create_label_header(xml_structures, 10)

    xml_in_labeltable = cifti_utils.extract_brainmodel(lt_header, 'ALL',
                                                       'COLUMN')

    for original, retrieved in zip(xml_structures, xml_in_labeltable):
        numpy.testing.assert_equal(original.attrib['BrainStructure'],
                                   retrieved.attrib['BrainStructure'])

    extension_xml = xml.fromstring(lt_header.extensions[0].get_content())
    labels = extension_xml.findall('.//Label')
    
    # Number of labels + '???'background
    numpy.testing.assert_equal(len(labels), nlabels+1)
