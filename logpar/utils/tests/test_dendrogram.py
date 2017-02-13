''' Tests for the dendrogram utils '''
import xml.etree.ElementTree as xml
from tempfile import NamedTemporaryFile

import numpy
from .. import dendrogram_utils

def test_save_and_load():
    ''' Tests the dendrogram saving function '''
    dendrogram = numpy.random.random((20, 10))

    single_structure = xml.Element('TEST', attrib={'at1':'1'})
    xml.SubElement(single_structure, 'SubElem1', attrib={'sub1':'sub1'})

    output = NamedTemporaryFile(mode='w', delete=True, suffix='.csv').name
    dendrogram_utils.save(output, dendrogram, [single_structure])

    loaded_dendro, loaded_xml = dendrogram_utils.load(output, return_xml=True)

    numpy.testing.assert_equal(loaded_dendro, dendrogram)

    print xml.tostring(loaded_xml[0])

    numpy.testing.assert_equal(xml.tostring(single_structure),
                               xml.tostring(loaded_xml[0]))

    second_structure = xml.Element('TEST_2', attrib={'at_1':'1'})
    xml.SubElement(single_structure, 'SubElem_2', attrib={'sub_2':'sub2'})

    xml_structures = [single_structure, second_structure]
    dendrogram_utils.save(output, dendrogram, xml_structures)

    loaded_dendro, loaded_xml = dendrogram_utils.load(output, return_xml=True)
    numpy.testing.assert_equal(loaded_dendro, dendrogram)

    for i, structure in enumerate(xml_structures):

        numpy.testing.assert_equal(xml.tostring(structure),
                                   xml.tostring(loaded_xml[i]))
