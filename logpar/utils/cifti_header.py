''' Functions to operate with CIFTI HEADERS '''
import os

import numpy
import nibabel

import xml.etree.ElementTree as xml
from logpar.utils import cifti_utils


def header_intersection(header1, header2):
    ''' Retrieves a new CIFTI header, which contains the structures
        present in both input headers '''
    xml_header = xml.fromstring(header2.extensions[0].get_content())

    for dire in ['ROW', 'COLUMN']:
        new_bmodels = cifti_utils.extract_brainmodel(header1, 'ALL', dire)
        new_strucs = set(b.attrib['BrainStructure'] for b in new_bmodels)

        new_parcels = cifti_utils.extract_parcel(header1, 'ALL', dire)
        new_parcels = set(p.attrib['Name'] for p in new_parcels)

        idx = 0 if dire == 'ROW' else 1
        query = ".//MatrixIndicesMap[@AppliesToMatrixDimension='{}']".format(idx)
        mimap = xml_header.find(query)
        
        # If this direction has labels, we keep only the labels present
        # in both headers
        parcels = mimap.findall('Parcels')
        for parcel in parcels:
            if parcel.attrib['Name'] not in new_parcels:
                mimap.remove(parcel)  # Remove it
        
        # If the direction has BrainModels, we keep only the BM present
        # in both headers. Moreover, we keep only the indices they share
        bmodels = mimap.findall('BrainModel')
        for bmodel in bmodels:
            bstr = bmodel.attrib['BrainStructure']
            btype = bmodel.attrib['ModelType']

            if bstr not in new_strucs:
                mimap.remove(bmodel)  # Not present: remove it

            if cifti_utils.is_model_surf(btype):
                # It's a surface, lets update its indices
                _, new_vertices = cifti_utils.surface_attributes(header1, 
                                                                 bstr,
                                                                 dire)
                _, vertices = cifti_utils.surface_attributes(header2,
                                                             bstr, dire)
                common = sorted(set(vertices).intersection(new_vertices))
                common_txt = cifti_utils.indices2text(common)
                bmodel.find('VertexIndices').text = common_txt
                bmodel.attrib['IndexCount'] = str(len(common))
            else:
                # It's a volume, lets update its voxels
                raise NotImplemented()
        
        # Finally, fix the attributes of each brain model
        offset = 0
        for bmodel in bmodels:
            bmodel.attrib['IndexOffset'] = str(offset)
            offset += int(bmodel.attrib['IndexCount'])

        new_xml_string = xml.tostring(xml_header)
        new_extension = nibabel.nifti1.Nifti1Extension(32, new_xml_string)
        
        header2.extensions[0] = new_extension

        return header2


def soft_colors_xml_label_map(nlabels):
    ''' Returns an xml Element which represents a GiftiLabelTable
        with soft colors

        Parameters
        ----------
        nlabels: int
            Number of colors to represent in the table

        Returns
        -------
        xml Element
            XML CIFTI LabelTable'''
    named_map = xml.Element('NamedMap')
    map_name = xml.SubElement(named_map, 'MapName')
    map_name.text = 'Parcel'

    label_table = xml.SubElement(named_map, 'LabelTable')
    colors_path = os.path.dirname(os.path.abspath(__file__))
    colors = numpy.loadtxt(os.path.join(colors_path, 'colors.txt'))
    ncolors = len(colors)

    background = xml.SubElement(label_table, 'Label',
                                attrib={'Alpha':'0', 'Blue':'0', 'Red':'0',
                                        'Green':'0', 'Key':'0'})
    background.text = '???'

    for key in range(1, nlabels+1):
        red, green, blue = colors[(key-1)%ncolors]
        if red == green and green == blue:
            # remove this gray
            red, blue = 0.7*red, 0.95*blue
        label = xml.SubElement(label_table, 'Label',
                               attrib={'Alpha':'1', 'Blue':str(blue),
                                       'Red':str(red), 'Green':str(green),
                                       'Key':str(key)})
        label.text = str(key)
    return named_map


def create_label_header(xml_structures, nparcels):
    ''' Creates a label header for different structures. Right now this is
        a draft. TODO: Extend to work with many structures at the same time '''
    cifti_extension = xml.Element('CIFTI', {'Version': '2'})

    matrix = xml.SubElement(cifti_extension, 'Matrix')

    LABELS = 'CIFTI_INDEX_TYPE_LABELS'
    BRAIN_MODEL = 'CIFTI_INDEX_TYPE_BRAIN_MODELS'

    # First dimention: LABEL
    mat_indx_map_0 = xml.SubElement(matrix, 'MatrixIndicesMap',
                                    {'AppliesToMatrixDimension': '0',
                                     'IndicesMapToDataType': LABELS})

    mat_indx_map_0.insert(0, soft_colors_xml_label_map(nparcels))

    # Second dimention: what the columns represents.
    mat_indx_map_1 = xml.SubElement(matrix, 'MatrixIndicesMap',
                                    {'AppliesToMatrixDimension': '1',
                                     'IndicesMapToDataType': BRAIN_MODEL})

    for i, structure in enumerate(xml_structures):
        mat_indx_map_1.insert(i, structure)

    cifti_header = nibabel.nifti2.Nifti2Header()

    cifti_header.extensions.append(
        nibabel.nifti1.Nifti1Extension(32, xml.tostring(cifti_extension)))

    return cifti_header
