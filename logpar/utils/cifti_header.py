''' Functions to operate with CIFTI HEADERS '''
import os

import numpy
import nibabel

import xml.etree.ElementTree as xml
from logpar.utils import cifti_utils


def header_intersection(header1, header2):
    ''' Crates a new CIFTI header, which contains only the structures/indices
        present in both headers at the same time '''
    # We will base the new xml on the xml from header 2
    xml_new = cifti_utils.extract_xml_header(header2)

    for direction in ['ROW', 'COLUMN']:
        bmodels_h1 = cifti_utils.extract_brainmodel(header1, direction)
        structs_h1 = set(b.attrib['BrainStructure'] for b in bmodels_h1)

        parcels_h1 = cifti_utils.extract_parcel(header1, direction)
        parcels_h1 = set(p.attrib['Name'] for p in parcels_h1)

        # I want to modify the MatrixIndicesMap of xml_h2, therefore, I
        # don't have other choise but to use xml.find
        dim = cifti_utils.direction2dimention(direction)
        mims = xml_new.findall(".//MatrixIndicesMap")
        for mim in mims:
            if str(dim) in mim.attrib['AppliesToMatrixDimension'].split(','):
                mim_new = mim

        # If this direction has labels, we keep only the labels present
        # in both headers
        parcels = mim_new.findall('.//Parcel')
        for parcel in parcels:
            if parcel.attrib['Name'] not in parcels_h1:
                mim_new.remove(parcel)  # Remove i

        # If the direction has BrainModels, we keep only the BM present
        # in both headers. Moreover, we keep only the indices they share
        bmodels = mim_new.findall('.//BrainModel')
        for bmodel in bmodels:
            bstr = bmodel.attrib['BrainStructure']
            btype = bmodel.attrib['ModelType']

            if bstr not in structs_h1:
                mim_new.remove(bmodel)  # Not present: remove it
                continue

            if cifti_utils.is_model_surf(btype):
                # It's a surface, lets update its indices
                _, vertices1 = cifti_utils.offset_and_indices(header1, btype,
                                                              bstr, direction)
                _, vertices2 = cifti_utils.offset_and_indices(header2, btype,
                                                              bstr, direction)
                common = sorted(set(vertices1).intersection(vertices2))
                common_txt = cifti_utils.indices2text(common)
                bmodel.find('VertexIndices').text = common_txt
                bmodel.attrib['IndexCount'] = str(len(common))
            else:
                # It's a volume, lets update its voxels
                raise NotImplementedError()

        # Finally, fix the attributes of each brain model
        offset = 0
        bmodels = mim_new.findall('.//BrainModel')
        for bmodel in bmodels:
            bmodel.attrib['IndexOffset'] = str(offset)
            offset += int(bmodel.attrib['IndexCount'])

    xml_string = xml.tostring(xml_new)
    header_new = header2.copy()
    header_new.extensions[0] = nibabel.nifti1.Nifti1Extension(32, xml_string)

    return header_new


def header_union(header1, header2):
    ''' Retrieves a new CIFTI header, which contains both the
        structures/indices of headers 1 and 2 '''
    # We will base the new xml on the xml from header 1
    xml_new = cifti_utils.extract_xml_header(header1)

    for direction in ['ROW', 'COLUMN']:
        bmodels_h1 = cifti_utils.extract_brainmodel(header1, direction)
        structs_h1 = set(b.attrib['BrainStructure'] for b in bmodels_h1)

        parcels_h1 = cifti_utils.extract_parcel(header1, direction)
        parcels_h1 = set(p.attrib['Name'] for p in parcels_h1)

        # I want to modify the MatrixIndicesMap of xml_new, therefore, I
        # don't have other choise but to use xml.find
        dim = cifti_utils.direction2dimention(direction)
        mims = xml_new.findall(".//MatrixIndicesMap")
        for mim in mims:
            if str(dim) in mim.attrib['AppliesToMatrixDimension'].split(','):
                mim_new = mim

        # If this direction has labels, we keep only the labels present
        # in both headers
        parcels = cifti_utils.extract_parcel(header2, direction)
        for parcel in parcels:
            if parcel.attrib['Name'] not in parcels_h1:
                mim_new.append(parcel)  # Add it

        # If the direction has BrainModels, we unite the BM present
        # in both headers. Moreover, we keep all the used indices
        bmodels = cifti_utils.extract_brainmodel(header2, direction)
        for bmodel in bmodels:
            bstr = bmodel.attrib['BrainStructure']
            btype = bmodel.attrib['ModelType']

            if bstr not in structs_h1:
                mim_new.append(bmodel)  # Not present: add it
                continue

            if cifti_utils.is_model_surf(btype):
                # It's a present surface, lets unite the indices
                _, vertices1 = cifti_utils.offset_and_indices(header1, btype,
                                                              bstr, direction)
                _, vertices2 = cifti_utils.offset_and_indices(header2, btype,
                                                              bstr, direction)
                common = sorted(set(vertices1).union(vertices2))
                common_txt = cifti_utils.indices2text(common)

                # Update structure in xml_new
                query = ".//BrainModel[@ModelType='{}'][@BrainStructure='{}']"
                bmodel = mim_new.find(query.format(btype, bstr))

                bmodel.find('VertexIndices').text = common_txt
                bmodel.attrib['IndexCount'] = str(len(common))
            else:
                # It's a volume, lets update its voxels
                raise NotImplementedError()

        # Finally, fix the attributes of each brain model
        offset = 0
        bmodels = mim_new.findall('.//BrainModel')
        for bmodel in bmodels:
            bmodel.attrib['IndexOffset'] = str(offset)
            offset += int(bmodel.attrib['IndexCount'])

    xml_string = xml.tostring(xml_new)
    header_new = header1.copy()
    header_new.extensions[0] = nibabel.nifti1.Nifti1Extension(32, xml_string)

    return header_new


def soft_colors_xml_label_map(nlabels, offset=0):
    ''' Returns an xml Element which represents a GiftiLabelTable
        with soft colors

        Parameters
        ----------
        nlabels: int
            Number of colors to represent in the table
        offset: int
            Number from where to start counting the labels

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
                                       'Key':str(key+offset)})
        label.text = str(key+offset)
    return named_map


def create_label_header(xml_structures, nparcels, offset=0):
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

    mat_indx_map_0.insert(0, soft_colors_xml_label_map(nparcels, offset))

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


def volume_xml(dimention, affine):
    """ Creates the xml of a volume """
    affine = " ".join(map(str, affine.reshape(-1)))
    dimention = ",".join(map(str, dimention[:3]))

    volume = xml.Element('Volume', {'VolumeDimensions':dimention})
    transform = xml.SubElement(volume,
                               'TransformationMatrixVoxelIndicesIJKtoXYZ',
                               {'MeterExponent':'-3'})
    transform.text = affine
    return volume


def create_conn_header(row_structures, col_structures, row_dimention=None,
                       col_dimention=None, row_affine=None, col_affine=None):
    ''' Creates the header for a dconn matrix '''

    cifti_extension = xml.Element('CIFTI', {'Version': '2'})

    matrix = xml.SubElement(cifti_extension, 'Matrix')

    BRAIN_MODEL = 'CIFTI_INDEX_TYPE_BRAIN_MODELS'

    # First dimention: ROW
    mat_indx_map_0 = xml.SubElement(matrix, 'MatrixIndicesMap',
                                    {'AppliesToMatrixDimension': '0',
                                     'IndicesMapToDataType': BRAIN_MODEL})
    if row_dimention is not None:
        mat_indx_map_0.insert(0, volume_xml(row_dimention, row_affine))

    for i, structure in enumerate(row_structures):
        mat_indx_map_0.insert(i, structure)

    # Second dimention: what the columns represents.
    mat_indx_map_1 = xml.SubElement(matrix, 'MatrixIndicesMap',
                                    {'AppliesToMatrixDimension': '1',
                                     'IndicesMapToDataType': BRAIN_MODEL})
    if col_dimention is not None:
        mat_indx_map_1.insert(0, volume_xml(col_dimention, col_affine))

    for i, structure in enumerate(col_structures):
        mat_indx_map_1.insert(i, structure)

    cifti_header = nibabel.nifti2.Nifti2Header()

    cifti_header.extensions.append(
        nibabel.nifti1.Nifti1Extension(32, xml.tostring(cifti_extension)))

    return cifti_header


def brain_model_xml(mtype, name, coord, offset, size):
    name, mtype, offset, size = map(str, [name, mtype, offset, size])

    brain_model = xml.Element('BrainModel', attrib={'IndexOffset':offset,
                                                    'IndexCount':str(len(coord)),
                                                    'ModelType':mtype,
                                                    'BrainStructure':name})
    if mtype == "CIFTI_MODEL_TYPE_VOXELS":
        voxijk = xml.SubElement(brain_model, 'VoxelIndicesIJK')
        voxijk.text = " ".join(["{} {} {}".format(x,y,z) for x, y, z in coord])
    else:
        brain_model.attrib['SurfaceNumberOfVertices'] = size
        vertx = xml.SubElement(brain_model, 'VertexIndices')
        vertx.text = " ".join(map(str, coord))

    return brain_model
