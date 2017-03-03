''' Functions to operate with CIFTI HEADERS '''
import numpy
import nibabel

import xml.etree.ElementTree as xml
from logpar.utils import cifti_utils


def pos_in_array(arr1, arr2, offset):
    ''' Returns in which position of arr2 is each element of arr1.
        If the element is not found, returns the position -1 '''
    pos_indice = numpy.zeros_like(arr1, dtype=int)

    for i, elem in enumerate(arr1):
        pos_in_arr2 = (arr2==elem).nonzero()[0]
        if pos_in_arr2:
            pos_indice[i] = pos_in_arr2 + offset
        else:
            pos_indice[i] = -1
    return pos_indice


def cifti_filter_indices(cifti, direction, structure, indices):
    
    offset, vertices = cifti_utils.surface_attributes(cifti.header, structure,
                                                      direction)
    return pos_in_array(indices, vertices, offset)


def cifti_filter_parcels(cifti, direction, parcels):

    extracted = cifti_utils.extract_parcel(cifti.header, 'ALL', direction)
    extracted_names = numpy.array([p.attrib['Name'] for p in extracted])
    
    return pos_in_array(parcels, extracted_names, offset=0)


class CiftiMinimumCommonHeader():

    def __init__(self):
        self._header = None

    def intersect_header(self, header):
        if self._header is None:
            self._header = header
        else:
            self.update_header(header)
    
    def update_header(self, header):
        # We need to take the xml_header, modify ir and save it again
        xml_header = xml.fromstring(self._header.extensions[0].get_content())

        for dire in ['ROW', 'COLUMN']:
            new_bmodels = cifti_utils.extract_brainmodel(header, 'ALL', dire)
            new_strucs = set(b.attrib['BrainStructure'] for b in new_bmodels)

            new_parcels = cifti_utils.extract_parcel(header, 'ALL', dire)
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
                    _, new_vertices = cifti_utils.surface_attributes(header, 
                                                                     bstr,
                                                                     dire)
                    _, vertices = cifti_utils.surface_attributes(self._header,
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
            self._header.extensions[0] = new_extension


    def get_matrix_size(self):
        size = []
        for dire in ['ROW', 'COLUMN']:
            acum = 0

            bmodels = cifti_utils.extract_brainmodel(self._header, 'ALL', dire)
            parcels = cifti_utils.extract_parcel(self._header, 'ALL', dire)

            acum += len(parcels)
            acum += sum(int(b.attrib['IndexCount']) for b in bmodels)
            size.append(acum)
        return size


    def extract_common_struc(self, cifti_matrix):
                
        data = cifti_matrix.get_data()[0, 0, 0, 0]
        common_data = numpy.zeros(self.get_matrix_size())
        map_indices = {}
        
        for dire in ['ROW', 'COLUMN']:
            parcels = cifti_utils.extract_parcel(self._header, 'ALL', dire)
            if parcels:
                names = [p.attrib['Name'] for p in parcels]
                map_indices[dire] = cifti_filter_parcels(cifti_matrix, 
                                                         dire, names)
                continue

            bmodels = cifti_utils.extract_brainmodel(self._header, 'ALL', dire)
            itmp = []
            for bmodel in bmodels:
                bstr = bmodel.attrib['BrainStructure']
                btype = bmodel.attrib['ModelType']
                
                if cifti_utils.is_model_surf(btype):
                    _, indices = cifti_utils.surface_attributes(self._header,
                                                                bstr, dire)
                    itmp += cifti_filter_indices(cifti_matrix, dire,
                                                 bstr, indices).tolist()
                else:
                    raise NotImplemented()

            map_indices[dire] = numpy.ravel(itmp).astype(int)

        common_data = data[map_indices['ROW'][:, None], map_indices['COLUMN']]

        return common_data
