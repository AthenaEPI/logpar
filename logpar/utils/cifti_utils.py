''' Utils to manipulate CIFTI files '''
import os
import xml.etree.ElementTree as xml

import numpy
import nibabel

from ..constrained_ahc import mat2cond_index


CIFTI_STRUCTURES = ["CIFTI_STRUCTURE_ACCUMBENS_LEFT",
                    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT",
                    "CIFTI_STRUCTURE_ALL_WHITE_MATTER",
                    "CIFTI_STRUCTURE_ALL_GREY_MATTER",
                    "CIFTI_STRUCTURE_AMYGDALA_LEFT",
                    "CIFTI_STRUCTURE_AMYGDALA_RIGHT",
                    "CIFTI_STRUCTURE_BRAIN_STEM",
                    "CIFTI_STRUCTURE_CAUDATE_LEFT",
                    "CIFTI_STRUCTURE_CAUDATE_RIGHT",
                    "CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_LEFT",
                    "CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_RIGHT",
                    "CIFTI_STRUCTURE_CEREBELLUM",
                    "CIFTI_STRUCTURE_CEREBELLUM_LEFT",
                    "CIFTI_STRUCTURE_CEREBELLUM_RIGHT",
                    "CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_LEFT",
                    "CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_RIGHT",
                    "CIFTI_STRUCTURE_CORTEX",
                    "CIFTI_STRUCTURE_CORTEX_LEFT",
                    "CIFTI_STRUCTURE_CORTEX_RIGHT",
                    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT",
                    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT",
                    "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT",
                    "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT",
                    "CIFTI_STRUCTURE_OTHER",
                    "CIFTI_STRUCTURE_OTHER_GREY_MATTER",
                    "CIFTI_STRUCTURE_OTHER_WHITE_MATTER",
                    "CIFTI_STRUCTURE_PALLIDUM_LEFT",
                    "CIFTI_STRUCTURE_PALLIDUM_RIGHT",
                    "CIFTI_STRUCTURE_PUTAMEN_LEFT",
                    "CIFTI_STRUCTURE_PUTAMEN_RIGHT",
                    "CIFTI_STRUCTURE_THALAMUS_LEFT",
                    "CIFTI_STRUCTURE_THALAMUS_RIGHT"]


def save_cifti(filename, data, header=None, affine=None, version=2):
    ''' Simple wrapper around nibabel.save '''
    if version == 1:
        nif_image = nibabel.Nifti1Image(data, affine, header)
    else:
        nif_image = nibabel.Nifti2Image(data, affine, header)
    nibabel.save(nif_image, filename)


def load_data(filename):
    ''' Return ONLY the data from the matrix '''
    return nibabel.load(filename).get_data()


def is_model_surf(model):
    return model == 'CIFTI_MODEL_TYPE_SURFACE'


def text2voxels(text):
    voxels = text.split()
    indices = numpy.reshape(voxels, (len(voxels)/3, 3))
    return indices


def voxels2text(voxels):
    return " ".join(["{0} {1} {2}".format(x, y, z) for x, y, z in voxels])


def text2indices(text):
    return numpy.array(text.split(), dtype=int)


def indices2text(indices):
    return " ".join(map(str, indices))


def surface_attributes(cifti_header, surface, direction):
    ''' Retrieves the offset and used indices of a surface
        structure in a cifti file.

        Parameters
        ----------
        cifti_header: cifti header
            Header of the cifti file
        surface: string
            Name of surface structure.
            (CIFTI_STRUCTURE_CORTEX_LEFT or CIFTI_STRUCTURE_CORTEX_RIGHT)
        direction: string
            ROW or COLUMN

        Returns
        -------
        offset: int
            index where surface information starts in the cifti matrix
        vertices: array_like
            array with indices of the surface used in the cifti matrix
        '''
    brain_model = extract_brainmodel(cifti_header, surface, direction)[0]

    offset = int(brain_model.attrib['IndexOffset'])
    vertices = numpy.array(brain_model.find('VertexIndices').text.split(),
                           dtype=int)

    return offset, vertices


def extract_matrixIndicesMap(cifti_header, direction):
    ''' Retrieves the xml of a Matrix Indices Map from a cifti file.

       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       direction: string
           ROW or COLUMN

       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    dim = 0 if direction == 'ROW' else 1

    cxml = xml.fromstring(cifti_header.extensions[0].get_content())

    query = ".//MatrixIndicesMap[@AppliesToMatrixDimension='{}']".format(dim)
    matrix_indices_map = cxml.find(query)

    return matrix_indices_map


def extract_volume(cifti_header, direction):
    ''' Retrieves the xml of a Volume from a cifti file.

       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       direction: string
           ROW or COLUMN

       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    matrix_indices = extract_matrixIndicesMap(cifti_header, direction)
    volume_xml = matrix_indices.findall('.//Volume')
    return volume_xml


def extract_brainmodel(cifti_header, structure, direction):
    ''' Retrieves the xml of a brain model structure from a cifti file.

       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       structure: string
           Name of structure
       direction: string
           ROW or COLUMN

       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    matrix_indices = extract_matrixIndicesMap(cifti_header, direction)

    if structure == 'ALL':
        query = "./BrainModel"
        brain_model = matrix_indices.findall(query)
    else:
        query = "./BrainModel[@BrainStructure='{}']".format(structure)
        brain_model = matrix_indices.findall(query)
    return brain_model


def extract_parcel(cifti_header, name, direction):
    ''' Retrieves the xml of a parcel from a cifti file.

       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       name: string
           Name of label
       direction: string
           ROW or COLUMN

       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    matrix_indices = extract_matrixIndicesMap(cifti_header, direction)

    if name == 'ALL':
        query = "./Parcel"
        parcel = matrix_indices.findall(query)
    else:
        query = "./Parcel[@Name='{}']".format(name)
        parcel = matrix_indices.findall(query)
    return parcel


def principal_structure(gifti_obj):
    ''' Retrieves the principal structure of the gifti file.

        Parameters
        ----------
        gifti: gifti object
            gifti object from which extract name
        Returns
        -------
        name: string
            Name of the cifti structure '''
    cifti_nomenclature = {'CortexLeft':'CIFTI_STRUCTURE_CORTEX_LEFT',
                          'CortexRight':'CIFTI_STRUCTURE_CORTEX_RIGHT'}
    root = xml.fromstring(gifti_obj.to_xml())
    structure = root.find(".//*[Name='AnatomicalStructurePrimary']/Value")

    return cifti_nomenclature[structure.text]


def matrix_size(header):
    size = []
    for dire in ['ROW', 'COLUMN']:
        acum = 0

        bmodels = extract_brainmodel(header, 'ALL', dire)
        parcels = extract_parcel(header, 'ALL', dire)

        acum += len(parcels)
        acum += sum(int(b.attrib['IndexCount']) for b in bmodels)
        size.append(acum)
    return size


def cifti_filter_indices(cifti, direction, structure, indices):
    
    offset, vertices = surface_attributes(cifti.header, structure,
                                                      direction)
    return pos_in_array(indices, vertices, offset)


def cifti_filter_parcels(cifti, direction, parcels):

    extracted = extract_parcel(cifti.header, 'ALL', direction)
    extracted_names = numpy.array([p.attrib['Name'] for p in extracted])
    
    return pos_in_array(parcels, extracted_names, offset=0)


def retrieve_common_data(header, cifti_matrix):
    data = cifti_matrix.get_data()[0, 0, 0, 0]
    common_data = numpy.zeros(matrix_size(header))
    map_indices = {}
    
    for dire in ['ROW', 'COLUMN']:
        parcels = extract_parcel(header, 'ALL', dire)
        if parcels:
            names = [p.attrib['Name'] for p in parcels]
            map_indices[dire] = cifti_filter_parcels(cifti_matrix, 
                                                     dire, names)
            continue

        bmodels = extract_brainmodel(header, 'ALL', dire)
        itmp = []
        for bmodel in bmodels:
            bstr = bmodel.attrib['BrainStructure']
            btype = bmodel.attrib['ModelType']
            
            if is_model_surf(btype):
                _, indices = surface_attributes(header, bstr, dire)
                itmp += cifti_filter_indices(cifti_matrix, dire,
                                             bstr, indices).tolist()
            else:
                raise NotImplemented()

        map_indices[dire] = numpy.ravel(itmp).astype(int)

    common_data = data[map_indices['ROW'][:, None], map_indices['COLUMN']]

    return common_data


def constraint_from_surface(surface, vertices=None):
    ''' Retrieves the constraint matrix between vertices from a surface

        Parameters
        ----------
        surface : gii structure
            gii structure with triangles and edges.
        vertices : array_like (optional)
            If setted, then only the adjacency matrix regarding these
            vertices is computed

        Returns
        ------
        array_like
            A condensed adyacency matrix. The squareform can be retrieved
            using scipy.spatial.distance.squareform '''

    surf_size = len(surface.darrays[0].data)
    edges_map = numpy.zeros(surf_size) - 1

    if vertices is not None:
        nvertices = len(vertices)
        edges_map[vertices] = range(nvertices)
        neighbors = numpy.zeros(nvertices*(nvertices-1)/2, dtype=numpy.int8)
    else:
        neighbors = numpy.ones(surf_size*(surf_size-1)/2, dtype=numpy.int8)

    edges = surface.darrays[1].data

    for edge1, edge2, edge3 in edges:
        edge1, edge2 = edges_map[edge1], edges_map[edge2]
        edge3 = edges_map[edge3]
        if edge1 != -1 and edge2 != -1:
            i = mat2cond_index(nvertices, edge1, edge2)
            neighbors[i] = 1
        if edge1 != -1 and edge3 != -1:
            i = mat2cond_index(nvertices, edge1, edge3)
            neighbors[i] = 1
        if edge3 != -1 and edge2 != -1:
            i = mat2cond_index(nvertices, edge3, edge2)
            neighbors[i] = 1

    return neighbors

# --- AUX ---
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

