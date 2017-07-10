''' Utils to manipulate CIFTI files '''
import itertools
import xml.etree.ElementTree as xml
from collections import defaultdict

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


def save_nifti(filename, data, header=None, affine=None, version=2):
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
    ''' Returns true if 'model' if of type surface '''
    return model == 'CIFTI_MODEL_TYPE_SURFACE'


def text2voxels(text):
    vxs = map(int, text.split())
    indices = [(vxs[i], vxs[i+1], vxs[i+2]) for i in xrange(0, len(vxs), 3)]
    return indices


def voxels2text(voxels):
    return " ".join(["{0} {1} {2}".format(x, y, z) for x, y, z in voxels])


def text2indices(text):
    return map(int, text.split())


def indices2text(indices):
    return " ".join(map(str, indices))


def direction2dimention(direction):
    return 0 if direction == 'ROW' else 1


def modeltext2indices(text, model):
    if model == 'CIFTI_MODEL_TYPE_VOXELS':
        return text2voxels(text)
    else:
        return text2indices(text)


def indices2modeltext(indices, model):
    if model == 'CIFTI_MODEL_TYPE_VOXELS':
        return voxels2text(indices)
    else:
        return indices2text(indices)


def offset_and_indices(cifti_header, modeltype, structure, direction):
    ''' Retrieves the offset and used indices of a brainmodel in a cifti file

        Parameters
        ----------
        cifti_header: cifti header
            Header of the cifti file
        modeltype: string
            Name of the CIFTI MODELTYPE to retrieve
        structure:
            Name of the CIFTI structure to retrieve
        direction: string
            ROW or COLUMN

        Returns
        -------
        offset: int
            index where surface information starts in the cifti matrix
        indices: array_like
            array with indices of the surface used in the cifti matrix
        '''
    brain_model = extract_brainmodel(cifti_header, direction,
                                     modeltype, structure)
    if brain_model == []:
        raise ValueError('BrainModel not found')
    brain_model = brain_model[0]

    offset = int(brain_model.attrib['IndexOffset'])
    indices = modeltext2indices(brain_model[0].text, modeltype)

    return offset, indices


def volume_attributes(cifti_header, direction):
    ''' Retrieves the dimention and affine of the volume defined in a given
        direction.

        Parameters
        ----------
        cifti_header: cifti header
            Header of the cifti file
        direction: string
            ROW or COLUMN

        Returns
        -------
        dimention: array_like
            a 3-d array representing the dimention of the volume
        affine: array_like
            affine matrix defined in the volume
        '''
    matrix_indices_map = extract_matrixindicesmap(cifti_header, direction)

    volume = matrix_indices_map.find('.//Volume')
    if volume is None:
        return None, None

    dimentions = map(int, volume.attrib['VolumeDimensions'].split(','))
    affine = numpy.array(volume[0].text.split(), dtype=float).reshape((4, 4))

    return dimentions, affine


def extract_xml_header(cifti_header):
    ''' Retrieves the xml header from the extension of a cifti header.

       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file

       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    cxml = xml.fromstring(cifti_header.extensions[0].get_content())

    return cxml


def extract_matrixindicesmap(cifti_header, direction):
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
    dim = direction2dimention(direction)
    cxml = extract_xml_header(cifti_header)

    query = ".//MatrixIndicesMap"
    mims = cxml.findall(".//MatrixIndicesMap")
    
    for mim in mims:
        if dim in map(int, mim.attrib['AppliesToMatrixDimension'].split(',')):
            return mim


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
    matrix_indices = extract_matrixindicesmap(cifti_header, direction)
    volume_xml = matrix_indices.findall('.//Volume')
    return volume_xml


def extract_brainmodel(cifti_header, direction,
                       modeltype=None, structure=None):
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
    matrix_indices = extract_matrixindicesmap(cifti_header, direction)

    query = "./BrainModel"
    if structure is not None:
        query += "[@BrainStructure='{}']".format(structure)
    if modeltype is not None:
        query += "[@ModelType='{}']".format(modeltype)
    brain_model = matrix_indices.findall(query)

    return brain_model


def extract_parcel(cifti_header, direction, name=None):
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
    matrix_indices = extract_matrixindicesmap(cifti_header, direction)

    query = ".//Parcel"
    if name is not None:
        query += "[@Name='{}']".format(name)
    parcel = matrix_indices.findall(query)
    return parcel


def extract_label(cifti_header, direction, key=None):
    ''' Retrieves the xml of a label from a cifti file.

       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       direction: string
           ROW or COLUMN
       key: string
           Key of label

       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    matrix_indices = extract_matrixindicesmap(cifti_header, direction)

    query = ".//Label"
    if key is not None:
        query += "[@Key='{}']".format(key)
    label = matrix_indices.findall(query)
    return label


def get_feature_type(cifti_header, direction):
    """ Returns either VOXELS, SURFACE or MIXED, taking into
        account the ModelType values in the direction"""
    brainmodels = extract_brainmodel(cifti_header, direction)

    modeltypes = set(b.attrib['ModelType'] for b in brainmodels)

    return 'CIFTI_MODEL_TYPE_MIXED' if len(modeltypes) > 1 else next(modeltypes)


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

        bmodels = extract_brainmodel(header, dire)
        parcels = extract_parcel(header, dire)

        acum += len(parcels)
        acum += sum(int(b.attrib['IndexCount']) for b in bmodels)
        size.append(acum)
    return size


def cifti_filter_indices(cifti_header, direction, modeltype,
                         structure, idx_search):
    offset, indices = offset_and_indices(cifti_header, modeltype,
                                         structure, direction)
    return pos_in_array(idx_search, indices, offset)


def cifti_filter_parcels(cifti_header, direction, parcels):
    extracted = extract_parcel(cifti_header, direction)
    extracted_names = numpy.array([p.attrib['Name'] for p in extracted])

    return pos_in_array(parcels, extracted_names, offset=0)


def retrieve_common_data(header, cifti_matrix):
    ''' Retrieves data in cifti_matrix from the structures in header '''
    data = cifti_matrix.get_data()[0, 0, 0, 0]
    common_data = numpy.zeros(matrix_size(header))
    map_indices = {}

    for dire in ['ROW', 'COLUMN']:
        parcels = extract_parcel(header, dire)
        if parcels:
            names = [p.attrib['Name'] for p in parcels]
            map_indices[dire] = cifti_filter_parcels(cifti_matrix,
                                                     dire, names)
            continue

        bmodels = extract_brainmodel(header, dire)
        itmp = []
        for bmodel in bmodels:
            bstr = bmodel.attrib['BrainStructure']
            btype = bmodel.attrib['ModelType']

            if is_model_surf(btype):
                _, indices = offset_and_indices(header, btype, bstr, dire)
                itmp += cifti_filter_indices(cifti_matrix, dire, btype,
                                             bstr, indices).tolist()
            else:
                raise NotImplementedError()

        map_indices[dire] = numpy.ravel(itmp).astype(int)

    common_data = data[map_indices['ROW'][:, None], map_indices['COLUMN']]

    return common_data


def constraint_from_voxels(cifti_header, direction, vertices=None):
    ''' Retrieves the adyacency matrix between voxels '''
    brainmodels = extract_brainmodel(cifti_header, direction)

    def get_voxels(brainmodel):
        structure = brainmodel.attrib['BrainStructure']
        _, voxels = offset_and_indices(cifti_header, 'CIFTI_MODEL_TYPE_VOXELS',
                                       structure, direction)
        return voxels

    # Retrieve the voxels for each brainmodel in the specified direction
    voxels = [get_voxels(b) for b in brainmodels]

    if vertices is not None:
        # Filter voxels
        voxels = [vx for i, vx in enumerate(voxels) if i in vertices]

    def are_neighbors(vox_i, vox_j):
        return next((False for i, j in zip(vox_i, vox_j) if abs(i-j) > 1), True)

    # Compute adyacency matrix assuming that combinations has the right order
    ady_matrix = [are_neighbors(*b) for b in itertools.combinations(voxels, 2)]
    ady_matrix = numpy.array(ady_matrix, dtype=numpy.int8)

    return ady_matrix


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
    elem2pos_in_arr2 = defaultdict(lambda: -1,
                                   {e:i+offset for i, e in enumerate(arr2)})
    return [elem2pos_in_arr2[e] for e in arr1]
