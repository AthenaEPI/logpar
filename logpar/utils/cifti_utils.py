''' Utils to manipulate CIFTI files '''
import os
import xml.etree.ElementTree as xml

import numpy
import nibabel

from ..constrained_ahc import mat2cond_index


def save_cifti(filename, data, header=None, affine=None, version=2):
    ''' Wrapper around nibabel.save '''
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


def label_header(xml_structures, nparcels):
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
