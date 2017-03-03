#!/usr/bin/env python
''' Function called by the cli cifti_parcellate '''
import logging

import numpy
import nibabel
from nibabel import gifti

from ..clustering import clustering
from ..utils import cifti_utils, transform, dendrogram_utils


def check_input(cifti_file, direction, constraints, min_size):
    ''' Basic input checking '''

    if cifti_file.split('.')[-2] not in ['pconn', 'dconn', 'pdconn', 'dpconn']:
        raise ValueError('The given file is not a connectivity file')

    if direction not in ['ROW', 'COLUMN']:
        raise ValueError('Direction should be either ROW or COLUMN')

    if constraints is not None and "surf.gii" != constraints[-8:]:
        raise NotImplementedError('Sorry, we only accept surface files as \
                                   constraints right now')
    if min_size < 0:
        raise ValueError("min_size MUST be greater or equal to zero")

    if constraints is None and min_size > 0:
        raise ValueError("If no constraints are given then \
                          min_size MUST be zero")


def cifti_parcellate(cifti_file, outfile, direction="ROW", to_logodds=True,
                     constraint=None, threshold=0, min_size=0, verbose=0):
    ''' Clusters the cifti file using as features the vectors in the given
        direction. Furthermore, the clustering can be constrained to happen
        only between neighbors UNTIL a minimum size is reached. In this case
        the constraints MUST come either as: a surface file, from where we will
        extract the constraints, or as volume with a mask.
            1. A surface file: only vectors asociated to that surface
               will be clustered. Neighboring constraints will we derived from
               the vertices of the surface
            2. If a volume file is used, only the vectors associated to
               voxels with a value greater than zero are used. Neighboring
               constraints will be derived from the location of each voxel

        Parameters
        ----------
        cifti_file: string
            Connectivity file in CIFTI format
        outfile: string
            File where to output the dendrogram. The file SHOULD have a csv
            extension, if not, the extension is appended to the file name.
        direction: string (optional)
            Direction to parcellate from the CIFTI file. Default: 'ROW'
        to_logodds: bool (optional)
            If true, features are transformed using the logit function.
            Default: True
        constraint: cifti surface (optional)
            Either a surface or a volume file. Default: None
        threshold: float (optional)
            Thresolhold to apply to connectivity vectors before clustering.
            Default: 0
        min_size: int (optional)
            Minimum size for the resulting clusters. Default: 0

        Returns
        -------
        None
            One or more files are created '''
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    check_input(cifti_file, direction, constraint, min_size)

    cifti = nibabel.load(cifti_file)
    features = cifti.get_data()[0, 0, 0, 0]

    if direction == 'COLUMN':
        features = numpy.transpose(features)

    if constraint is not None:
        constraint = gifti.read(constraint)
        structure = cifti_utils.principal_structure(constraint)

        # Get the indices in which the structure-features are
        offset, indices = cifti_utils.surface_attributes(cifti.header,
                                                         structure,
                                                         direction)
        features = features[offset:offset+len(indices)]
        logging.debug("structure: {}, direction: {}, offset: {},\
                       len(indices): {}, shape:{}".format(structure, direction,
                                                          offset, len(indices),
                                                          features.shape))

    logging.info("Shape of features: {}".format(features.shape))
    logging.info("Nonzero rows: {}".format((features.sum(1) > 0).sum()))

    if threshold > 0:
        logging.info("Applying Threshold({})".format(threshold))
        features[features < threshold] = 0

    logging.info("Discarding empty rows/columns of features")
    nzr_rows = (features.sum(1)).nonzero()[0]
    nzr_cols = (features.sum(0)).nonzero()[0]
    logging.info("Number of nzr rows/cols: {} {}".format(len(nzr_rows),
                                                         len(nzr_cols)))
    features = features[nzr_rows[:, None], nzr_cols]

    logging.info("New shape: {}".format(features.shape))

    if to_logodds:
        vmin, vmax = features.min(), features.max()
        if vmin < 0 or vmax > 1:
            raise ValueError('Matrix values MUST be in the range [0,1] to \
                              use logodds transform')

        logging.info("Sending to LogOdds")
        stp = 5000
        for i in xrange(0, features.shape[0], stp):
            # Send the vectors to logodds space and traslate them to mantain
            # sparsity. We can do this because the Euclidean distance is
            # invariant respect to traslations
            logging.info(i)
            features[i:i+stp] = transform.to_logodds(features[i:i+stp],
                                                     traslate=True)

    nbr_nzr_rows = len((features.sum(1)).nonzero()[0])
    nbr_nzr_cols = len((features.sum(0)).nonzero()[0])
    logging.info("Number of nzr rows/cols: {} {}".format(nbr_nzr_rows,
                                                         nbr_nzr_cols))
    ady_matrix = None
    if constraint is not None:
        # Lets calculate the adyacency matrix from the surface
        indices = indices[nzr_rows]  # Throw empty tractograms
        ady_matrix = cifti_utils.constraint_from_surface(constraint, indices)

    logging.debug("Min, Max in matrix: {}, {}".format(features.min(),
                                                      features.max()))
    # Let's cluster
    dendro = clustering(features, method='ward', constraints=ady_matrix,
                        min_size=min_size, copy=0)
    # And save in disk
    xml_structures = cifti_utils.extract_brainmodel(cifti.header,
                                                    'ALL',
                                                    direction)
    if constraint is not None:
        xml_structures = cifti_utils.extract_brainmodel(cifti.header,
                                                        structure,
                                                        direction)
    new_offset = 0
    vstruct = False
    for structure in xml_structures:

        if (structure.attrib['ModelType'] == 'CIFTI_MODEL_TYPE_SURFACE'):

            _, indices = cifti_utils.surface_attributes(cifti.header,
                                                        structure.attrib['BrainStructure'],
                                                        direction)
            indices = indices[nzr_rows[:len(indices)]]
            structure[0].text = cifti_utils.indices2text(indices)
            new_count = len(indices)
        else:
            #TODO: Implement this function and correct this else
            #_, indices = cifti_utils.voxels_attributes(cifti.header,
            #                                           structure.attrib['BrainStructure']
            #                                           direction)
            vstruct = True
            new_count = int(structure.attrib['IndexCount'])
        structure.attrib['IndexOffset'] = str(new_offset)
        structure.attrib['IndexCount'] = str(new_count)
        new_offset += new_count

    if vstruct > 0:
        # There's a voxel structure, we need to add volume information
        volume_xml = cifti_utils.extract_volume(cifti.header, direction)
        xml_structures += volume_xml

    dendrogram_utils.save(outfile, dendro, xml_structures=xml_structures)
