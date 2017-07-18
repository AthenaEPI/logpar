#!/usr/bin/env python
''' Function called by the cli cifti_parcellate '''
import itertools
import logging

import numpy
import nibabel
from nibabel import gifti

from ..clustering import clustering
from ..utils import cifti_utils, transform, dendrogram_utils


def check_input(cifti_file, direction, min_size):
    ''' Basic input checking '''

    if cifti_file.split('.')[-2] not in ['pconn', 'dconn', 'pdconn', 'dpconn']:
        raise ValueError('The given file is not a connectivity file')

    if direction not in ['ROW', 'COLUMN']:
        raise ValueError('Direction should be either ROW or COLUMN')

    if min_size < 0:
        raise ValueError("min_size MUST be greater or equal to zero")


def cifti_parcellate(cifti_file, outfile, direction="ROW", to_logodds=True,
                     constrained=False, surface=None, threshold=0, min_size=0,
                     verbose=0):
    ''' Clusters the cifti file using as features the vectors in the given
        direction. Furthermore, the clustering can be constrained to happen
        only between neighbors UNTIL a minimum size is reached. If the
        matrix possess a surface model type, then a surface must be given in
        order to apply a constrained clustering. For the voxel model type, the
        constrains will be derived from the cifti matrix.

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
        constrained: bool (optional)
            If True, the clustering happens only between neighbors, until a 
            minimum parcel size is reached. Default: False
        surface: gifti-file (optional)
            The surface from which to extract the constraint matrix
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

    check_input(cifti_file, direction, min_size)

    cifti = nibabel.load(cifti_file)
    features = cifti.get_data()[0, 0, 0, 0]

    if direction == 'COLUMN':
        features = numpy.transpose(features)
    
    modeltype, structure = None, None
    if constrained:
        if surface:
            constraint = gifti.read(surface)
            modeltype = cifti_utils.SURFACE
            structure = cifti_utils.principal_structure(constraint)

            # Get the indices in which the structure-features are
            offset, indices = cifti_utils.offset_and_indices(cifti.header,
                                                             modeltype,
                                                             structure,
                                                             direction)
            indices = numpy.array(indices)
            features = features[offset:offset+len(indices)]
        else:
            modeltype = cifti_utils.VOXEL
            bmodels = cifti_utils.extract_brainmodel(cifti.header, direction,
                                                     modeltype)
            offset_and_indices = []
            for bmodel in bmodels:
                structure = bmodel.attrib['BrainStructure']
                offset_and_indices.append(
                    cifti_utils.offset_and_indices(cifti.header, modeltype,
                                                   structure, direction)
                    )
            
            filtered = numpy.zeros((sum([off for off, _ in offset_and_indices]),
                                   features.shape[1]))
            
            off = 0
            for offset, indices in offset_and_indices:
                lindices = len(indices)
                filtered[off:off+lindices] = features[offset:offset+lindices]
                off += lindices
            
            features = filtered

    logging.info("Shape of features: {}".format(features.shape))

    if threshold > 0:
        logging.info("Applying Threshold({})".format(threshold))
        features[features < threshold] = 0

    logging.info("Discarding empty rows/columns of features")
    nzr_rows = (features.sum(1)).nonzero()[0]
    nzr_cols = (features.sum(0)).nonzero()[0]
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
            features[i:i+stp] = transform.to_logodds(features[i:i+stp],
                                                     traslate=True)

    ady_matrix = None
    if constrained:
        if surface:
            # Lets calculate the adyacency matrix from the surface
            indices = indices[nzr_rows]  # Throw empty tractograms
            ady_matrix = cifti_utils.constraint_from_surface(constraint,
                                                             indices)
        else:
            _, indices = zip(*offset_and_indices)
            indices = [v for vox in indices for v in vox]  # concat voxels
            ady_matrix = cifti_utils.constraint_from_voxels(cifti.header,
                                                            direction,
                                                            indices)
    # Let's cluster
    dendro = clustering(features, method='ward', constraints=ady_matrix,
                        min_size=min_size, copy=0)
    # And save in disk
    xml_structures = cifti_utils.extract_brainmodel(cifti.header, direction)
    if constrained:
        if surface:
            xml_structures = cifti_utils.extract_brainmodel(cifti.header,
                                                            direction,
                                                            modeltype,
                                                            structure)
        else:
            xml_structures = bmodels

    new_offset = 0
    vstruct = False
    for structure in xml_structures:
        modeltype = structure.attrib['ModelType']
        _, indices = cifti_utils.offset_and_indices(cifti.header,
                                                    modeltype,
                                                    structure.attrib['BrainStructure'],
                                                    direction)
        indices = numpy.array(indices)
        indices = indices[nzr_rows[:len(indices)]]
        
        if modeltype == cifti_utils.VOXEL:
            structure[0].text = cifti_utils.voxels2text(indices)
            vstruct = True
        else:
            structure[0].text = cifti_utils.indices2text(indices)

        new_count = len(indices)
        structure.attrib['IndexOffset'] = str(new_offset)
        structure.attrib['IndexCount'] = str(new_count)

        new_offset += new_count

    if vstruct > 0:
        # There's a voxel structure, we need to add volume information
        volume_xml = cifti_utils.extract_volume(cifti.header, direction)
        xml_structures += volume_xml

    dendrogram_utils.save(outfile, dendro, xml_structures=xml_structures)
