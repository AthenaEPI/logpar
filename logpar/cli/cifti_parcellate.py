#!/usr/bin/env python
''' Function called by the cli cifti_parcellate '''
import logging

import numpy as np
import citrix
import nimesh

from ..clustering import clustering
from ..utils import transform, dendrogram


def check_input(cifti_file, direction, constraints, min_size):
    ''' Basic input checking '''

    if cifti_file.split('.')[-2] not in ['pconn', 'dconn', 'pdconn', 'dpconn']:
        raise ValueError('The given file is not a connectivity file')

    if direction not in ['ROW', 'COLUMN']:
        raise ValueError('Direction should be either ROW or COLUMN')

    if constraints is not None and "surf.gii" != constraints[-8:]:
        raise NotImplementedError('Sorry, we only accept surf.gii files as \
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

    cifti = citrix.load(cifti_file)
    features = cifti.get_data()

    brainmodel = list(cifti.row.brain_models)
    if direction == 'COLUMN':
        features = np.transpose(features)
        brainmodel = list(cifti.column.brain_models)

    if len(brainmodel) > 1:
        raise ValueError('Multiple brain models in the direction to parcel')

    brainmodel = brainmodel[0]

    if threshold > 0:
        logging.info("Applying Threshold({})".format(threshold))
        features[features < threshold] = 0

    if to_logodds:
        vmin, vmax = features.min(), features.max()
        if vmin < 0 or vmax > 1:
            raise ValueError('Matrix values MUST be in the range [0,1] to \
                              use logodds transform')

        logging.info("Sending to LogOdds")
        stp = 5000
        for i in range(0, features.shape[0], stp):
            # Send the vectors to logodds space and traslate them to mantain
            # sparsity. We can do this because the Euclidean distance is
            # invariant respect to traslations
            logging.info(i)
            features[i:i+stp] = transform.to_logodds(features[i:i+stp],
                                                     traslate=True)

    ady_matrix = None
    if constraint is not None:
        model_type = brainmodel.model_type

        if model_type == citrix.models.VOXEL:
            raise NotImplementedError()
        elif model_type == citrix.models.SURFACE:
            used_indices = np.array(brainmodel.vertex_indices)
            ady_matrix = nimesh.io.load(constraint).adjacency_matrix.todense()
            ady_matrix = ady_matrix[used_indices[:, None], used_indices]

    # Let's cluster
    dendro = clustering(features, method='ward', constraints=ady_matrix,
                        min_size=min_size, copy=0)

    # And save in disk
    dendrogram.save(outfile, dendro, [brainmodel])
