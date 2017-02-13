''' Extract a parcellation from a dendrogram '''
import logging

import numpy
from scipy.cluster.hierarchy import fcluster

from ..utils import cifti_utils, dendrogram_utils


def extract_parcellation(dendrogram_file, nparcels,
                         txtout=None, labelout=None):
    ''' Extracts a parcellation with a predefined number of parcels. If
        setted, also writes a cifti label file

        Parameters
        ----------
        dendrogram_file: string
            File with the dendrogram, in csv format
        nparcels: int
            Numbers of parcels the extracted parcellation should have
        txtout: string (optional)
            File where to write the labels as plane text
        indices_file: string (optional)
            File from where to extract information of the structure to use in
            the cifti label file
        labelout: string (optional)
            File where to write the CIFTI label file

        Returns
        -------
        None
            One or more files a created '''
    dendrogram, xml_structures = dendrogram_utils.load(dendrogram_file)

    heights = sorted(numpy.unique(dendrogram[:, 2]))
    #print len(heights), 2-nparcels
    # If WARD was used, there's a direct mapping between the number of
    # parcels and the position of the height in the tree
    #heights = heights[2-nparcels] + heights
    logging.debug("len(heights): {}".format(len(heights)))
    for height in heights:
        parcellation = fcluster(dendrogram, height, criterion='distance')
        parcellation_size = parcellation.max()

        logging.debug('parcellation_size: {}'.format(parcellation_size))
        if parcellation_size == nparcels:
            break

    # Save the parcellation
    if txtout is not None:
        numpy.savetxt(txtout, parcellation, delimiter=',')

    if labelout is not None:
        cifti_label_header = cifti_utils.label_header(xml_structures, nparcels)

        cifti_utils.save_cifti(labelout,
                               parcellation[None, None, None, None, None, ...],
                               header=cifti_label_header)
