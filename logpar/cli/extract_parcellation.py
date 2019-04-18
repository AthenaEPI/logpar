''' Extract a parcellation from a dendrogram '''
import logging
import citrix
import numpy as np
from scipy.cluster.hierarchy import fcluster

from ..utils import cifti_header, dendrogram


def check_input(outfile):
    ''' Basic output checking '''
    if not outfile.endswith('txt') and not outfile.endswith('.dlabel.nii'):
        raise ValueError("Outfile MUST be .txt or .dlabel.nii")


def extract_parcellation(dendrogram_file, nparcels, outfile):
    ''' Extracts a parcellation with a predefined number of parcels. If
        setted, also writes a cifti label file

        Parameters
        ----------
        dendrogram_file: string
            File with the dendrogram, in csv format
        nparcels: int
            Numbers of parcels the extracted parcellation should have
        out: string
            File where to write the parcellation (Either TXT or CIFTI DLABEL)

        Returns
        -------
        None
            One or more files a created '''
    check_input(outfile)

    dendro, xml_structures = dendrogram.load(dendrogram_file)

    heights = sorted(np.unique(dendro[:, 2]))
    # If WARD was used, there's a direct mapping between the number of
    # parcels and the position of the height in the tree
    heights = np.hstack(([heights[2-nparcels]], heights))
    logging.debug("len(heights): {}".format(len(heights)))
    for height in heights:
        parcellation = fcluster(dendro, height, criterion='distance')
        parcellation_size = parcellation.max()

        if parcellation_size == nparcels:
            break

    # Save the parcellation
    if outfile.endswith('txt'):
        np.savetxt(outfile, parcellation, delimiter=',')
    else:
        cifti_label_header = cifti_header.create_label_header(xml_structures,
                                                              nparcels)

        citrix.save(outfile, parcellation[None, None, None, None, None, ...],
                    header=cifti_label_header)
