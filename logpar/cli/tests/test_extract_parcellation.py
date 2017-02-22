''' Testing CLI of extract_parcellation '''
import logging
import xml.etree.ElementTree as xml
from tempfile import NamedTemporaryFile

import numpy
import nibabel

from .. import extract_parcellation
from ...utils import dendrogram_utils, cifti_utils

def test_correctly_configured():
    ''' Tests the dendrogram extraction for different parameters
        TODO: Improve this '''

    for name in ['all', 'left', 'right']:
        logging.debug('---- Start {} -----'.format(name))
        # Result from cifti_parcellate CLI
        dendrogram = './logpar/cli/tests/data/{}.dendrogram.csv'.format(name)
        parcels = 18

        label_out = NamedTemporaryFile(mode='w', delete=False, suffix='.dlabel.nii')
        label_out = label_out.name

        extract_parcellation.extract_parcellation(dendrogram, parcels,
                                                  label_out)
        dlabel = nibabel.load(label_out)

        dlabel_brainmodels = cifti_utils.extract_brainmodel(dlabel.header,
                                                            'ALL',
                                                            'COLUMN')
        _, dendro_brainmodels = dendrogram_utils.load(dendrogram)

        for label_bm, dendro_bm in zip(dlabel_brainmodels, dendro_brainmodels):
            numpy.testing.assert_equal(xml.tostring(label_bm),
                                       xml.tostring(dendro_bm))

        dlabel_xml_header = xml.fromstring(
            dlabel.header.extensions[0].get_content()
            )

        labels = dlabel_xml_header.findall('.//Label')
        
        # We should have #parcels + '???'background
        numpy.testing.assert_equal(len(labels), parcels+1)

        print dlabel.get_data()[0, 0, 0, 0, 0,].shape
        numpy.testing.assert_equal(dlabel.get_data()[0, 0, 0, 0, 0].max(),
                                   parcels)
