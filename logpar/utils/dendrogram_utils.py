''' Utils to manipulate dendrograms with CIFTI information '''
import xml.etree.ElementTree as xml
import numpy

def save(outfile, dendrogram, xml_structures=None):
    ''' Saves the dendrogram in csv format, including the xml
        information as a comment in the header

        Parameters
        ----------
        outfile: string
            File where to save the dendrogram
        dendrogram: array-like
            Dendrogram formated as an array
        xml_structures: list(xml_entities) (optional)
            List with one or more XML Entities

        Returns
        -------
        None
            A file is created
        '''
    outfile += '.csv' if outfile[-4:] != '.csv' else ''

    header = ''
    if xml_structures is not None:
        for structure in xml_structures:
            # Remove newlines
            cifti_string = " ".join(xml.tostring(structure).split())
            header += 'CIFTI {}\n'.format(cifti_string)
    numpy.savetxt(outfile, dendrogram, delimiter=',', header=header)


def load(dendrogram_file, return_xml=True):
    ''' Reads a dendrogram from a csv file. If setted, it also returns
        the CIFTI XML data encoded in the dendrogram file header

        Parameters
        ----------
        dendrogram_file: string
            Route to the dendrogram file
        return_xml: bool (optional)
            If true (default value), the CIFTI XML data encoded in the
            dendrogram is returned
        Returns
        ------
        array_like
            Dendrogram
        list
            List of xml_entities
        '''
    dendrogram = numpy.loadtxt(dendrogram_file, delimiter=',')

    if not return_xml:
        return dendrogram

    xml_structures = []
    with open(dendrogram_file) as dfile:
        for line in dfile.readlines():
            if line[:7] == '# CIFTI':
                xml_structures.append(xml.fromstring(line[7:]))

    return dendrogram, xml_structures
