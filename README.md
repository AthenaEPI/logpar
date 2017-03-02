# Logistic Random Effects Based Parcellation 
Groupwise Structural Parcellation of the Whole Cortex: A Logistic Random Effects Model Based Approach

[![Build Status](https://travis-ci.org/AthenaEPI/logpar.svg?branch=master)](https://travis-ci.org/AthenaEPI/logpar)

### Installation
- clone repository
- python setup.py install

### Command Line Tools
Please read the ![examples](https://github.com/AthenaEPI/logpar/tree/master/examples). More examples coming soon.


#### Averaging CIFTI connectivity files
The *cifti_average* tool computes the average connectivity from a group of connectivity matrices. If the *in_logodds* flag is present, then the matrices are transformed to the LogOdds space, averaged, and transformed back.
~~~~~~
cifti_average -matrices CIFTI_FILE_1 CIFTI_FILE_2 ... -out CIFTI_OUT -in_logodds
~~~~~~


#### Parcelling a CIFTI connectivity file
The *cifti_parcellate* tool parcellates a CIFTI connectivity file in a given DIRECTION. If the flag *transform* is present, the data is transformed to the LogOdds space as explained in Gallardo et al. (2017). If a *constraint* and a *minimum_size* are given, then the clustering is performed only between neighbors, until the *min_size* is reached.
~~~~~~
cifti_parcellate -cifti CIFTI_FILE [-direction ROW/COLUMN] [-transform] [-constraints SURF/VOLUME-FILE] [-min_size N] -out DENDROGRAM_FILE.csv
~~~~~~


#### Extracting a parcellation from the dendrogram
The *extract_parcellation* retrieves a parcellation with a defined *number of parcels* from a *dendrogram*. The output file can be both a CIFTI LABEL file (dlabel.nii) or a TXT file.
~~~~~~
extract_parcellation -dendrogram DENDROGRAM_FILE.csv -parcels nparcels -out OUT_FILE.
~~~~~~


##### Reference
Guillermo Gallardo, William Wells III, Rachid Deriche, Demian Wassermann, Groupwise structural parcellation of the
whole cortex: a logistic random effects model based approach. 2017, in press. NeuroImage. ![http://dx.doi.org/10.1016/j.neuroimage.2017.01.070](http://dx.doi.org/10.1016/j.neuroimage.2017.01.070)
