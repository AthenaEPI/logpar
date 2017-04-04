''' Given a set of seeds, computes streamlines for each seeds and returns
    a visits' map over a mask '''
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import CsaOdfModel
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

import nibabel
import numpy
import itertools
from functools import partial
import os
import multiprocessing
import logging

from logpar.utils import cifti_utils, seeds_utils, streamline_utils
import dipy.tracking.utils as dipy_utils

def streamline(particles, shm, mask, affine, step_size, maxlen,
               outdir, zooms, shape, enum_seeds):
    ''' Function to run in parallel

        Params:
            particles: number of particles to simulate in each seed
            shm: SHM computed from the dwi file
            gfa: GFA computed from the dwi file
            affine: Affine matrix of the dwi
            step_size: size in mm of each step in the tracking
            maxlen: maxlen of each streamline
            seeds: list of points were to track from
        Returns:
            list of streamlines '''
    from dipy.tracking.local import BinaryTissueClassifier
    from dipy.direction import ProbabilisticDirectionGetter
    from dipy.data import default_sphere
    from dipy.tracking.local import LocalTracking
    
    wpid, seeds = enum_seeds
    print "Worker {} started".format(wpid)

    shm = cifti_utils.load_data(shm)
    mask = cifti_utils.load_data(mask)
    dir_get = ProbabilisticDirectionGetter.from_shcoeff(shm, max_angle=30.,
                                                        sphere=default_sphere)
    classifier = BinaryTissueClassifier(mask)

    streamlines = []
    percent = max(1, len(seeds)/5)
    for i, s in enumerate(seeds):
        if i % percent == 0:
            print("{}, {}/{} strm".format(wpid, i, len(seeds)))
        
        repeated_seeds = itertools.cycle(s)

        res = LocalTracking(dir_get, classifier, repeated_seeds, affine,
                            step_size=step_size, maxlen=maxlen)
        it = res._generate_streamlines()  # This is way faster, just remember
                                          #  after to move them into mm space
                                          #  again.
        streams = list(itertools.islice(it, particles))
        streamlines += streams

    out_trk = os.path.join(outdir, '{}_strm.trk'.format(wpid))

    print "Worker {} saving streamlines".format(wpid)
    streamline_utils.save_stream(out_trk, streamlines, affine)

    print("Worker {} finished".format(wpid))
    return out_trk


def vmgenerator(dmri_file, bvals_file, bvecs_file, mask_file, seeds_file,
                particles, nbr_process, spp, step, maxlen,  outdir, verbose=1):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Read bvals, bvecs and make gtab
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
    
    # Load diffusion and tractography mask
    diffusion_img = nibabel.load(dmri_file)
    diffusion_data = diffusion_img.get_data()

    mask_img = nibabel.load(mask_file)
    mask = mask_img.get_data().astype(bool)
    affine = mask_img.get_affine()

    # Fit CSD model
    logging.debug("Fitting CSD model")
    response, ratio = auto_response(gtab, diffusion_data)
    #csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    #csd_fit = csd_model.fit(diffusion_data, mask=mask)
    #shm = csd_fit.shm_coeff
    shm_file = os.path.join(outdir, 'shm.nii')
    #cifti_utils.save_cifti(shm_file, shm)

    #Start multiprocessing environment
    if not nbr_process:
        nbr_process = multiprocessing.cpu_count()

    start_pnts = seeds_utils.starting_points(seeds_file)
    
    s = 1000
    chunks = [start_pnts[i:i+s] for i in xrange(0, len(start_pnts), s)]
    info = "Working with {} chunks of {} seeds and {} particles".format(
        len(chunks), len(chunks[0]), particles)
    logging.debug(info)

    pool = multiprocessing.Pool(nbr_process)

    zooms, shape = diffusion_img.header.get_zooms(), diffusion_img.shape[:3]

    streamlines_ = partial(streamline, particles, shm_file, mask_file, affine,
                           step, maxlen, outdir, zooms, shape)
    logging.debug("Starting multiprocessing environment")
    res_streamlines = pool.map(streamlines_, list(enumerate(chunks)))

    pool.close()
    pool.join()
