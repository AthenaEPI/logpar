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

def streamline(particles, shm, mask, affine, step_size, maxlen, algo,
               outdir, wpid_seeds_info):
    ''' Function to run in parallel

        Params:
            particles: number of particles to simulate in each seed
            shm: SHM computed from the dwi file
            mask: Mask were to perform tractography
            affine: Affine matrix of the dwi
            step_size: size in mm of each step in the tracking
            maxlen: maxlen of each streamline
            algo: tract type: probabilistic or deterministic
            outdir: Directory were to save tractograms
            wpid_seeds_info: tuple which contains:
                - wpid: The id of this worker
                - seeds: One list for each seed with points to track from
                - info: CIFTI information for each seed:
                    -mtype: A valid CIFTI MODELTYPE
                    -name: A valid CIFTI BRAINSTRUCTURE
                    -coord: Voxel or vertex to which the seed makes reference
                    -size: size of the CIFTI SURFACE (if applies)
        Returns:
            list of streamlines '''
    from dipy.tracking.local import BinaryTissueClassifier
    from dipy.tracking.local import LocalTracking

    wpid, (seeds, cifti_info) = wpid_seeds_info
    print "Worker {} started".format(wpid)

    shm = cifti_utils.load_data(shm)
    mask = cifti_utils.load_data(mask)
    dir_get = streamline_utils.direction_getter(shm, max_angle=30, algo=algo)
    classifier = BinaryTissueClassifier(mask)

    percent = max(1, len(seeds)/5)
    streamlines = []
    for i, s in enumerate(seeds):
        if i % percent == 0:
            print("{}, {}/{} strm".format(wpid, i, len(seeds)))

        # Repeat the seeds as long as needed
        repeated_seeds = itertools.cycle(s)

        res = LocalTracking(dir_get, classifier, repeated_seeds, affine,
                            step_size=step_size, maxlen=maxlen)
        it = res._generate_streamlines()  # This is way faster, just remember
                                          #  after to move them into mm space
                                          #  again.
        for streamline in itertools.islice(it, particles*len(s)):
            if streamline != []:
                streamlines.append(streamline)

    outfile = os.path.join(outdir, "stream_{}.trk".format(wpid))
    streamline_utils.save_stream(outfile, streamlines, affine)
    # I have a image which represents the connectivity of each seed over a
    # mask. Now I need to create the cifti header
    print("Worker {} finished".format(wpid))
    return


def trkgenerator(dmri_file, bvals_file, bvecs_file, mask_file, seeds_file,
                 algorithm, particles, nbr_process, spp, step, maxlen, outdir,
                 verbose=1):
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
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(diffusion_data, mask=mask)
    shm = csd_fit.shm_coeff
    shm_file = os.path.join(outdir, 'shm.nii')
    cifti_utils.save_cifti(shm_file, shm)

    #Start multiprocessing environment
    if not nbr_process:
        nbr_process = multiprocessing.cpu_count()

    cifti_info, seeds_pnts = seeds_utils.load_seeds(seeds_file)
    s = 1000
    seed_chunks = [seeds_pnts[i:i+s] for i in xrange(0, len(seeds_pnts), s)]
    info_chunks = [cifti_info[i:i+s] for i in xrange(0, len(cifti_info), s)]

    logging.debug("Chunks: {}, Seeds: {}, Particles: {}".format(
        len(seed_chunks), len(seed_chunks[0]), particles))

    pool = multiprocessing.Pool(nbr_process)

    streamline_ = partial(streamline, particles, shm_file, mask_file,
                           affine, step, maxlen, algorithm, outdir)

    logging.debug("Starting multiprocessing environment")
    res_streamlines = pool.map(streamline_, list(enumerate(zip(seed_chunks,
                                                               info_chunks))))
    pool.close()
    pool.join()
