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

from logpar.utils import cifti_utils, cifti_header, seeds_utils, streamline_utils
import dipy.tracking.utils as dipy_utils

def tractogram(particles, shm, mask, affine, step_size, maxlen, algo,
               outdir, save_stream, wpid_seeds_info):
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
            save_stream: if True, streamlines are saved
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
    nzr_mask = mask.nonzero()
    tract = numpy.zeros((len(seeds), len(nzr_mask[0])))
    visit_map = numpy.zeros_like(mask, dtype=numpy.int16)
    streamlines = []
    used_seeds = []
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
        visit_map *= 0
        created = False
        for streamline in itertools.islice(it, particles*len(s)):
            if streamline is not None and len(streamline) > 1:
                positions = numpy.round(streamline).astype(int)
                posx, posy, posz = positions.T
                visit_map[posx, posy, posz] += 1
                created = True
                if save_stream:
                    streamlines.append(streamline)
        if created:
            used_seeds.append(map(int, cifti_info[i][2]))
        tract[i] = visit_map[nzr_mask]

    if save_stream:
        outinfo = os.path.join(outdir, "info_{}.trk".format(wpid))
        numpy.savetxt(outinfo, used_seeds)
        outfile = os.path.join(outdir, "stream_{}.trk".format(wpid))
        streamline_utils.save_stream(outfile, streamlines, affine)

    # I have a image which represents the connectivity of each seed over a
    # mask. Now I need to create the cifti header
    pmtype, pname, coord, psize = cifti_info[0]
    bm_coords = [coord]
    col_structures = []
    offset = 0
    for mtype, name, coord, size in cifti_info[1:]:
        if mtype == pmtype and name == pname:
            bm_coords.append(coord)
        else:
            # Save previous structure
            xml = cifti_header.brain_model_xml(pmtype, pname, bm_coords,
                                               offset, psize)
            col_structures.append(xml)
            offset += len(bm_coords)

            bm_coords = [coord]
            pmtype, pname, psize = mtype, name, size
    xml = cifti_header.brain_model_xml(pmtype, pname, bm_coords,
                                       offset, psize)
    col_structures.append(xml)

    row_structures = [cifti_header.brain_model_xml(
                        'CIFTI_MODEL_TYPE_VOXELS',
                        'CIFTI_STRUCTURE_ALL_WHITE_MATTER',
                        numpy.transpose(nzr_mask), 0, None)]
    header = cifti_header.create_conn_header(row_structures, col_structures,
                                             mask.shape, affine)
    outfile = os.path.join(outdir, "tractogram_{}.dconn.nii".format(wpid))
    cifti_utils.save_cifti(outfile, tract.T[None, None, None, None, :],
                           header=header)
    print("Worker {} finished".format(wpid))
    return


def vmgenerator(dmri_file, bvals_file, bvecs_file, mask_file, seeds_file,
                algorithm, particles, nbr_process, spp, step, maxlen, outdir,
                save_stream=False, verbose=0):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Read bvals, bvecs and make gtab
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Load diffusion and tractography mask
    diffusion_img = nibabel.load(dmri_file)
    diffusion_data = diffusion_img.get_data()
    affine = diffusion_img.get_affine()

    mask_img = nibabel.load(mask_file)
    mask = mask_img.get_data().astype(bool)

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

    tractogram_ = partial(tractogram, particles, shm_file, mask_file,
                          affine, step, maxlen, algorithm, outdir,
                          save_stream)

    logging.debug("Starting multiprocessing environment")
    res_streamlines = pool.map(tractogram_, list(enumerate(zip(seed_chunks,
                                                               info_chunks))))
    pool.close()
    pool.join()
