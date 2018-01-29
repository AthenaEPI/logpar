import os

import nibabel
from nibabel import gifti
import numpy
from logpar.utils import cifti_utils, spatial

def create_signed_distance_files(surface, wmmask, outdir, wbcommand):
    ''' This function creates the volumes representing the signed distance
        from surfaces to the white matter '''
    # FIX THIS
    hemisphere = "L" if ".L." in surface else "R"
    outfile = os.path.join(outdir, "{}_signed_distance.nii".format(hemisphere))

    # wbcommand -create-signed-distance-volume surface refspace outvolume
    exec_str = "{0} -create-signed-distance-volume {1} {2} {3}"
    exec_str = exec_str.format(wbcommand, surface, wmmask, outfile)

    print("Running {}".format(exec_str))
    os.system(exec_str)

    return outfile

def shrink_vertices(surface, mask, sig_dis, wmmask, mm, outdir):
    ''' Shrinks the surfaces some mm in the wmmask followind the sig_dis '''
    # Load points from the surface
    surf = gifti.read(surface)
    points = numpy.copy(surf.darrays[0].data)
    
    # Load mask and signed_distance map 
    mask = gifti.read(mask).darrays[0].data.astype(bool)
    sd = nibabel.load(sig_dis).get_data()

    # Load wmmask and calculate unit distance in mm
    wm = nibabel.load(wmmask)
    affine = wm.affine
    header = wm.header
    dis_axis = numpy.array(header.get_zooms())[:3]

    dis_axis = abs(affine.dot([2, 2, 2, 1]) - affine.dot([1, 1, 1, 1]))[:3]

    # Transform the points to voxels
    inv_affine = numpy.linalg.inv(affine)
    points = nibabel.affines.apply_affine(inv_affine, points)

    # Shrink surfaces into the wmm following the gradient
    gradient = numpy.array(numpy.gradient(-sd))  #  !This needs to be an array
    seeds = []
    for pnt in points[mask]:
        seed = spatial.grad_descend(pnt, gradient, mm, dis_axis)
        seeds.append(map(float, seed))
    
    # Back to mm
    seeds_mm = nibabel.affines.apply_affine(affine, seeds)

    return seeds_mm, mask.nonzero()[0]


def seeds_from_surface(surface, mask, wmmask, mm, outfile, vol_out, wbcommand):
    outdir = os.path.dirname(outfile)

    signed_dist = create_signed_distance_files(surface, wmmask,
                                               outdir, wbcommand)

    seeds, indices = shrink_vertices(surface, mask, signed_dist, wmmask,
                                     mm, outdir)

    if vol_out:
        # Save volume with seeds
        wm = nibabel.load(wmmask)
        inv_affine = numpy.linalg.inv(wm.affine)

        volume = numpy.zeros(wm.shape)
        seeds_vox = nibabel.affines.apply_affine(inv_affine, seeds)
        seeds_vox = numpy.floor(seeds_vox).astype(int)

        volume[tuple(numpy.transpose(seeds_vox))] = 1
        cifti_utils.save_nifti(vol_out, volume, affine=wm.affine, version=1)

    # GIFTI information
    modeltype = 'CIFTI_MODEL_TYPE_SURFACE'
    brainstru = "CIFTI_STRUCTURE_CORTEX_" + ("LEFT" if "L" == surface.split('.')[1] else "RIGHT")
    surface_size = len(gifti.read(surface).darrays[0].data)

    with open(outfile, 'w') as f:
        f.write('#ModelType BrainStructure Index Surf_size Seeding_Points\n')
        for seed, idx in zip(seeds, indices):
            f.write('{} {} {} {} {} {} {}\n'.format(modeltype, brainstru, idx,
                                                    surface_size, *seed))
