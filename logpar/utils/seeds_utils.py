import numpy


def load_seeds(seeds_file):
    ''' Returns starting points grouped by seeds '''
    cifti_info = []
    seeds_pnts = []
    with open(seeds_file) as f:
        f.readline()
        
        for line in f:
            splitted = line.split()
            
            model_type = splitted[0]
            brain_structure = splitted[1]

            if model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                coord = splitted[2:5]
                size = None
                seeds = splitted[5:]
            else:
                coord = splitted[2]
                size = splitted[3]
                seeds = splitted[4:]

            seeds = numpy.array(map(float, seeds))
            seeds.resize((len(seeds)/3, 3))
            cifti_info.append((model_type, brain_structure, coord, size))
            seeds_pnts.append(seeds)

    return cifti_info, seeds_pnts


def save_seeds(args, seeds_file):
    raise NotImplemented()
