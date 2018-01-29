import numpy as np

from scipy import ndimage

def grad_descend(start_pos, gradient, dist=2, weight=[1, 1, 1],
                 step_size=0.1, eps=1e-4):
    ''' Walks a determinated distance following the gradient field '''

    weight = np.abs(weight)
    x, y, z = pos_act = start_pos
    walked_distance = 0
    step_length = np.inf

    while walked_distance < dist and step_length > eps:
        x, y, z = pos_act

        direction = ndimage.map_coordinates(gradient,
                                            [[0, 1, 2], [x, x, x],
                                             [y, y, y], [z, z, z]],
                                            order=1)

        step = step_size * direction

        step_length = np.linalg.norm(np.multiply(step, weight))
        pos_new = pos_act + step

        walked_distance += step_length

        pos_act = pos_new

    return pos_act
