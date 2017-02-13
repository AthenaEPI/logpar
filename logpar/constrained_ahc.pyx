# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as np
cimport libc.math
cimport cython
from cython cimport floating

cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double centroid(double d_xi, double d_yi, double d_xy,
                     double size_x, double size_y, double size_i):
    ''' Return the euclidean distance between centroid(x,y) and i '''

    cdef double e1 = (size_x * d_xi * d_xi)
    cdef double e2 = (size_y * d_yi * d_yi)
    cdef double e3 = (size_x * size_y * d_xy * d_xy)
    cdef double e4 = (size_x + size_y)

    cdef double r = libc.math.sqrt(((e1 + e2) - (e3 / e4)) / e4)
    return r


@cython.wraparound(False) 
@cython.boundscheck(False)
cdef double ward(double d_xi, double d_yi, double d_xy,
                 int size_x, int size_y, int size_i):
    cdef double t = 1.0 / (size_x + size_y + size_i)
    return libc.math.sqrt((size_i + size_x) * t * d_xi * d_xi +
                          (size_i + size_y) * t * d_yi * d_yi -
                           size_i * t * d_xy * d_xy)


@cython.wraparound(False)
@cython.boundscheck(False)
def cond2mat_index(int n, int i):
    ''' Indexes (i, j) of element i in a n x n condensed matrix '''
    cdef double b = 1. - 2.*n
    cdef double x = libc.math.floor((-b - libc.math.sqrt(b**2. - 8.*i))/2.)
    cdef double y = i + x*(b+x+2.)/2. + 1.
    return int(x), int(y)


def mat2cond_index(int n, int i, int j):
    ''' Calculate the condensed index of element (i, j) in an n x n condensed
        matrix. '''
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int cluster_size(int id_c, int n, np.ndarray[double, ndim=2] Z):
    return 1 if id_c < n else <int>Z[id_c - n, 3]   

@cython.wraparound(False)
@cython.boundscheck(False)
def all_distances_and_supervoxels(heap,
                                  np.ndarray[double, ndim=2] Z,
                                  int n,
                                  int[:] id_map,
                                  signed char[:] neighbors,
                                  int size,
                                  np.ndarray[floating, ndim=2] F):
    cdef int i, j, indx, ref_ij
    cdef double dist
    cdef double max_dist = 0.

    if size > 0:
        # Imposing minimum cluster size: forcing minimum granularity

        for i in range(n-1):
            dist = Z[i,2]
            if dist > max_dist:
                max_dist = dist
        
        for i in range(n-1):
            Z[i,2] = max_dist #max_dist

        neighbors[:] = 0
        heap.reset()

    # Calculate distance between all the active clusters
    for i in range(n-1):
        if id_map[i] == -1:
            continue

        for j in range(i+1,n):
            if id_map[j] == -1:
                continue

            dist = np.linalg.norm(F[i] - F[j])
            ref_ij = mat2cond_index(n, i, j)
            heap.push(dist, ref_ij)
            neighbors[ref_ij] = 1

    return max_dist

@cython.wraparound(False)
@cython.boundscheck(False)
def agglomerative(np.ndarray[floating, ndim=2] features, 
                  np.ndarray[double, ndim=2] Z, int n,
                  heap, signed char[:] neighbors, int size, int method,
                  int verbose=0, int copy=1):
                       
    cdef short impose_min_size, neighbour_x, neighbour_y
    cdef int k, x, y, id_x, id_y, id_i, ref, ref_ix, ref_iy
    cdef double min_height, dxy, dyi, dxi, new_dist
    cdef signed char[:] N = np.ndarray(n * (n - 1) / 2, dtype=np.int8)

    cdef float nx, ny, ni

    distances = [ward, centroid]
    dist = distances[method]
    impose_min_size = (size>0)
    min_height = 0
    
    if copy > 0:
        F = features.copy()
    else:
        print "Modifying features matrix, please be aware."
        F = features
    N[:] = neighbors
   
    cdef int[:] id_map = np.ndarray(n, dtype=np.int32)

    for i in range(n):
        id_map[i] = i
    
    for k in range(n-1):
        
        if verbose and k % 1000 == 0:
            print "itera", k, n

        dxy, ref = heap.pop()

        if dxy == np.inf:
            # We run out of distances: there are no more neighbors
            if verbose:
                print "out of neigh", max(Z[:,3])
            impose_min_size = 0
            min_height = all_distances_and_supervoxels(heap, Z, n, id_map,
                                                       N, size, F)
            #I need a new value
            dxy, ref = heap.pop()

        x, y = cond2mat_index(n, ref)

        id_x = id_map[x]
        id_y = id_map[y]

        if id_x < 0 or id_y < 0:
            raise ValueError("Wrong id in selected clusters")

        nx = cluster_size(id_x, n, Z)
        ny = cluster_size(id_y, n, Z)
        min_size_reached = (nx + ny >= size)

        if verbose and impose_min_size:
            if (nx + ny > 2*size):
                raise ValueError("Code error? Cluster bigger than imposed")

        # record the new node
        if id_x < id_y:
            Z[k, 0] = id_x
            Z[k, 1] = id_y
        else:
            Z[k, 0] = id_y
            Z[k, 1] = id_x

        Z[k, 2] = dxy + min_height
        Z[k, 3] = nx + ny

        if k == n-2:
            continue  # Last step done, we finished

        id_map[x] = -1      # cluster x will be dropped
        id_map[y] = n + k   # cluster y will be replaced with the union        
    
        F[y] = (ny*F[y]+nx*F[x])/(ny+nx)

        for i in xrange(n):
            # Calculate distance to the rest of the clusters
            id_i = id_map[i]

            if id_i == -1 or id_i == n + k:
                continue
            
            ref_ix = mat2cond_index(n, i, x)
            ref_iy = mat2cond_index(n, i, y)
            neighbour_x = N[ref_ix]
            neighbour_y = N[ref_iy]
            if not (neighbour_x or neighbour_y):
                # We are only interested in distances with neighbors
                continue

            if min_size_reached and impose_min_size:
                # Isolate it
                new_dist = np.inf
                N[ref_iy] = 0
            else:
                # Calculate distance
                if (neighbour_x>0) and (neighbour_y>0):
                    # If the three are neigbours then we use L&W formula
                    dxi = heap.value_of(ref_ix)
                    dyi = heap.value_of(ref_iy)
                    ni = cluster_size(id_i, n, Z)
                    new_dist = dist(dxi, dyi, dxy, nx, ny, ni)
                else:
                    # We do it manually
                    new_dist = np.linalg.norm(F[y] - F[i])
                    if method == 0:
                        # Ward
                        ni = cluster_size(id_i, n, Z)
                        new_dist *= np.sqrt(2.*ni*(nx+ny)/float(ni+nx+ny))

                N[ref_iy] = 1

            heap.push(new_dist, ref_iy)
            
            # X is gone
            heap.push(np.inf, ref_ix)
            N[ref_ix] = 0
