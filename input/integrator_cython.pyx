# cython: language_level = 3

cimport cython
import numpy as np
cimport numpy as np
import sys, traceback
cimport numpy
from cython.parallel import prange
import time

cdef extern from "integrator.c":
    void kernel_GetGeodesic(double *Xinp, double *Kinp, int size, double epsilon, double *out)
    void ConvertCoords(double XKS[4], double XMKS[4]);
    void ConvertCoordsInverse(double XMKS[4], double XKS[4]);
    void ConvertVectors(double XMKS[4], double KKS[3], double KMKS[4]);
    void ConvertVectorsInverse(double XMKS[4], double KMKS[4], double KKS[4]);
    void GcovFunc(double XMKS[4], double g[4][4]);
    void GetConnection(double X[4], double gamma[4][4][4]);
    double stepsize(double XKS[4], double K[4], double epsilon);
    void init_dKdlam(double X[], double Kcon[], double dK[]);
    void push_photon(double X[4], double Kcon[4], double dKcon[4],
                     double dl, int n);
    void normalize_k(double *X, double *K);
    void kernel_GetGeodesicArray(double *Xinp, double *Kinp, int points_per_geodesic, int num_geodesics, 
                        double epsilon, double *out);
    pass

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef GetGeodesicCython(double [:]X, double  [:]K, int points_per_geodesic, double epsilon):
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] out = np.zeros(shape=(10*points_per_geodesic), dtype=float)
    kernel_GetGeodesic(&X[0], &K[0], points_per_geodesic, epsilon, &out[0])
    return out

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64

cpdef GetGeodesicArrayCython(double [:,:] Xs, double[:,:] Ks, int points_per_geodesic, double epsilon):
    cdef int i, N
    N = Xs.shape[0]
    cdef np.ndarray[cDOUBLE, ndim=2, mode="c"] full_carr_out = np.zeros(shape=(N, 10 * points_per_geodesic), dtype=float)

    for i in range(N):
        full_carr_out[i,:] = GetGeodesicCython(Xs[i], Ks[i], points_per_geodesic, epsilon)
    return full_carr_out


cpdef GetGeodesicArrayCythonFast(np.ndarray[np.double_t, ndim=2, mode="c"]  Xs, np.ndarray[np.double_t, ndim=2, mode="c"] Ks, 
                            int points_per_geodesic, double epsilon):

    cdef int num_geodesics = Xs.shape[0]
    cdef np.ndarray[cDOUBLE, ndim=1, mode="c"] full_carr_out = np.zeros(
                                            shape=(num_geodesics * 10 * points_per_geodesic), dtype=float)
    # Change Xs and Ks such that every 4 consecutive elements are the 4 vectors
    cdef np.ndarray[cDOUBLE, ndim=1, mode="c"] Xcpy = Xs.ravel()
    cdef np.ndarray[cDOUBLE, ndim=1, mode="c"] Kcpy = Ks.ravel()
    kernel_GetGeodesicArray(&Xcpy[0], &Kcpy[0], points_per_geodesic, num_geodesics, epsilon, &full_carr_out[0])
    return full_carr_out.reshape(num_geodesics, 10 * points_per_geodesic)
