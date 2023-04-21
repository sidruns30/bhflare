#ifndef __gdsc_integrator__
    #define __gdsc_integrator__
    #include "stdio.h"
    #include "math.h"
    #include "stdlib.h"
    // Things to change for the integrator
    double a_BH = 0.9375;
    double M_BH = 1.;
    double R_min_MKS = 0.;
    double R_max_MKS = 9.21034;

    #define SQR(x) ((x)*(x))
    #define PI (3.141592653589793238)
    // Definitions that may be useful for the MKS coordinatees
    #define h_ks (0.9)
    #define R0_ks (0.)
    // Definitions for the geodesics
    #define ETOL 1.e-6
    #define MAX_ITER 6
    #define EPS (0.01)
    #define EPS_END (0.0001)
    #define SMALL (1.e-40)
    #define FAST_CPY(in, out)                                                      \
    {                                                                            \
        out[0] = in[0];                                                            \
        out[1] = in[1];                                                            \
        out[2] = in[2];                                                            \
        out[3] = in[3];                                                            \
    }
void kernel_GetManyGeodesics(int ngeodesics, int size_per_geodesic, double *XKS_list, double *KKS_list, double *out);
void kernel_GetGeodesic(double Xinp[4], double Kinp[4], int size, double epsilon, double *out);
#endif