#ifndef _HAMR_MHD_CSG
#define _HAMR_MHD_CSG
#ifndef DEF_HEAD
#define DEF_HEAD (1)
#include "../defs.hpp"
#endif

#include "../metric/metric.hpp"


// The coords and primitives are given by the reader. The file only requires to be linked to the
// metric file

// Check if MHD is ideal
extern const bool idealMHD;

// Return global fluid quantities
namespace HAMR_MHD
{
    void Getbfluid(ARRAY &bfluid0, ARRAY &bfluid1, ARRAY &bfluid2, ARRAY &bfluid3, ARRAY &b2,
                    const ARRAY2D &COORDS, const ARRAY2D &PRIMS);


    void GetBsqr(ARRAY &Bsqr, const ARRAY2D &COORDS, const ARRAY2D &PRIMS);
    void GetEsqr(ARRAY &Esqr, const ARRAY2D &COORDS, const ARRAY2D &PRIMS);
    
    void GetTemp(ARRAY &temp, const ARRAY2D &COORDS, const ARRAY2D &PRIMS);    
    void GetSigma(ARRAY &sigma, ARRAY &b2, const ARRAY2D &COORDS, const ARRAY2D &PRIMS);
    void GetBeta(ARRAY &beta, ARRAY &Bsqr, const ARRAY2D &COORDS, const ARRAY2D &PRIMS);
};
#endif