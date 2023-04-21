#ifndef DEF_HEAD
#define DEF_HEAD (1)
#include "../defs.hpp"
#endif

// Coordinates also imports metric
#include "../metric/metric.hpp"
#include "HAMR_MHD.hpp"
#include "../input/regrid.hpp"

extern ARRAY bfluid0, bfluid1, bfluid2, bfluid3, b2;

// Get the cells that are next to the cells in the upstream (i.e. cells on the outside of the current sheet)
void FindCurrentSheet(iARRAY2D &indices, const iARRAY &i_sheet,
    const ARRAY2D &COORDS, const ARRAY2D &PRIMS, Grid* pGrid);

void ConstructWavevectors(ARRAY2D &X_K, double delta_theta, int geodesics_per_point,
                        const iARRAY2D &indices, 
                        ARRAY2D &COORDS, ARRAY2D &PRIMS, std::string OUTNAME);

