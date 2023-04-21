#include "input/load.hpp"
#include "input/regrid.hpp"
#include "fluid/getk.hpp"

int main()
{
    int i;
    std::string DIRNAME("/home/siddhant/scratch/TeVlightcurve/npy_data/");
    std::string DUMP("1912_reduced");
    std::string FNAME = DIRNAME + DUMP;

    ARRAY2D COORDS, PRIMS;
    iARRAY i_sheet;
    iARRAY2D indices;

    // Load all data (PRIMS + COORDS + Current Sheet Indices)
    std::cout<<"Loading data \n";
    InitializeNumpyArrays(COORDS, PRIMS, i_sheet, FNAME);

    // Make the coarse grid
    const double rmin = 1.3;
    const double rmax = 15;
    const double thetamin = 30*PI/180;
    const double thetamax = 150*PI/180;
    const double phimin = -PI;
    const double phimax = PI;
    const double dr = 0.3;
    const double dtheta = 5 * PI/180;
    const double dphi = 5 * PI/180;
    bool spherical = true;
    Grid *pGrid = new Grid( COORDS, rmin, rmax, thetamin, thetamax, phimin, phimax, dr,
                             dtheta, dphi, spherical);
    
    // Find all cells near the current sheet
    FindCurrentSheet(indices, i_sheet, COORDS, PRIMS, pGrid);
    delete pGrid;

    // Make wavevectors in 10 degree increments
    int geodesics_per_point = 1;
    ARRAY2D X_K_small, X_K_big;

    double delta_theta_small = 1.e-4;
    double delta_theta_big = 90 * PI/180;

    // Construct and write wavevectors
    std::string OUTNAME("/home/siddhant/scratch/TeVlightcurve/geodesics/");
    OUTNAME += DUMP;
    
    for (int i=0; i<4; i++)
    {
        //ConstructWavevectors(X_K_small, delta_theta_small, geodesics_per_point, indices, COORDS, PRIMS, OUTNAME);
        ConstructWavevectors(X_K_big, delta_theta_big, geodesics_per_point, indices, COORDS, PRIMS, OUTNAME);
        geodesics_per_point *= 10;
    }
    return 0;
}
