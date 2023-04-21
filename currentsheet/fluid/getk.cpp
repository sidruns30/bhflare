#include "getk.hpp"

ARRAY bfluid0, bfluid1, bfluid2, bfluid3, b2;

// Find all cells which have lie within a box_threshold of the cell
// in the current sheet

void GetClosebyCells(size_t ind, std::vector<size_t> &goodindices,
                    const ARRAY2D &COORDS, Grid *pGrid)
{
    double _x = COORDS[0][ind];
    double _y = COORDS[1][ind];
    double _z = COORDS[2][ind];
    double r = sqrt(SQR(_x) + SQR(_y) + SQR(_z));
    double theta = acos(_z/r);
    double phi = atan2(_y, _x);

    // Find the cell in the grid that contains our point
    Cell *pcell = pGrid->GetCell(r, theta, phi);
    goodindices.clear();
    iARRAY i_cells;
    // Get the indices of the neighboring points
    pcell->GetIndices(i_cells);
    for (size_t i=0; i<i_cells.size(); i++)
    {
        double x = COORDS[0][i_cells[i]];
        double y = COORDS[1][i_cells[i]];
        double z = COORDS[2][i_cells[i]];
        if ((abs(_x-x)<BOX_THRESHOLD) && (abs(_y-y)<BOX_THRESHOLD)  && (abs(_z-z)<BOX_THRESHOLD))
        {
            goodindices.push_back(i_cells[i]);
        }
    }
    return;
}

// Return an array of tuples of indices where first index corresponds to
// cell in the current sheet and the second index corresponds to the index 
// of the nearest cell not in the current sheet, which has a high B field
void FindCurrentSheet(iARRAY2D &indices, const iARRAY &i_sheet,
    const ARRAY2D &COORDS, const ARRAY2D &PRIMS, Grid* pGrid)
{

    indices.clear();
    printvar("Intial number of cells in current sheet: ", i_sheet.size());
    // Now find upstream magnetic field cells close to the current sheet cells
    int i, j, k;
    HAMR_MHD::Getbfluid(bfluid0, bfluid1, bfluid2, bfluid3, b2, COORDS, PRIMS);
    print("Finding current sheet neighbors");
    #pragma omp parallel
    {
        iARRAY2D indices_private;
        #pragma omp for nowait schedule(static) private(i,j,k)
        for (i=0; i<i_sheet.size(); i++)
        {
            std::vector<size_t> closecells;
            size_t imax;
            double x, y, z, r, bsqr_max=0.;
            bool flag = false;

            x = COORDS[0][i_sheet[i]];
            y = COORDS[1][i_sheet[i]];
            z = COORDS[2][i_sheet[i]];

            r = sqrt(SQR(x) + SQR(y) + SQR(z));

            GetClosebyCells(i_sheet[i], closecells, COORDS, pGrid);
            for (k=0; k<closecells.size(); k++)
            {
                // Upstream B dependence computed from a python script
                if (b2[closecells[k]] > BSQR_THRESHOLD/pow(r, 3.7))
                {
                    if (b2[closecells[k]] > bsqr_max)
                    {
                        bsqr_max = b2[closecells[k]];
                        imax = closecells[k];
                        flag = true;
                    }
                }
            }
            if (flag)
            {
                std::vector <size_t> temp = {i_sheet[i], imax};
                indices_private.push_back(temp);
            }
        }
        #pragma omp for schedule(static) ordered
        for(i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            indices.insert(indices.end(), indices_private.begin(), indices_private.end());
        }
    }
        printvar("Number of cells with neighbors in the current sheet: ", indices.size());
    
    return;
}


// Returns the wavevector in KS
void Initialize_XK(ARRAY &X_K, double _x, double _y, double _z, 
                                    double _u0, double _u1, double _u2, double _u3,
                                    double _bfluid0, double _bfluid1, double _bfluid2, double _bfluid3,
                                    double delta_theta)
{
    // theta and phi represent the angle of the wavevector with the b field as the z axis
    size_t i,j;

    // KS
    int metric_type = 2;

    // Get KS coordinates
    double XKS[NDIM];
    CartToKS(_x, _y, _z, XKS);

    // Get both upper and lower metrics
    double gcov[NDIM][NDIM];
    double gcon[NDIM][NDIM];

    GcovFunc(metric_type, XKS, gcov);
    GconFunc(metric_type, XKS, gcon);

    // Input fields and 4 velocity
    double U_con_KS[NDIM] = {_u0, _u1, _u2, _u3}, b_con_KS[NDIM] = {_bfluid0, _bfluid1, _bfluid2, _bfluid3};

    // Make the tetrad
    double Econ[NDIM][NDIM];
    double Ecov[NDIM][NDIM];
    make_tetrad(U_con_KS, b_con_KS, gcov, Econ, Ecov);

    double b_con_tetrad[NDIM];
    // Get bfluid in the fluid drame
    coordinate_to_tetrad(Ecov, b_con_KS, b_con_tetrad);

    // The photon wavevector in MKS
    double k_con_tetrad[NDIM], k_con_KS[NDIM];
    
    // Use random values for the direction of the wavevector from b field (theta min - 0, theta max - delta theta)
    double costheta = cos(delta_theta) + ((double)rand() / (double)RAND_MAX) * (1 - cos(delta_theta));
    // Uniform in cos phi
    // Multiply cos theta by a random number that is either -1 or 1
    int factor = (int) (rand() % 2 - 1);
    if (factor == 0)
    {
        factor += 1;
    }
    costheta *= (double) factor;
    factor = (int) (rand() % 2 - 1);
    if (factor == 0)
    {
        factor += 1;
    }
    double sintheta = factor * sqrt(1 - SQR(costheta));
    double cosphi = -1 + ((double)rand() / (double)RAND_MAX) * 2;
    double sinphi = sqrt(1 - SQR(cosphi));

    k_con_tetrad[1] = costheta;
    k_con_tetrad[2] = sintheta * cosphi;
    k_con_tetrad[3] = sintheta * sinphi;


    // Normalize the wavevector
    k_con_tetrad[0] = sqrt(SQR(k_con_tetrad[1]) + SQR(k_con_tetrad[2]) + SQR(k_con_tetrad[3]));

    // Get the coordinate frame
    tetrad_to_coordinate(Econ, k_con_tetrad, k_con_KS);

    /*
    // Write vectors to file
    #pragma omp critical
    {
        std::ofstream vector_oufile;
        vector_oufile.open("/home/siddhant/scratch/TeVlightcurve/geodesics/wavevectors.txt", std::ios::app);
        for (i=0; i<NDIM; i++)
        {
            vector_oufile << k_con_tetrad[i] << "\t";
        }
        for (i=0; i<NDIM; i++)
        {
            vector_oufile << k_con_KS[i] << "\t";
        }
        for (i=0; i<NDIM; i++)
        {
            vector_oufile << b_con_tetrad[i] << "\t";
        }
        for (i=0; i<NDIM; i++)
        {
            vector_oufile << b_con_KS[i] << "\t";
        }
            for (i=0; i<NDIM; i++)
        {
            vector_oufile << U_con_KS[i] << "\t";
        }
            for (i=0; i<NDIM; i++)
        {
            vector_oufile << XKS[i] << "\t";
        }
        vector_oufile << "\n";
        vector_oufile.close();
    }
    */

    // Write to full vector
    X_K.clear();
    for (i=0; i<NDIM; i++)
    {
        X_K.push_back(XKS[i]);
    }
    for (i=0; i<NDIM; i++)
    {
        X_K.push_back(k_con_KS[i]);
    }

    return;
}

// Return an array of arrays X_k that contains the initial position and 
// wavevector of the photon. Needs the indices of the cells in the current
// sheet as well as the indices of the upstream cells
void ConstructWavevectors(ARRAY2D &X_K, double delta_theta, int geodesics_per_point,
                        const iARRAY2D &indices,
                        ARRAY2D &COORDS, ARRAY2D &PRIMS, std::string OUTNAME)
{
    
    // Now iterate through the cells of the current sheet to get
    // wavevectors of all the cells
    size_t i, j, N;
    N = indices.size();
    if (b2.size() != COORDS[0].size())
    {
        throw std::invalid_argument("b^i must be computed before constructive wavevectors \n Call HAMR_MHD::get_bfluid \n");
    }

    // Clear input array
    X_K.clear();

    //std::ofstream vector_oufile;
    //vector_oufile.open("/home/siddhant/scratch/TeVlightcurve/geodesics/wavevectors.txt", std::ios::out);
    //vector_oufile.close();
    #pragma omp parallel
    {
        ARRAY2D X_K_private;
        #pragma omp for nowait private(i,j) schedule (static)
        for (i=0; i<N; i++)
        {
            size_t i_sheet, i_upstream;
            i_sheet = indices[i][0];
            i_upstream = indices[i][1];

            // Use magnetic fields and 4 velocity of the upstream cells
            // but the position vector of the current sheet cell
            double _x, _y, _z;
            double _u0, _u1, _u2, _u3;
            double _bfluid0, _bfluid1, _bfluid2, _bfluid3;
            _x = COORDS[0][i_sheet];
            _y = COORDS[1][i_sheet];
            _z = COORDS[2][i_sheet];
            _u0 = PRIMS[iU0][i_upstream];
            _u1 = PRIMS[iU1][i_upstream];
            _u2 = PRIMS[iU2][i_upstream];
            _u3 = PRIMS[iU3][i_upstream];
            _bfluid0 = bfluid0[i_upstream];
            _bfluid1 = bfluid1[i_upstream];
            _bfluid2 = bfluid2[i_upstream];
            _bfluid3 = bfluid3[i_upstream];

            ARRAY X_K_local;

            for (int k=0; k<geodesics_per_point; k++)
            {
                Initialize_XK(X_K_local, _x, _y, _z, _u0, _u1, _u2, _u3, _bfluid0, 
                                _bfluid1, _bfluid2, _bfluid3, delta_theta);
                //printvec(X_K_local);
                X_K_private.push_back(X_K_local);
            }
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++)
        {
            #pragma omp ordered
            {
                X_K.insert(X_K.end(), X_K_private.begin(), X_K_private.end());
            }
        }
    }
    printvar("Number of geodesics initialized: ", X_K.size());

    // Write wavevectors to a file
    OUTNAME += "_DeltaTheta_";
    OUTNAME += std::to_string(delta_theta);
    OUTNAME += "_GeodesicsPerPoint_";
    OUTNAME += std::to_string(geodesics_per_point);
    OUTNAME += ".txt";
    WriteVectorToFile(OUTNAME, X_K);
    return;
}
