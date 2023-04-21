#include "load.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "cnpy.h"

// Global fields
ARRAY2D COORDS;
ARRAY2D PRIMS;

// Load double/float numpy array into a double vector
void LoadNumpyArray(std::string dirname, std::string arrname, ARRAY &vec)
{
    std::string path = dirname + "/" + arrname + ".npy";
    std::cout<<"Loading array "<<arrname<<std::endl;
    cnpy::NpyArray *parray = new cnpy::NpyArray;
    *parray = cnpy::npy_load(path);
    size_t i, nelements = parray->shape[0];
    size_t word_size = parray->word_size, float32=4;
    vec.reserve(nelements);
    if (word_size == float32)
    {
        float *fdata = parray->data<float>();
        for (i=0; i<nelements; i++)
        {
            vec.push_back(static_cast<double>(fdata[i]));
        }

    }
    else
    {
        double *data = parray->data<double>();
        for (i=0; i<nelements; i++)
        {
            vec.push_back(static_cast<double>(data[i]));
        }
    }

    //GetArrayInfo(arrname, vec);
    delete parray;
    return;
}

// Load size_t numpy array into a size_t vector
void LoadNumpyArray(std::string dirname, std::string arrname, iARRAY &vec)
{
    std::string path = dirname + "/" + arrname + ".npy";
    std::cout<<"Loading array "<<arrname<<std::endl;
    cnpy::NpyArray *parray = new cnpy::NpyArray;
    *parray = cnpy::npy_load(path);
    size_t i, nelements = parray->shape[0];
    vec.reserve(nelements);
    size_t *fdata = parray->data<size_t>();
    for (i=0; i<nelements; i++)
    {
        vec.push_back(static_cast<size_t>(fdata[i]));
    }
    //GetArrayInfo(arrname, vec);
    delete parray;
    return;
}


// Initialize numpy arrays
void InitializeNumpyArrays(ARRAY2D &COORDS, ARRAY2D &PRIMS, iARRAY &i_sheet, 
                            std::string fname)
{
    ARRAY x, y, z;
    ARRAY rho, p, u0, u1, u2, u3, B0, B1, B2, B3;

    LoadNumpyArray(fname, "x", x);
    LoadNumpyArray(fname, "y", y);
    LoadNumpyArray(fname, "z", z);
    LoadNumpyArray(fname, "rho", rho);
    LoadNumpyArray(fname, "press", p);
    LoadNumpyArray(fname, "u0", u0);
    LoadNumpyArray(fname, "u1", u1);
    LoadNumpyArray(fname, "u2", u2);
    LoadNumpyArray(fname, "u3", u3);
    LoadNumpyArray(fname, "B0", B0);
    LoadNumpyArray(fname, "B1", B1);
    LoadNumpyArray(fname, "B2", B2);
    LoadNumpyArray(fname, "B3", B3);
    LoadNumpyArray(fname, "indices", i_sheet);

    // Push back data to global arrays    
    COORDS.push_back(x);
    COORDS.push_back(y);
    COORDS.push_back(z);
    PRIMS.push_back(rho);
    PRIMS.push_back(p);
    PRIMS.push_back(u0);
    PRIMS.push_back(u1);
    PRIMS.push_back(u2);
    PRIMS.push_back(u3);
    PRIMS.push_back(B0);
    PRIMS.push_back(B1);
    PRIMS.push_back(B2);
    PRIMS.push_back(B3);

    
    x.clear();
    y.clear();
    z.clear();
    rho.clear();
    p.clear();
    u0.clear();
    u1.clear();
    u2.clear();
    u3.clear();
    B0.clear();
    B1.clear();
    B2.clear();
    B3.clear();
    return;
}