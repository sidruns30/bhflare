#ifndef _REGRID_CSG_
#define _REGRID_CSG_ (1)
#ifndef DEF_HEAD
#define DEF_HEAD (1)
#include "../defs.hpp"
#endif


class Cell
{
    private:
        double x, y, z, dx, dy, dz;
        std::vector <size_t> indices;
        double data;
    public:
        Cell();
        Cell(double x, double y, double z);
        Cell(double x, double y, double z, double dx, double dy, double dz);
        void GetCoordinates(double &x, double &y, double &z);
        void GetIndices(std::vector <size_t> &indices);
        void GetSize(size_t &size);
        void AddIndex(size_t ind);
        double GetVolume();
        void GetData(double &data);
        void AverageArray(const ARRAY &arr);
        void RMSArray(const ARRAY &arr);
        void AverageVecMagnitude(const ARRAY &arrx, const ARRAY &arry, const ARRAY &arrz);
        void AverageMagnitudeVec(const ARRAY &arrx, const ARRAY &arry, const ARRAY &arrz);

};

class Grid
{
    private:
        double xmin, xmax, ymin, ymax, zmin, zmax;
        double dx, dy, dz;
        size_t ncells, nx, ny, nz;
        std::vector <Cell *> p_cell;
        void MakeCells();
        bool populated, spherical_grid;
    public:
        std::vector <double> xarr, yarr, zarr;
        Grid();
        Grid(double xmin, double xmax, double ymin, double ymax, double zmin, 
            double zmax, double dx, double dy, double dz);
        Grid(const ARRAY2D &COORDS, const double dl);
        Grid(const ARRAY2D &COORDS, const double dr, const double dtheta, const double dphi);
        Grid(const ARRAY2D &COORDS, const double rmin, const double rmax, const double thetamin,
        const double thetamax, const double phimin, const double phimax,
        const double dr, const double dtheta, const double dphi, bool spherical);
        Cell *GetCell(double x, double y, double z);
        void FillGrid(const ARRAY2D &COORDS);
        void PrintGrid();
        void ComputeQuantity(std::string method, const ARRAY &arr);
        void ComputeQuantity(std::string method, const ARRAY &arrx, const ARRAY &arry, const ARRAY &arrz);
        ARRAY GetGridData();
        void ClearCells();
        void GridToVTK(std::string outname, bool ASCII);
};
#endif