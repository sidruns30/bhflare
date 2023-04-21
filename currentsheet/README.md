### CurrentSheetGeodesics
##  written by Siddhant Solanki

The script generates geodesics from (but not limited to) current sheets in GRMHD simulations. The code takes in
(1D) indices of the cells that form the current sheet along with 1D (flattened) coordinate and primitive arrays.
The wavevectors for the geodesics are calculated from magnetic field in the upstream -- not in the current sheet
-- by iteratively finding all the cells that are close to the current sheet cells but are not in it.

## Building and running:
module load intel
make clean
make
./build/CurrentSheetGeodesics

## Inputs:

# Location
DATA_DIR and DUMP_ID must be specified in the main.cpp. That is all that the user needs to do.

# Format
The simulation data must be saved in the form of .npy arrays, in a folder that represents the simulation time 
(DUMP_ID). For instance, simulation dump '1646' must have its data saved in [DATA_DIR]/[DUMP_ID]/ where DUMP_ID is
1646. 

The code requires the following 1D arrays in the data directory:
x.npy, y.npy, z.npy, rho.npy, press.npy, u0.npy, u1.npy, u2.npy, u3.npy, B0.npy, B1.npy, B2.npy, B3.npy, ind.npy

x, y and z must be 'cartesianized' Kerr Schild coordinates, i.e.,
x = r sin(theta) cos(phi), y = r sin(theta) sin(phi) and z = r cos(theta)
u[0-4] and B[0-4] are the 4-velocity and magnetic field primitives in HAMR
ind.npy represents indices of the cells in the current sheet

All primitives must be stored in Kerr Schild coordinates.

## Lessons Learnt
For optimal parallelization, code scalar functions (instead of loops in functions) and then vectorize them
Avoid vector.push_back(). Allocate space and then use #pragma parallel for/ #pragma simd
