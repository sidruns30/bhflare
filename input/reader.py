'''
    Python script that use's Matthew Liska's code (functions.c, pp.pyx) to read in HAMR data
    There is no dependency on the pp.py file
'''
import numpy as np
import os, sys

pp_c = None

'''
    Function to build the cython file. Needs to be called before other functions
    Named as 'set_mpi()' in the pp.py file
    input:
        MPI_FLAG: (int) Flag to use MPI for load script
'''
def BuildCython(MPI_FLAG=0):
    from Cython.Build import cythonize
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    global comm, numtasks, rank,setmpi
    if (MPI_FLAG == 1):
        print("Building with MPI")
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        numtasks = comm.Get_size()
        rank = comm.Get_rank()
        setmpi=1
        if len(sys.argv) > 1:
            if sys.argv[1] == "build_ext":
                if (rank == 0):
                    setup(
                        cmdclass={'build_ext': build_ext},
                        ext_modules=[Extension("pp_c", sources=["pp_c.pyx", "functions.c"],
                        include_dirs=[np.get_include()], extra_compile_args=["-fopenmp"],
                        extra_link_args=["-O3 -fopenmp"])]
                    )
    else:
        print("Building without MPI")
        numtasks = 1
        rank = 0
        setmpi=0
        if len(sys.argv) > 1:
            if sys.argv[1] == "build_ext":
                if (rank == 0):
                    setup(
                        cmdclass={'build_ext': build_ext},
                        ext_modules=[Extension("pp_c", sources=["pp_c.pyx", "functions.c"],
                        include_dirs=[np.get_include()], extra_compile_args=["-fopenmp"],
                        extra_link_args=["-O3 -fopenmp"])]
                    )

    if (setmpi == 1):
        comm.barrier()
    return


'''
    Get dump info from the output
    input:
        dump: (int) name of the output dump
'''
def RblockNew(dump):
    global AMR_ACTIVE, AMR_LEVEL,AMR_LEVEL1,AMR_LEVEL2,AMR_LEVEL3, AMR_REFINED, AMR_COORD1, AMR_COORD2, AMR_COORD3, AMR_PARENT
    global AMR_CHILD1, AMR_CHILD2, AMR_CHILD3, AMR_CHILD4, AMR_CHILD5, AMR_CHILD6, AMR_CHILD7, AMR_CHILD8
    global AMR_NBR1, AMR_NBR2, AMR_NBR3, AMR_NBR4, AMR_NBR5, AMR_NBR6, AMR_NODE, AMR_POLE, AMR_GROUP
    global AMR_CORN1, AMR_CORN2, AMR_CORN3, AMR_CORN4, AMR_CORN5, AMR_CORN6
    global AMR_CORN7, AMR_CORN8, AMR_CORN9, AMR_CORN10, AMR_CORN11, AMR_CORN12
    global AMR_NBR1_3, AMR_NBR1_4, AMR_NBR1_7, AMR_NBR1_8, AMR_NBR2_1, AMR_NBR2_2, AMR_NBR2_3, AMR_NBR2_4, AMR_NBR3_1, AMR_NBR3_2, AMR_NBR3_5, AMR_NBR3_6, AMR_NBR4_5, AMR_NBR4_6, AMR_NBR4_7, AMR_NBR4_8
    global AMR_NBR5_1, AMR_NBR5_3, AMR_NBR5_5, AMR_NBR5_7, AMR_NBR6_2, AMR_NBR6_4, AMR_NBR6_6, AMR_NBR6_8
    global AMR_NBR1P, AMR_NBR2P, AMR_NBR3P, AMR_NBR4P, AMR_NBR5P, AMR_NBR6P
    global block, nmax, n_ord, AMR_TIMELEVEL

    AMR_ACTIVE = 0
    AMR_LEVEL = 1
    AMR_REFINED = 2
    AMR_COORD1 = 3
    AMR_COORD2 = 4
    AMR_COORD3 = 5
    AMR_PARENT = 6
    AMR_CHILD1 = 7
    AMR_CHILD2 = 8
    AMR_CHILD3 = 9
    AMR_CHILD4 = 10
    AMR_CHILD5 = 11
    AMR_CHILD6 = 12
    AMR_CHILD7 = 13
    AMR_CHILD8 = 14
    AMR_NBR1 = 15
    AMR_NBR2 = 16
    AMR_NBR3 = 17
    AMR_NBR4 = 18
    AMR_NBR5 = 19
    AMR_NBR6 = 20
    AMR_NODE = 21
    AMR_POLE = 22
    AMR_GROUP = 23
    AMR_CORN1 = 24
    AMR_CORN2 = 25
    AMR_CORN3 = 26
    AMR_CORN4 = 27
    AMR_CORN5 = 28
    AMR_CORN6 = 29
    AMR_CORN7 = 30
    AMR_CORN8 = 31
    AMR_CORN9 = 32
    AMR_CORN10 = 33
    AMR_CORN11 = 34
    AMR_CORN12 = 35
    AMR_LEVEL1=  110
    AMR_LEVEL2 = 111
    AMR_LEVEL3 = 112  
    AMR_NBR1_3=113
    AMR_NBR1_4=114
    AMR_NBR1_7=115
    AMR_NBR1_8=116
    AMR_NBR2_1=117
    AMR_NBR2_2=118
    AMR_NBR2_3=119
    AMR_NBR2_4=120
    AMR_NBR3_1=121
    AMR_NBR3_2=122
    AMR_NBR3_5=123
    AMR_NBR3_6=124
    AMR_NBR4_5=125
    AMR_NBR4_6=126
    AMR_NBR4_7=127
    AMR_NBR4_8=128
    AMR_NBR5_1=129
    AMR_NBR5_3=130
    AMR_NBR5_5=131
    AMR_NBR5_7=132
    AMR_NBR6_2=133
    AMR_NBR6_4=134
    AMR_NBR6_6=135
    AMR_NBR6_8=136
    AMR_NBR1P=161
    AMR_NBR2P=162
    AMR_NBR3P=163
    AMR_NBR4P=164
    AMR_NBR5P=165
    AMR_NBR6P=166
    AMR_TIMELEVEL=36
    
    # Read in data for every block
    if (os.path.isfile("dumps%d/grid" % dump)):
        fin = open("dumps%d/grid" % dump, "rb")
        size = os.path.getsize("dumps%d/grid" % dump)
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = 36
    elif(os.path.isfile("gdumps/grid")):
        fin = open("gdumps/grid", "rb")
        size = os.path.getsize("gdumps/grid")
        nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
        NV = (size - 1) // nmax // 4
    else:
        print("Cannot find grid file!")


    # Allocate memory
    block = np.zeros((nmax, 200), dtype=np.int32, order='C')
    n_ord = np.zeros((nmax), dtype=np.int32, order='C')

    gd = np.fromfile(fin, dtype=np.int32, count=NV * nmax, sep='')
    gd = gd.reshape((NV, nmax), order='F').T
    block[:,0:NV] = gd
    if(NV<170):
        block[:, AMR_LEVEL1] = gd[:, AMR_LEVEL]
        block[:, AMR_LEVEL2] = gd[:, AMR_LEVEL]
        block[:, AMR_LEVEL3] = gd[:, AMR_LEVEL]

    i = 0
    if (os.path.isfile("dumps%d/grid" % dump)):
        for n in range(0, nmax):
            if block[n, AMR_ACTIVE] == 1:
                n_ord[i] = n
                i += 1

    fin.close()

'''
    Read more block data from the file
    input:
        dump: (int) nalme of the output dump
'''
def RparNew(dump):
    global t, n_active, n_active_total, nstep, Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
    global bs1, bs2, bs3, nb1, nb2, nb3, startx1, startx2, startx3, _dx1, _dx2, _dx3
    global tf, a, gam, cour, Rin, Rout, R0, fractheta,REF_1,REF_2,REF_3, RAD_M1
    global nx, ny, nz, nb, rhor,temp_array, gd1_temp,gd2_temp, NODE, TIMELEVEL,flag_restore,r1,r2,r3, export_raytracing_RAZIEH, interpolate_var, rank

    fin = None
    if (os.path.isfile("dumps%d/parameters" % dump)):
        fin = open("dumps%d/parameters" % dump, "rb")
    else:
        raise ValueError("Rpar error!")

    t = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    n_active = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    n_active_total = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nstep = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    Dtd = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtl = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Dtr = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    dump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    rdump_cnt = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    dt = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    failed = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    bs1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    bs3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nmax = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb1 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb2 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    nb3 = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    startx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    startx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    _dx1 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]*r1
    _dx2 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]*r2
    _dx3 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]*r3

    tf = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    a = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    gam = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    cour = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rin = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    Rout = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    R0 = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    fractheta = np.fromfile(fin, dtype=np.float64, count=1, sep='')[0]
    for n in range(0,13):
        trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
    if(trash==10):
        RAD_M1=1
    else:
        RAD_M1=0
    trash = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]

    nb = n_active_total
    rhor = 1 + (1 - a ** 2) ** 0.5

    NODE=np.copy(n_ord)
    TIMELEVEL=np.copy(n_ord)

    REF_1=1
    REF_2=1
    REF_3=1
    flag_restore = 0
    size = os.path.getsize("dumps%d/parameters" % dump)
    if(size>=66*4+3*n_active_total*4):
        n=0
        while n<n_active_total:
            n_ord[n]=np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            NODE[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n=n+1
    elif(size >= 66 * 4 + 2 * n_active_total * 4):
        n = 0
        flag_restore=1
        while n < n_active_total:
            n_ord[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            TIMELEVEL[n] = np.fromfile(fin, dtype=np.int32, count=1, sep='')[0]
            n = n + 1

    if(export_raytracing_RAZIEH==1 and (bs1%lowres1!=0 or bs2%lowres2!=0 or bs3%lowres3!=0 or ((lowres1 & (lowres1-1) == 0) and lowres1 != 0)!=1 or ((lowres2 & (lowres2-1) == 0) and lowres2 != 0)!=1 or ((lowres3 & (lowres3-1) == 0) and lowres3 != 0)!=1)):
        if(rank==0):
            print("For raytracing block size needs to be divisable by lowres!")
    if(export_raytracing_RAZIEH==1 and interpolate_var==0):
        if (rank == 0):
            print("Warning: Variable interpolation is highly recommended for raytracing!")
    fin.close()
    return

'''
    Read the grid data including the coordinates and the metric
    inputs:
        dir: (string) directory where the hamr data is stored
'''
def RgdumpNew(dir):
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3
    import pp_c
    set_cart=0
    set_xc=0

    if((bs1%lowres1)!=0 or (bs2%lowres2)!=0 or (bs3%lowres3)!=0):
        print("Incompatible lowres settings in rgdump_new")

    bs1new = int(bs1 / lowres1)
    bs2new = int(bs2 / lowres2)
    bs3new = int(bs3 / lowres3)

    nx = bs1new * nb1
    ny = bs2new * nb2
    nz = bs3new * nb3

    # Allocate memory
    x1 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    x2 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    x3 = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    r = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    h = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
    ph = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')

    if axisym:
        gcov = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')
        gcon = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')
        gdet = np.zeros((nb, bs1new, bs2new, 1), dtype=np.float32, order='C')
        dxdxp = np.zeros((4, 4, nb, bs1new, bs2new, 1), dtype=np.float32, order='C')
    else:
        gcov = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
        gcon = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
        gdet = np.zeros((nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')
        dxdxp = np.zeros((4, 4, nb, bs1new, bs2new, bs3new), dtype=np.float32, order='C')

    size = os.path.getsize('gdumps/gdump%d' %n_ord[0])
    if(size==58*bs3*bs2*bs1*8 and bs3!=1):
        flag=1
    else:
        flag=0

    pp_c.rgdump_new(flag, dir, axisym, n_ord,lowres1,lowres2,lowres3,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet)
    return

'''
    inputs:
        dir: (string) directory where the data is stored
'''
def RgdumpGriddata(dir):
    global ti, tj, tk, x1, x2, x3, r, h, ph, gcov, gcon, gdet, drdx, dxdxp, alpha, axisym, interpolate_var
    global nx, ny, nz, bs1, bs2, bs3, bs1new, bs2new, bs3new, set_cart, set_xc, lowres1,lowres2,lowres3
    global nb1, nb2, nb3, REF_1, REF_2, REF_3
    global startx1,startx2,startx3,_dx1,_dx2,_dx3, export_raytracing_RAZIEH
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, i_min, i_max, j_min, j_max, z_min, z_max, do_box, rank, gridsizex1, gridsizex2, gridsizex3
    import pp_c

    set_cart=0
    set_xc=0

    ACTIVE1 = np.max(block[n_ord, AMR_LEVEL1])*REF_1
    ACTIVE2 = np.max(block[n_ord, AMR_LEVEL2])*REF_2
    ACTIVE3 = np.max(block[n_ord, AMR_LEVEL3])*REF_3

    if ((int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1) % lowres1) != 0 or (int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2) % lowres2) != 0 or (int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3) % lowres3) != 0):
        print("Incompatible lowres settings in rgdump_griddata")

    gridsizex1 = int(nb1 * (1 + REF_1) ** ACTIVE1 * bs1/lowres1)
    gridsizex2 = int(nb2 * (1 + REF_2) ** ACTIVE2 * bs2/lowres2)
    gridsizex3 = int(nb3 * (1 + REF_3) ** ACTIVE3 * bs3/lowres3)

    _dx1 = _dx1 * lowres1 * (1.0 / (1.0 + REF_1) ** ACTIVE1)
    _dx2 = _dx2 * lowres2 * (1.0 / (1.0 + REF_2) ** ACTIVE2)
    _dx3 = _dx3 * lowres3 * (1.0 / (1.0 + REF_3) ** ACTIVE3)

    #Calculate inner and outer boundaries of selection box after upscaling and downscaling; Assumes uniform grid x1=log(r) etc
    if(do_box==1):
        i_min = max(np.int32((np.log(r_min)-(startx1+0.5*_dx1)) / _dx1) + 1, 0)
        i_max = min(np.int32((np.log(r_max)-(startx1+0.5*_dx1)) / _dx1) + 1, gridsizex1)
        j_min=max(np.int32(((2.0/np.pi*(theta_min)-1.0)-(startx2+0.5*_dx2))/_dx2) + 1,0)
        j_max=min(np.int32(((2.0/np.pi*(theta_max)-1.0)-(startx2+0.5*_dx2))/_dx2) + 1,gridsizex2)
        z_min=max(np.int32((phi_min-(startx3+0.5*_dx3))/_dx3) + 1,0)
        z_max=min(np.int32((phi_max-(startx3+0.5*_dx3))/_dx3) + 1,gridsizex3)

        gridsizex1 = i_max-i_min
        gridsizex2 = j_max-j_min
        gridsizex3 = z_max-z_min

        if((j_max<j_min or i_max<i_min or z_max<z_min) and rank==0):
            print("Bad box selection")
    else:
        i_min=0
        i_max=gridsizex1
        j_min=0
        j_max=gridsizex2
        z_min=0
        z_max=gridsizex3

    nx = gridsizex1
    ny = gridsizex2
    nz = gridsizex3

    # Allocate memory
    x1 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    x2 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    x3 = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    r = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    h = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    ph = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')

    if axisym:
        gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=np.float32, order='C')
        gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=np.float32, order='C')
        gdet = np.zeros((1, gridsizex1, gridsizex2, 1), dtype=np.float32, order='C')
        dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, 1), dtype=np.float32, order='C')
    else:
        gcov = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
        gcon = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
        gdet = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
        dxdxp = np.zeros((4, 4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')

    size = os.path.getsize('gdumps/gdump%d' %n_ord[0])
    if(size==58*bs3*bs2*bs1*8):
        flag=1
    else:
        flag=0

    pp_c.rgdump_griddata(flag, interpolate_var, dir, axisym, n_ord,lowres1, lowres2, lowres3 ,nb,bs1,bs2,bs3, x1,x2, x3, r,h, ph,gcov, gcon,dxdxp,gdet,block, nb1, nb2, nb3, REF_1, REF_2, REF_3, np.max(block[n_ord, AMR_LEVEL1]), np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]), startx1,startx2,startx3,_dx1,_dx2,_dx3, export_raytracing_RAZIEH, i_min, i_max, j_min, j_max, z_min, z_max)
    return

'''
    Read the simulationd data
    inputs:
        dir: (string) directory where the data is stored
        dump: (int) id of the simulation dump
'''
def RdumpGriddata(dir, dump):
    global rho, ug, uu,uu_rad, E_rad, RAD_M1, B, nb2d, bs1,bs2,bs3,bs1new,bs2new,bs3new,lowres1, lowres2, lowres3, gcov,gcon,axisym,_dx1,_dx2,_dx3, nb, nb1, nb2, nb3, REF_1, REF_2, REF_3, n_ord, interpolate_var, export_raytracing_GRTRANS,export_raytracing_RAZIEH, DISK_THICKNESS, a, gam, bsq, Rdot
    global startx1,startx2,startx3,_dx1,_dx2,_dx3,x1,x2,x3
    global r_min, r_max, theta_min, theta_max, phi_min, phi_max, i_min, i_max, j_min, j_max, z_min, z_max, do_box
    import pp_c

    # Allocate memory
    rho = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    ug = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    uu = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    B = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    if(export_raytracing_RAZIEH):
        Rdot = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    else:
        Rdot = np.zeros((1, 1, 1, 1), dtype=np.float32, order='C')
    bsq = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')

    if(RAD_M1):
        E_rad = np.zeros((1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
        uu_rad = np.zeros((4, 1, gridsizex1, gridsizex2, gridsizex3), dtype=np.float32, order='C')
    else:
        E_rad=np.copy(ug)
        uu_rad=np.copy(uu)
    if (os.path.isfile("dumps%d/new_dump" % dump)):
        flag = 1
    else:
        flag = 0

    pp_c.rdump_griddata(flag, interpolate_var, np.int32(RAD_M1), dir, dump, n_active_total, lowres1, lowres2, lowres3, nb,bs1,bs2,bs3, rho,ug, uu, B, E_rad, uu_rad, gcov,gcon,axisym,n_ord,block, nb1,nb2,nb3,REF_1, REF_2,REF_3, np.max(block[n_ord, AMR_LEVEL1]),np.max(block[n_ord, AMR_LEVEL2]), np.max(block[n_ord, AMR_LEVEL3]),export_raytracing_RAZIEH, DISK_THICKNESS, a, gam, Rdot, bsq, r, startx1,startx2,startx3,_dx1,_dx2,_dx3,x1,x2,x3, i_min, i_max, j_min, j_max, z_min, z_max)

    bs1new = gridsizex1
    bs2new = gridsizex2
    bs3new = gridsizex3

    if (do_box == 1):
        startx1 = startx1 + (i_min) * _dx1
        startx2 = startx2 + (j_min) * _dx2
        startx3 = startx3 + (z_min) * _dx3

    nb2d = nb
    nb = 1
    nb1 = 1
    nb2 = 1
    nb3 = 1
    return


'''
    Actual read function that you probably want to use
    inputs:
        DumpID: (list of ints) the ids of the output to be loaded
'''
def ReadData(DumpIDs=[1646, 1912]):
    # first build and import the cython script
    BuildCython(MPI_FLAG=1)
    import pp_c
    # set the directory where the data is stored
    DIR = "/home/siddhant/scratch/PLASMOID2048/reduced/"
    os.chdir(DIR)
    # set the directory where output data must be stored
    OUTDIR = "/home/siddhant/scratch/TeVlightcurve/npy_data/"
    # flag to downsample data
    REDUCE_FLAG = False

    # change input domain here. Note that too big of a domanin causes errors
    global lowres1, lowres2, lowres3, axisym, export_raytracing_GRTRANS, export_raytracing_RAZIEH, interpolate_var, DISK_THICKNESS, set_cart
    global r1, r2, r3, do_box, r_min, r_max, theta_min, theta_max, phi_min, phi_max, notebook
    lowres1 = 1
    lowres2 = 1 #1152//2 #128//2 #128//2 #1152//2
    lowres3 = 1#1152//2 #1152//2
    axisym=1
    print_fieldlines=0
    export_raytracing_GRTRANS=0
    export_raytracing_RAZIEH=0
    interpolate_var=0
    DISK_THICKNESS=0.1
    set_cart=0
    r1=2 #1 for full res, 2 for reduced data
    r2=2
    r3=2
    do_box=1 #1 for 3D, 0 for 2D
    r_min=1.34798527268
    r_max=10.0 #10.5
    theta_min=45*np.pi/180#-10.*np.pi/180.#80.*np.pi/180.
    theta_max=np.pi-45*np.pi/180 #1000.0*np.pi/180.#100.*np.pi/180.
    phi_min=-1000.*np.pi/180.
    phi_max=1000.*np.pi/180.

    # Fluid variables
    global rho, B, uu, ug, r, h, ph, dxdxp
    
    for DumpID in DumpIDs:
        RblockNew(DumpID)
        RparNew(DumpID)
        RgdumpGriddata(DIR)
        RdumpGriddata(DIR,DumpID)

        if REDUCE_FLAG:
            rho = rho[0,::2,::3,::4]
            B = B[:,0,::2,::3,::4]
            UG = ug[0,::2,::3,::4]
            U = uu[:,0,::2,::3,::4]

            X = np.empty(shape=(4, rho.shape[0], rho.shape[1], rho.shape[2]), dtype=np.float32)
            X[1,:,:,:] = r[::2,::3,::4]
            X[2,:,:,:] = h[::2,::3,::4]
            X[3,:,:,:] = ph[::2,::3,::4]

        else:
            rho = rho[0,:,:,:]
            B = B[:,0,:,:,:]
            UG = ug[0,:,:,:]
            U = uu[:,0,:,:,:]

            X = np.empty(shape=(4, rho.shape[0], rho.shape[1], rho.shape[2]), dtype=np.float32)
            X[1,:,:,:] = r
            X[2,:,:,:] = h
            X[3,:,:,:] = ph

        X[0,:,:,:] = DumpID
        B[1,:,:,:] *= dxdxp[1,1,0,:,:,:]
        B[2,:,:,:] *= dxdxp[2,2,0,:,:,:]
        U[1,:,:,:] *= dxdxp[1,1,0,:,:,:]
        U[2,:,:,:] *= dxdxp[2,2,0,:,:,:]

        # Finally save all the loaded data
        if REDUCE_FLAG: OUTDIR += "%d_reduced/" % DumpID
        else:   OUTDIR += "%d/" % DumpID

        x = X[1] * np.sin(X[2]) * np.cos(X[3]);   np.save(DIR + "x.npy", x.ravel())
        y = X[2] * np.sin(X[2]) * np.sin(X[3]);   np.save(DIR + "y.npy", y.ravel())
        z = X[3] * np.cos(X[2]);                np.save(DIR + "z.npy", z.ravel())
        np.save(DIR + "rho.npy", rho.ravel())
        np.save(DIR + "press.npy", (gam - 1) * UG.ravel())
        u0 = U[0,:,:,:];                    np.save(DIR + "u0.npy", u0.ravel())
        u1 = U[1,:,:,:];                    np.save(DIR + "u1.npy", u1.ravel())
        u2 = U[2,:,:,:];                    np.save(DIR + "u2.npy", u2.ravel())
        u3 = U[3,:,:,:];                    np.save(DIR + "u3.npy", u3.ravel())
        B0 = B[0,:,:,:];                    np.save(DIR + "B0.npy", B0.ravel())
        B1 = B[1,:,:,:];                    np.save(DIR + "B1.npy", B1.ravel())
        B2 = B[2,:,:,:];                    np.save(DIR + "B2.npy", B2.ravel())
        B3 = B[3,:,:,:];                    np.save(DIR + "B3.npy", B3.ravel())

        # plot fields
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        imid = rho.shape[1] // 2
        lognorm = mpl.colors.LogNorm(vmin=1.e-5, vmax=1.e2)
        symlognorm = mpl.colors.SymLogNorm(vmin=-1.e2, vmax=1.e2, linthresh=1.e-4)
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10,10))
        axes[0,0].pcolormesh(rho[:,imid,:], norm=lognorm, cmap="inferno"); axes[0,0].set_title("rho")
        axes[0,1].pcolormesh((gam - 1)*UG[:,imid,:], norm=lognorm, cmap="viridis"); axes[0,1].set_title("press")
        axes[1,0].pcolormesh(u0[:,imid,:], norm=symlognorm, cmap="seismic"); axes[1,0].set_title("u0")
        axes[1,1].pcolormesh(u1[:,imid,:], norm=symlognorm, cmap="seismic"); axes[1,1].set_title("u1")
        axes[2,0].pcolormesh(u2[:,imid,:], norm=symlognorm, cmap="seismic"); axes[2,0].set_title("u2")
        axes[2,1].pcolormesh(u3[:,imid,:], norm=symlognorm, cmap="seismic"); axes[2,1].set_title("u3")
        axes[3,0].pcolormesh(B0[:,imid,:], norm=symlognorm, cmap="PiYG"); axes[3,0].set_title("B0")
        axes[3,1].pcolormesh(B1[:,imid,:], norm=symlognorm, cmap="PiYG"); axes[3,1].set_title("B1")
        axes[4,0].pcolormesh(B2[:,imid,:], norm=symlognorm, cmap="PiYG"); axes[4,0].set_title("B2")
        axes[4,1].pcolormesh(B3[:,imid,:], norm=symlognorm, cmap="PiYG"); axes[4,1].set_title("B3")
        plt.savefig("%d_plot.png" % DumpID, dpi=200)
        plt.close()
    return

def main():
    ReadData()
    return

if __name__ == "__main__":
    main()
