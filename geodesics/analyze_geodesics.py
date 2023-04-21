import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import integrator_cython

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.size'] = 15
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
mpl.rcParams['figure.dpi']=180


DATA_DIR = "/home/siddhant/scratch/TeVlightcurve/geodesics/1912_theta_"
theta = 9 * (np.pi / 180) * np.arange(10)

def convertMKScoordtoKS(xmks):
    h_ks = 0.9
    xks = np.copy(xmks)
    xks[:,1] = np.exp(xmks[:,1])
    xks[:,2] = xmks[:,2] + (2 * h_ks * xmks[:,2] / np.pi**2) * (np.pi - 2*xmks[:,2]) * (np.pi - xmks[:,2])
    return xks

def convertMKSvectotKS(xmks, v):
    h_ks = 0.9
    vks = np.copy(v)
    vks[:,1] *= np.exp(xmks[:,1])
    vks[:,2] *= (1 + h_ks * np.cos(2 * xmks[:,2]))
    return vks

def convertKSvectortoCART(xks, vks):
    vcart = np.copy(vks)
    sth = np.sin(xks[:,2])
    cth = np.cos(xks[:,2])
    sph = np.sin(xks[:,3])
    cph = np.cos(xks[:,3])
    r = xks[:,1]
    vr, vth, vph = vks[:,1], vks[:,2], vks[:,3]
    v1 = sth*cph*vr + r*cth*cph*vth - r*sth*sph*vph
    v2 = sth*sph*vr + r*cth*sph*vth + r*sth*cph*vph
    v3 = cth*vr - r*sth*vth
    return v1, v2, v3


def load(gdsc):
    xi = np.empty((gdsc.shape[0]//16, 4))
    xi[:,0] = gdsc[::16]
    xi[:,1] = gdsc[1::16]
    xi[:,2] = gdsc[2::16]
    xi[:,3] = gdsc[3::16]

    ki = np.empty((gdsc.shape[0]//16, 4))
    ki[:,0] = gdsc[4::16]
    ki[:,1] = gdsc[5::16]
    ki[:,2] = gdsc[6::16]
    ki[:,3] = gdsc[7::16]

    xf = np.empty((gdsc.shape[0]//16, 4))
    xf[:,0] = gdsc[8::16]
    xf[:,1] = gdsc[9::16]
    xf[:,2] = gdsc[10::16]
    xf[:,3] = gdsc[11::16]

    kf = np.empty((gdsc.shape[0]//16, 4))
    kf[:,0] = gdsc[12::16]
    kf[:,1] = gdsc[13::16]
    kf[:,2] = gdsc[14::16]
    kf[:,3] = gdsc[15::16]

    # convert everything to kerr schild
    ki = convertMKSvectotKS(xmks=xi, v=ki)
    kf = convertMKSvectotKS(xmks=xf, v=kf)
    xi = convertMKScoordtoKS(xi)
    xf = convertMKScoordtoKS(xf)

    return xi, ki, xf, kf

# plot to show the wave-vectors that escape and ones that don't
def plotquiver(xi, ki, xf, kf, theta, skip=10):
    # skip the first n points
    xi, ki, xf, kf = xi[::skip], ki[::skip], xf[::skip], kf[::skip]

    i_horizon = (xf[:,1] < 10)
    i_exit = (xf[:,1] > 100)
    xi = xi[i_exit, :]
    ki = ki[i_exit, :]
    xf = xf[i_exit, :]
    kf = kf[i_exit, :]
    
    # prepare for a quiver plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,10), subplot_kw={'projection': '3d'})
    # plot the initial geodesics only for now

    i_early = (xf[:,0] < 9975)
    i_late = (xf[:,0] > 1025)

    xcpy, kcpy = np.copy(xi), np.copy(ki)

    xi, ki = xi[i_early, :], ki[i_early, :]
    X = xi[:,1] * np.sin(xi[:,2]) * np.cos(xi[:,3])
    Y = xi[:,1] * np.sin(xi[:,2]) * np.sin(xi[:,3])
    Z = xi[:,1] * np.cos(xi[:,2])

    U, V, W = convertKSvectortoCART(xi, ki)
    axes.quiver(X,Y,Z,U,V,W, color="blue", label="early", alpha=1, zorder=10)

    xi, ki = xcpy[i_late, :], kcpy[i_late, :]
    X = xi[:,1] * np.sin(xi[:,2]) * np.cos(xi[:,3])
    Y = xi[:,1] * np.sin(xi[:,2]) * np.sin(xi[:,3])
    Z = xi[:,1] * np.cos(xi[:,2])

    U, V, W = convertKSvectortoCART(xi, ki)
    axes.quiver(X,Y,Z,U,V,W, color="red", label="late", zorder=0, alpha=0.3)

    axes.set_xlim((-8,8))
    axes.set_ylim((-8,8))
    axes.set_zlim((-2,2))
    plt.legend()
    plt.savefig("../../figs/wavevectors_theta_%f.png" %theta, bbox_inches="tight", dpi=150)
    plt.close()
    return


def plothist(xi, ki, xf, kf, theta):
    i_horizon = (xf[:,1] < 10)
    i_exit = (xf[:,1] > 100)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
    
    binst = np.linspace(1.e4-1.e2,1e4+1.e2,100)
    binsR = np.linspace(0,10,50)
    binsTH = np.linspace(0,np.pi,100)
    binsPH = np.linspace(-np.pi,np.pi,100)

    density = True
    # t plot
    axes[0,0].hist(xf[:,0], bins=binst, edgecolor="k", density=density, label="final", color="red", alpha=0.5)
    axes[0,0].set_xlabel("$\\Delta t$")
    axes[0,0].set_title("Recorded Time Coordinate")

    # r plot
    axes[0,1].hist(xi[i_horizon,1], bins=binsR, edgecolor="k", density=density, label="(trapped)", color="blue")
    axes[0,1].hist(xi[i_exit,1], bins=binsR, edgecolor="k", density=density, label="(escape)", color="green")
    axes[0,1].set_xlabel("$r_{i}$")
    axes[0,1].set_title("Radial Position")

    # theta plot
    axes[1,0].hist(xi[i_horizon,2], edgecolor="k", bins=binsTH, density=density, label="initial (trapped)", color="blue")
    axes[1,0].hist(xi[i_exit,2], edgecolor="k", bins=binsTH, density=density, label="initial (escape)", color="green")
    axes[1,0].hist(xf[:,2], edgecolor="k", bins=binsTH, density=density, label="final", color="red", alpha=0.5)
    axes[1,0].set_xlabel("$\\theta$")
    axes[1,0].set_title("Initial and Final Latitudes")

    #phi plot
    axes[1,1].hist(xi[i_horizon,3], edgecolor="k", bins=binsPH, density=density, label="initial (trapped)", color="blue")
    axes[1,1].hist(xi[i_exit,3], edgecolor="k", bins=binsPH, density=density, label="initial (escape)", color="green")
    axes[1,1].hist(xf[:,3], edgecolor="k", bins=binsPH, density=density, label="final", color="red", alpha=0.5)
    axes[1,1].set_xlabel("$\\Phi$")
    axes[1,1].set_title("Initial and Final Longitudes")


    for ax in axes.ravel(): ax.legend(); ax.set_ylabel("normalized counts")
    plt.suptitle("Angle between $b^{\mu}$ and $k^{\mu}$ is between $|\\Delta \\theta_{b-k}| = %.2f^{\circ}$" % (180*theta/np.pi))
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("../../figs/geodesic_histograms_theta_%f.png" %theta, 
                bbox_inches="tight", dpi=150)
    plt.close()

    # Now plot the the photon for early and late photons
    xi, ki, xf, kf = xi[i_exit], ki[i_exit], xf[i_exit], kf[i_exit]
    i_early = (xf[:,0] < 9975) & (xf[:,1] > 100)
    i_late = (xf[:,0] > 1035) & (xf[:,1] > 100)

    # save the data in a numpy array to integrate the geodesics
    gdsc_output_early = np.empty(8 * i_early.sum(), dtype=float)
    gdsc_output_late = np.empty(8 * i_late.sum(), dtype=float)

    gdsc_output_early[::8] = xi[i_early, 0]
    gdsc_output_early[1::8] = xi[i_early, 1]
    gdsc_output_early[2::8] = xi[i_early, 2]
    gdsc_output_early[3::8] = xi[i_early, 3]
    gdsc_output_early[4::8] = ki[i_early, 0]
    gdsc_output_early[5::8] = ki[i_early, 1]
    gdsc_output_early[6::8] = ki[i_early, 2]
    gdsc_output_early[7::8] = ki[i_early, 3]

    gdsc_output_late[::8] = xi[i_late, 0]
    gdsc_output_late[1::8] = xi[i_late, 1]
    gdsc_output_late[2::8] = xi[i_late, 2]
    gdsc_output_late[3::8] = xi[i_late, 3]
    gdsc_output_late[4::8] = ki[i_late, 0]
    gdsc_output_late[5::8] = ki[i_late, 1]
    gdsc_output_late[6::8] = ki[i_late, 2]
    gdsc_output_late[7::8] = ki[i_late, 3]

    print(gdsc_output_early)
    np.save("/home/siddhant/scratch/TeVlightcurve/geodesics/times/early_%f" %theta, gdsc_output_early)
    np.save("/home/siddhant/scratch/TeVlightcurve/geodesics/times/late_%f" % theta, gdsc_output_late)

    kbins = 100
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    # r plot
    axes[0,1].hist(ki[i_early,1], bins=kbins, edgecolor="k", density=density, label="(early)", color="blue")
    axes[0,1].hist(ki[i_late,1], bins=kbins, edgecolor="k", density=density, label="(late)", color="red", alpha=0.5)
    axes[0,1].set_xlabel("$k_{r}$")
    axes[0,1].set_title("Radial Intial Wavevector")

    # theta plot
    axes[1,0].hist(ki[i_early,2], edgecolor="k", bins=kbins, density=density, label="(early)", color="blue")
    axes[1,0].hist(ki[i_late,2], edgecolor="k", bins=kbins, density=density, label="(late)", color="red", alpha=0.5)
    axes[1,0].set_xlabel("$k_{\\theta}$")
    axes[1,0].set_title("Polar Initial Wavevector")

    #phi plot
    axes[1,1].hist(ki[i_early,3], edgecolor="k", bins=kbins, density=density, label="(early)", color="blue")
    axes[1,1].hist(ki[i_late,3], edgecolor="k", bins=kbins, density=density, label="(late)", color="red", alpha=0.5)
    axes[1,1].set_xlabel("$k_{\\Phi}$")
    axes[1,1].set_title("Azimuthal Initial Wavevector")


    for ax in axes.ravel(): ax.legend(); ax.set_ylabel("normalized counts")
    plt.suptitle("Angle between $b^{\mu}$ and $k^{\mu}$ is between $|\\Delta \\theta_{b-k}| = %.2f^{\circ}$" % (180*theta/np.pi))
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("../../figs/wavevector_histograms_theta_%f.png" %theta, 
                bbox_inches="tight", dpi=150)
    plt.close()

def plot_geodesics(xi, ki, xf, kf):
    # get the relevant indices and number of geodesics
    i_early = (xf[:,0] < 9975) & (xf[:,1] > 100)
    i_late = (xf[:,0] > 1035) & (xf[:,1] > 100)
    ngdsc_early = i_early.sum(); ngdsc_late = i_late.sum()

    # allocate arrays
    size = 200
    gdsc_data_early = integrator_cython.GetGeodesicArrayCython(xi[i_early], ki[i_early], size)
    gdsc_data_late = integrator_cython.GetGeodesicArrayCython(xi[i_late], ki[i_late], size)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    axes[0,0].loglog(gdsc_data_early[:,::10], gdsc_data_early[:,1::10], '.', color="red", alpha=0.1)
    axes[0,0].set_title("Radial Positions and Times"); axes[0,0].set_xlabel("Times"); axes[0,0].set_ylabel("Radial Position")
    axes[0,0].set_ylim((1, 1.e4))

    axes[0,1].semilogx(gdsc_data_early[:,::10], gdsc_data_early[:,2::10], '.', color="blue", alpha=0.1)
    axes[0,1].set_title("Latitude and Times"); axes[0,1].set_xlabel("Times"); axes[0,1].set_ylabel("Latitude")


    axes[1,0].semilogx(gdsc_data_early[:,::10], gdsc_data_early[:,3::10], '.', color="green", alpha=0.1)
    axes[1,0].set_title("Longitude and Times"); axes[1,0].set_xlabel("Times"); axes[1,0].set_ylabel("Longitude")

    axes[1,1].semilogx(gdsc_data_early[:,::10], gdsc_data_early[:,9::10]/gdsc_data_early[:,9].reshape(-1,1), '.', color="purple", alpha=0.1)
    axes[1,1].set_title("$\Delta$ Carter's constant and Times"); axes[1,1].set_xlabel("Times"); axes[1,1].set_ylabel("Carter's constant")

    for ax in axes.ravel(): ax.set_xlim((1, 1.e4))

    plt.savefig("early_geodesic_data.png")
    plt.close()

    # compare the final distributions with the C integrator
    binst = np.linspace(1.e4-1.e2,1e4+1.e2,100)
    binsR = np.linspace(0,10,50)
    binsTH = np.linspace(0,np.pi,100)
    binsPH = np.linspace(-np.pi,np.pi,100)
    density=True

    i_exit = (xf[:,1] > 100)
    size = 100
    gdscs = integrator_cython.GetGeodesicArrayCython(xi[i_exit], ki[i_exit], size)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

     # t plot
    axes[0,0].hist(xf[:,0], bins=binst, edgecolor="k", density=density, label="C integrate", color="blue", alpha=0.5)
    axes[0,0].hist(gdscs[:,-10], bins=binst, edgecolor="k", density=density, label="Cython", color="red", alpha=0.5)
    #axes[0,0].hist(gdscs[:,-9], bins=binst, edgecolor="k", density=density, label="Cython (r)", color="blue", alpha=0.5)
    
    axes[0,0].set_xlabel("$\\Delta t$")
    axes[0,0].set_title("Recorded Time Coordinate")

    # r plot
    axes[0,1].hist(gdscs[:,-9], bins=binst, edgecolor="k", density=density, label="Cython", color="red", alpha=0.5)
    axes[0,1].set_xlabel("$r_{i}$")
    axes[0,1].set_title("Radial Position")

    # theta plot
    axes[1,0].hist(gdscs[:,-8], bins=binsTH, edgecolor="k", density=density, label="Cython", color="red", alpha=0.5)
    axes[1,0].hist(xi[i_exit,2], edgecolor="k", bins=binsTH, density=density, label="initial", color="green")
    axes[1,0].hist(xf[:,2], edgecolor="k", bins=binsTH, density=density, label="C integrate", color="blue", alpha=0.5)
    axes[1,0].set_xlabel("$\\theta$")
    axes[1,0].set_title("Initial and Final Latitudes")

    #phi plot
    axes[1,1].hist(gdscs[:,-7], bins=binsPH, edgecolor="k", density=density, label="Cython", color="red", alpha=0.5)
    axes[1,1].hist(xi[i_exit,3], edgecolor="k", bins=binsPH, density=density, label="initial", color="green")
    axes[1,1].hist(xf[:,3], edgecolor="k", bins=binsPH, density=density, label="C integrate", color="blue", alpha=0.5)
    axes[1,1].set_xlabel("$\\Phi$")
    axes[1,1].set_title("Initial and Final Longitudes")

    for ax in axes.ravel(): ax.legend()

    plt.savefig("final_comparisions.png")
    plt.close()

def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def plot3Dgeodesics(xi, ki, xf, kf):
    i_exit = (xf[:,1] > 100)
    xi, ki = xi[i_exit], ki[i_exit]
    points_per_geodesic = 200
    epsilon = 1.e-2
    geodesics = integrator_cython.GetGeodesicArrayFast(xi[::1], ki[::1], points_per_geodesic, epsilon)

    i_exit = np.abs(geodesics[:,-9] - 10000) < 1
    geodesics = geodesics[i_exit,:]

    rks, theta, phi = geodesics[:, 1::10], geodesics[:, 2::10], geodesics[:, 3::10]
    all_times = geodesics[:, ::10]
    tend = geodesics[:, -10]
    xgdsc = rks * np.sin(theta) * np.cos(phi)
    ygdsc = rks * np.sin(theta) * np.sin(phi)
    zgdsc = rks * np.cos(theta)

    import pyevtk
    i_early = all_times[:, -1] < 9000
    i_late = all_times[:, -1] < 9000

    for ind in [i_early, i_late]:
        pyevtk.hl.pointsToVTK("gdscs_early", x=xgdsc[i_early].ravel(), y=ygdsc[i_early].ravel(), 
                            z=zgdsc[i_early].ravel(), data={"t":all_times[i_early].ravel()})
        pyevtk.hl.pointsToVTK("gdscs_late", x=xgdsc[i_late].ravel(), y=ygdsc[i_late].ravel(), 
                            z=zgdsc[i_late].ravel(), data={"t":all_times[i_late].ravel()})
    

    plt.figure()
    for r,t,th,ph in zip(rks, all_times,theta,phi):
        if t[-1] < 9000:
            plt.plot(t[:50],ph[:50], ls = '-', color="blue", alpha=1)
        if t[-1] > 10100:
            plt.plot(t[:50], ph[:50], ls = '-',  color="red", alpha=1)

    plt.xlim((0, 10))
    plt.savefig("times.png")
    plt.close()
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_box_aspect([1,1,1])
    
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    r = 1.34
    x = r * np.cos(u)*np.sin(v)
    y = r * np.sin(u)*np.sin(v)
    z = r * np.cos(v)
    ax.plot_wireframe(x, y, z, color="k") 

    for t, x, y, z in zip(all_times, xgdsc, ygdsc, zgdsc):
        if t[-1] < 9000:
            ax.plot(x, y, z, "-", markersize=1, color="blue")
        if t[-1] > 10100:
            ax.plot(x, y, z, "-", markersize=1, color="red")
    #plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.set_xlim((-6,6))
    ax.set_ylim((-6,6))
    ax.set_zlim((-6,6))
    # Hide grid lines
    ax.grid(False)
    plt.axis("off")
    plt.savefig("geodesic_3d.png")
    plt.close()

'''
    Convergence test for the geodesics
'''
def convergence_test(xi, ki, xf, kf):
    ind = (xf[:, 1] > 100)
    xi, ki = xi[ind], ki[ind]

    epsilon = [1.e-1, 1.e-2, 1.e-3, 1.e-4]
    colors = ["blue", "green", "yellow", "orange", "red"]
    labels = ["$10^{%d}$" % i for i in range(-1,-5,-1)]
    points_per_geodesic = 100

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    # compare the final distributions with the C integrator
    binst = np.linspace(1.e4-1.e2,1e4+1.e2,100)
    binsR = np.linspace(0,10,50)
    binsTH = np.linspace(0,np.pi,100)
    binsPH = np.linspace(-np.pi,np.pi,100)
    density=True

    xicpy, kicpy = np.copy(xi), np.copy(ki)
    for i, eps in enumerate(epsilon):
        gdscs = integrator_cython.GetGeodesicArrayCython(xi[::1], ki[::1], points_per_geodesic, eps)
        axes[0,0].hist(gdscs[:,-10], bins=binst, edgecolor="k", density=density, label=labels[i], color=colors[i], alpha=0.5)
        axes[0,1].hist(gdscs[:,-9], bins=binst, edgecolor="k", density=density, label=labels[i],  color=colors[i], alpha=0.5)
        axes[1,0].hist(gdscs[:,-8], bins=binsTH, edgecolor="k", density=density, label=labels[i],  color=colors[i], alpha=0.5)
        axes[1,1].hist(gdscs[:,-7], bins=binsPH, edgecolor="k", density=density, label=labels[i], color=colors[i], alpha=0.5)
        xi,ki = np.copy(xicpy), np.copy(kicpy)

    for ax in axes.ravel(): ax.legend(); ax.set_ylabel("cts (normalized)")
    axes[0,0].set_title("Arrival Kerr Schild Time")
    axes[0,1].set_title("Arrival Kerr Schild Radius")
    axes[1,0].set_title("Arrival Kerr Schild Latitude")
    axes[1,1].set_title("Arrival Kerr Schild Longitude")



    plt.suptitle("Convergence Test")
    plt.savefig("convergence.png")

'''
    Plot histograms of early and late photons
'''
def PlotEarlyLateHist(xi, ki, xf, kf):
    ind = (xf[:, 1] > 0)
    xi, ki = xi[ind], ki[ind]
    epsilon = 1e-2
    points_per_geodesic = 100
    geodesics = integrator_cython.GetGeodesicArrayCythonFast(xi[::1], ki[::1], points_per_geodesic, epsilon)
    print(geodesics)
    density = True

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,10))

    r_f = geodesics[:, -9]

    i_late = np.logical_and((geodesics[:, -10] > 10050) , (r_f > 9.99e3))
    i_early = np.logical_and((geodesics[:, -10] < 10100) , (r_f > 9.99e3))

    axes[0,0].set_title("Intial Radius")
    axes[0,1].set_title("Intial Latitude")
    axes[1,0].set_title("Intial $k_{\\phi}$")
    axes[1,1].set_title("Intial $k_{r}$")
    axes[2,0].set_title("Final $\Delta \\phi$")
    axes[2,1].set_title("Final $t$")

    gdscs = geodesics[i_late]
    r_i = gdscs[:, 1]
    th_i = gdscs[:, 2]
    ph_i = gdscs[:, 3]
    kt_i = gdscs[:,4]
    kph_i = gdscs[:, 7]
    kr_i = gdscs[:, 5]
    kth_i = gdscs[:, 6]
    ph_f = gdscs[:, -7]
    t_f = gdscs[:, -10]

    # Print the average values of the two distrubutions
    r_av, th_av, ph_av = r_i.mean(), th_i.mean(), ph_i.mean()
    kt_av, kr_av, kth_av, kph_av = kt_i.mean(), kr_i.mean(), kth_i.mean(), kph_i.mean()
    print("Late")
    print("mean vectors of late distribution: ", r_av, th_av, ph_av)
    print("mean wave vectors of late distribution: ", kt_av, kr_av, kth_av, kph_av)
    X_late = np.array([0, r_i[0], th_i[0], ph_i[0]]); K_late = np.array([kt_i[0], kr_i[0], kth_i[0], kph_i[0]])

    # Inital Radius
    axes[0,0].hist(r_i, bins=np.linspace(0,10,100), edgecolor="k", density=density, label="late", color="red", alpha=0.5)
    # Inital Latitude
    axes[0,1].hist(th_i, bins=np.linspace(0, np.pi, 100), edgecolor="k", density=density, label="late",  color="red", alpha=0.5)
    # Inital kphi
    axes[1,0].hist(kph_i, bins=np.linspace(-2, 2, 100), edgecolor="k", density=density, label="late",  color="red", alpha=0.5)
    # Inital kr
    axes[1,1].hist(kr_i, bins=np.linspace(-2, 2, 100), edgecolor="k", density=density, label="late", color="red", alpha=0.5)
    # final phi
    axes[2,0].hist(ph_f, bins=np.linspace(-np.pi, np.pi, 100), edgecolor="k", density=density, label="late",  color="red", alpha=0.5)
    # final t
    axes[2,1].hist(t_f, bins=np.linspace(1.e4-5.e2, 1.e4+5.e2, 100), edgecolor="k", density=density, label="late",  color="red", alpha=0.5)


    gdscs = geodesics[i_early]

    print(i_early.sum(), i_late.sum())
    r_i = gdscs[:, 1]
    th_i = gdscs[:, 2]
    ph_i = gdscs[:, 3]
    kt_i = gdscs[:,4]
    kph_i = gdscs[:, 7]
    kr_i = gdscs[:, 5]
    kth_i = gdscs[:, 6]
    ph_f = gdscs[:, -7]
    t_f = gdscs[:, -10]

    # Print the average values of the two distrubutions
    r_av, th_av, ph_av = r_i.mean(), th_i.mean(), ph_i.mean()
    kt_av, kr_av, kth_av, kph_av = kt_i.mean(), kr_i.mean(), kth_i.mean(), kph_i.mean()
    print("Early")
    print("mean vectors of late distribution: ", r_av, th_av, ph_av)
    print("mean wave vectors of late distribution: ", kt_av, kr_av, kth_av, kph_av)
    X_early = np.array([0, r_i[0], th_i[0], ph_i[0]]); K_early = np.array([kt_i[0], kr_i[0], kth_i[0], kph_i[0]])

    
    # Inital Radius
    axes[0,0].hist(r_i, bins=np.linspace(0,10,100), edgecolor="k", density=density, label="early", color="blue", alpha=0.5)
    # Inital Latitude
    axes[0,1].hist(th_i, bins=np.linspace(0, np.pi, 100), edgecolor="k", density=density, label="early",  color="blue", alpha=0.5)
    # Inital kphi
    axes[1,0].hist(kph_i, bins=np.linspace(-2, 2, 100), edgecolor="k", density=density, label="early",  color="blue", alpha=0.5)
    # Inital kr
    axes[1,1].hist(kr_i, bins=np.linspace(-2, 2, 100), edgecolor="k", density=density, label="early", color="blue", alpha=0.5)
    # final phi
    axes[2,0].hist(ph_f, bins=np.linspace(-np.pi, np.pi, 100), edgecolor="k", density=density, label="early",  color="blue", alpha=0.5)
    # final t
    axes[2,1].hist(t_f, bins=np.linspace(1.e4-5.e2, 1.e4+5.e2, 100), edgecolor="k", density=density, label="early",  color="blue", alpha=0.5)


    # geodesic test
    gdsc_early = integrator_cython.GetGeodesicArrayCythonFast(X_early.reshape(1,-1), K_early.reshape(1, -1), points_per_geodesic, epsilon)
    gdsc_late = integrator_cython.GetGeodesicArrayCythonFast(X_late.reshape(1,-1), K_late.reshape(1, -1), points_per_geodesic, epsilon)

    print(gdsc_early[:,9::10])
    print(gdsc_late[:,9::10])

    plt.savefig("histograms_early_late.png")


for th in theta[:1]:
    fname = DATA_DIR + ("%1.6f" + ".npy") % th
    geodesics = np.load(fname)
    xi, ki, xf, kf = load(geodesics)
    #plot3Dgeodesics(xi, ki, xf, kf)
    #convergence_test(xi, ki, xf, kf)
    PlotEarlyLateHist(xi, ki, xf, kf)
    #plotquiver(xi, ki, xf, kf, th)