import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import integrator_cython
import time

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.size'] = 15
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
mpl.rcParams['figure.dpi']=180


X = np.random.rand(100, 4)
K = np.random.rand(100, 4)

points_per_geodesic = 2
epsilon = 5.e-3

OUTDIR = "/home/siddhant/scratch/TeVlightcurve/geodesics/"
geodesics_per_point = [1, 10,100, 1000]
angles = [1.570796]#,0.0001]



colors = ["blue", "green", "yellow", "orange", "red"]
labels = ["$%d$" % i for i in geodesics_per_point]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

# compare the final distributions with the C integrator
binst = np.linspace(1.e4-1.e2,1e4+1.e2,100)
binsR = np.linspace(0,10,50)
binsTH = np.linspace(0,np.pi,180)
binsPH = np.linspace(-np.pi,np.pi,360)
density=True

for i, GperP in enumerate(geodesics_per_point):
    for angle in angles:
        fname = OUTDIR + "1912_reduced_DeltaTheta_%.6f_GeodesicsPerPoint_%d.txt" % (angle, GperP)
        data = np.genfromtxt(fname); print(data.shape)
        xi = data[:, :4]
        ki = data[:, 4:]

        xicpy, kicpy = np.copy(xi), np.copy(ki)

        points_per_geodesic = 2

        gdscs = integrator_cython.GetGeodesicArrayCythonFast(np.copy(xi), np.copy(ki), points_per_geodesic, epsilon)
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