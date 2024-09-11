import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import time
import os
import gc

basedir = os.path.abspath(os.getcwd())
src_dir = os.path.abspath(os.path.join(basedir, ".."))

x = np.load(src_dir + "/arrays_saved/x0.npy")
y = np.load(src_dir + "/arrays_saved/x1.npy")
t = np.load(src_dir + "/arrays_saved/time_evol/t.npy")

X, Y = np.meshgrid(x, y, indexing="ij")

if not os.path.exists(src_dir + "/imgs/time_evol/psi_x"):
    os.makedirs(src_dir + "/imgs/time_evol/psi_x")

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), layout="constrained")

for i in range(len(t)):
    ts = time.time()
    psi = np.load(src_dir + f"/arrays_saved/time_evol/psi_x/psi_t_{i}.npy")
    axs.set(
        aspect="equal",
        title=f"step={i} of {len(t)}; t = {t[i]:.{5}f} a.u.",
    )
    b = axs.pcolormesh(
        X,
        Y,
        np.abs(psi) ** 2,
        cmap=cm.jet,
        shading="auto",
        vmax=1e-7,
    )
    cb = plt.colorbar(b, ax=axs)
    fig.savefig(src_dir + f"/imgs/time_evol/psi_x/psi_t_{i}.png")
    axs.clear()
    cb.remove()
    gc.collect()
    print(f"step {i} of {len(t)}; time of step = {(time.time()-ts):.{5}f}")
