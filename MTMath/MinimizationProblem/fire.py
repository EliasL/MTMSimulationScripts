import numpy as np
# fire_algorithm.py
# https://github.com/elvissoares/PyFIRE/blob/master/fire.py
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-05
# Updated: 2021-05-31

"==== FIRE: Fast Inertial Relaxation Engine ====="

" References: "
"- Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Gumbsch, P. (2006). Structural relaxation made simple. Physical Review Letters, 97(17), 1–4. https://doi.org/10.1103/PhysRevLett.97.170201"
"- Guénolé, J., Nöhring, W. G., Vaid, A., Houllé, F., Xie, Z., Prakash, A., & Bitzek, E. (2020). Assessment and optimization of the fast inertial relaxation engine (FIRE) for energy minimization in atomistic simulations and its implementation in LAMMPS. Computational Materials Science, 175. https://doi.org/10.1016/j.commatsci.2020.109584"


# Global variables for the FIRE algorithm
alpha0 = 0.1
Ndelay = 5
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000


def optimize_fire2(x0, f, df, params, atol=1e-4, dt=1, logoutput=False):
    error = 10 * atol
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha = alpha0
    Npos = 0
    Nneg = 0

    x = x0.copy()
    V = np.zeros(x.shape)
    F = -df(x, params)
    path = [x.copy()]

    for i in range(Nmax):
        P = (F * V).sum()
        if P > 0:
            Npos = Npos + 1
            Nneg = 0
            if Npos > Ndelay:
                dt = min(dt * finc, dtmax)
                alpha = alpha * fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax:
                break
            if i > Ndelay:
                dt = max(dt * fdec, dtmin)
                alpha = alpha0
            x = x - 0.5 * dt * V
            V = np.zeros(x.shape)
        # VelocityVerlet
        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * np.linalg.norm(V) / np.linalg.norm(F)
        x = x + dt * V
        F = -df(x, params)
        V = V + 0.5 * dt * F

        error = max(abs(F))
        if error < atol:
            break

        path.append(x.copy())

        if logoutput:
            print(f(x, params), error)

    del V, F
    return [x, f(x, params), i, np.array(path)]
