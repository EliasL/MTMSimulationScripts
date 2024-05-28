import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp


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

        P = (F * V).sum()  # dissipated power

        if P > 0:
            Npos = Npos + 1
            Nneg = 0
            if Npos > Ndelay:
                dt = min(dt * finc, dtmax)
                alpha = alpha * fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax: break
            if i > Ndelay:
                dt = max(dt * fdec, dtmin)
                alpha = alpha0
            x = x - 0.5 * dt * V
            V = np.zeros(x.shape)

        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * np.linalg.norm(V) / np.linalg.norm(F)
        x = x + dt * V
        F = -df(x, params)
        V = V + 0.5 * dt * F

        error = max(abs(F))
        if error < atol: break

        path.append(x.copy())

        if logoutput: print(f(x, params), error)

    del V, F
    return [x, f(x, params), i, np.array(path)]

# Define the hill with a hole function using sympy
X, Y = sp.symbols('X Y')
Z = -sp.exp(-0.001 * (X**2+(Y+10)**2))*20 - sp.exp(-0.1 * ((X-2)**2 + (Y+1)**2)) - sp.exp(-0.1 * ((X-3)**2 + (Y-10)**2))

# Calculate the gradient symbolically
grad_Z = [sp.diff(Z, var) for var in (X, Y)]

# Convert the symbolic gradient to a numerical function
f_func = sp.lambdify((X, Y), Z, 'numpy')
df_func = [sp.lambdify((X, Y), grad, 'numpy') for grad in grad_Z]

def f(x, params=None):
    X, Y = x
    return f_func(X, Y)

def df(x, params=None):
    X, Y = x
    return np.array([grad(X, Y) for grad in df_func])

LBFGS_path=[]
LBFGS_paths=[]
def callback(xk):
  LBFGS_path.append(xk.copy())

FIRE_paths = []

y_starts = [10, 8, 5, 2]
for y_start in y_starts:
  # Initial starting point
  x0 = np.array([-0.25, y_start])
  # Perform optimization
  result = optimize_fire2(x0, f, df, None)
  x_opt, f_opt, iterations, path = result
  FIRE_paths.append(np.array(path))


  x0 = np.array([-0.25, y_start])
  # Perform optimization using scipy's L-BFGS-B

  LBFGS_path.append(x0)
  result = minimize(f, x0, method='L-BFGS-B', jac=df, callback=callback)
  LBFGS_paths.append(np.array(LBFGS_path.copy()))
  LBFGS_path=[]

size=16
# Create a grid of points
x = np.linspace(-size, size, 100)
y = np.linspace(-size, size, 100)
X, Y = np.meshgrid(x, y)

# Define the hill with a hole function
Z = f((X,Y), None)

# Compute the derivatives
Zx, Zy = np.gradient(Z, x, y)
gradient_magnitude = np.sqrt(Zx**2 + Zy**2)

# Create the plots
fig, axs = plt.subplots(1, 1, figsize=(7, 6))

# Plot the filled contour plot of the hill with a hole
contour1 = axs.contourf(X, Y, Z, levels=50, cmap='viridis')
fig.colorbar(contour1, ax=axs, label='Height')
axs.set_title('Minimization path of the FIRE algorithm')
axs.set_xlabel('X-axis')
axs.set_ylabel('Y-axis')
# axs.set_xlim(-size, size)
# axs.set_ylim(-size, size)


# Plot the path of the optimization algorithm
for FIRE_path, LBFGS_path, start in zip(FIRE_paths, LBFGS_paths, y_starts):
  print(f"Fire: {len(FIRE_path)} LBGFS: {len(LBFGS_path)})")
  axs.plot(FIRE_path[:, 0], FIRE_path[:, 1], label=f'FIRE y0={start}')
  axs.plot(LBFGS_path[:, 0], LBFGS_path[:, 1], linestyle='--', label=f'LBFGS y0={start}')

  axs.scatter(FIRE_path[:, 0], FIRE_path[:, 1], label=f'FIRE y0={start}')
  axs.scatter(LBFGS_path[:, 0], LBFGS_path[:, 1], linestyle='--', label=f'LBFGS y0={start}')
axs.legend()


# Display the plots
plt.tight_layout()
plt.show()