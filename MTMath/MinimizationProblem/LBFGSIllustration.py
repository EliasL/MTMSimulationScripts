import numpy as np
import sympy as sp
import plotly.graph_objects as go

# Symbolic camel function and its derivatives
x_sym, y_sym = sp.symbols("x y")
camel_expr = (
    (4 - 2.1 * x_sym**2 + (x_sym**4) / 3) * x_sym**2
    + x_sym * y_sym
    + (-4 + 4 * y_sym**2) * y_sym**2
)
camel_func = sp.lambdify((x_sym, y_sym), camel_expr, "numpy")
grad_func = sp.lambdify(
    (x_sym, y_sym), [sp.diff(camel_expr, v) for v in (x_sym, y_sym)], "numpy"
)
hess_func = sp.lambdify(
    (x_sym, y_sym),
    [
        [sp.diff(g, v) for v in (x_sym, y_sym)]
        for g in [sp.diff(camel_expr, v) for v in (x_sym, y_sym)]
    ],
    "numpy",
)


def camel(x, y):
    return camel_func(x, y)


def grad_camel(x, y):
    return np.array(grad_func(x, y))


def hessian_camel(x, y):
    return np.array(hess_func(x, y))


def quadratic_approximation(X, Y, x0, y0):
    f0 = camel(x0, y0)
    grad = grad_camel(x0, y0)
    H = hessian_camel(x0, y0)
    dX, dY = X - x0, Y - y0
    approx = (
        f0
        + grad[0] * dX
        + grad[1] * dY
        + 0.5 * (H[0, 0] * dX**2 + 2 * H[0, 1] * dX * dY + H[1, 1] * dY**2)
    )
    approx[approx > 10] = np.nan
    return approx


# Grid and function values
x = np.linspace(-2, 2, 300)
y = np.linspace(-1, 1.3, 300)
X, Y = np.meshgrid(x, y)
Z = camel(X, Y)
x0, y0 = 0, 1.2
Z_approx = quadratic_approximation(X, Y, x0, y0)

# Plotly plot
fig = go.Figure()
fig.add_surface(z=Z, x=X, y=Y, colorscale="Viridis", name="Original")
fig.add_surface(z=Z_approx, x=X, y=Y, colorscale="RdBu", opacity=0.5, name="Approx")
fig.add_trace(
    go.Scatter3d(
        x=[x0],
        y=[y0],
        z=[camel(x0, y0)],
        mode="markers",
        marker=dict(size=5, color="red"),
        name="Expansion Point",
    )
)
fig.update_layout(
    title="Camel Function and Quadratic Approximation",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis=dict(title="f(x, y)", range=[np.nanmin(Z), 9]),
    ),
)
fig.show()

# ---- Plotly Animations for Video Export ----
import plotly.io as pio

# Video 1: rotating only the energy landscape
frames1 = []
for angle in np.linspace(0, 360, 60):
    frames1.append(
        go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=2 * np.cos(np.radians(angle)),
                        y=2 * np.sin(np.radians(angle)),
                        z=1.25,
                    )
                )
            )
        )
    )

fig1 = go.Figure(
    data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", name="Original")],
    frames=frames1,
)
fig1.update_layout(
    title="Rotating Energy Landscape",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis=dict(title="f(x, y)", range=[np.nanmin(Z), np.nanmax(Z)]),
        camera=dict(eye=dict(x=2, y=0, z=1.25)),
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play", method="animate", args=[None])],
        )
    ],
)
pio.write_html(fig1, "rotating_energy_landscape.html")

# Video 2: rotating energy + approximation
frames2 = []
for angle in np.linspace(0, 360, 60):
    frames2.append(
        go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=2 * np.cos(np.radians(angle)),
                        y=2 * np.sin(np.radians(angle)),
                        z=1.25,
                    )
                )
            )
        )
    )

fig2 = go.Figure(
    data=[
        go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", name="Original"),
        go.Surface(z=Z_approx, x=X, y=Y, colorscale="RdBu", opacity=0.5, name="Approx"),
        go.Scatter3d(
            x=[x0],
            y=[y0],
            z=[camel(x0, y0)],
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Expansion Point",
        ),
    ],
    frames=frames2,
)
fig2.update_layout(
    title="Rotating Energy Landscape and Approximation",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis=dict(title="f(x, y)", range=[np.nanmin(Z), np.nanmax(Z)]),
        camera=dict(eye=dict(x=2, y=0, z=1.25)),
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play", method="animate", args=[None])],
        )
    ],
)

pio.write_html(fig2, "rotating_energy_and_approximation.html")

# ---- Export Plotly Animations to MP4 ----
# Requires: pip install -U kaleido imageio[ffmpeg]
import imageio.v2 as imageio
from plotly.io import to_image
from tqdm import tqdm


def save_plotly_animation_as_mp4(fig, frames, filename, fps=20):
    images = []
    for frame in tqdm(frames, desc=f"Rendering {filename}"):
        fig.update(frames=[frame])
        fig.layout.update(frame.layout)
        img_bytes = to_image(fig, format="png", width=800, height=600, engine="kaleido")
        images.append(imageio.imread(img_bytes))
    imageio.mimsave(filename, images, fps=fps)


# Save both animations as MP4
save_plotly_animation_as_mp4(fig1, frames1, "rotating_energy_landscape.mp4")
save_plotly_animation_as_mp4(fig2, frames2, "rotating_energy_and_approximation.mp4")
