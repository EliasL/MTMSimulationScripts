import manim as M
from manim import WHITE, BLUE, GREEN, YELLOW, DEGREES
import numpy as np

from contiPotential import numericContiPotential

phi, _, _ = numericContiPotential()


# Custom Scene to display the surface
class SurfacePlot3D(M.ThreeDScene):
    def construct(self):
        # Setup the axes
        axes = M.ThreeDAxes()

        # Define the surface using the matrices of X, Y, Z
        def parametric_surface(u, v):
            v = np.where(v == 0, np.nan, v)
            C12 = u / v
            C22 = 1 / v
            C11 = (1 + C12**2) / C22
            # z = phi(u, v, 1, -0.25, 4, 1)

            # Precompute some common terms used in a, b, c12, c22, and c11 calculations
            denominator = u**2 - 2 * u + v**2 + 1
            a = (2 * v) / denominator
            b = -(u**2 + v**2 - 1) / denominator

            # Avoid division by zero or near-zero by masking those values in b
            safe_b = np.where(b == 0, np.nan, b)

            # Calculate c12, c22, and c11
            C12 = a / safe_b
            C22 = 1 / safe_b
            C11 = (1 + C12**2) / C22

            return np.nan_to_num(np.array([C11, C12, C22]))

        # Create the surface using the parametric equation
        surface = M.Surface(
            lambda u, v: parametric_surface(u, v),
            u_range=[-0.5, 0.5],  # Adjust according to your X and Y ranges
            v_range=[-0.5, 0.5],  # Adjust according to your X and Y ranges
            resolution=(50, 50),  # Adjust for higher resolution if needed
        )

        # Surface color and opacity
        surface.set_style(fill_opacity=0.8, stroke_color=WHITE)
        surface.set_fill_by_value(axes=axes, colors=[BLUE, GREEN, YELLOW])

        # Set initial camera orientation to view the surface
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES, zoom=0.1)

        # Add axes and surface to the scene
        self.add(axes, surface)

        # Begin ambient camera rotation with a slower rate
        self.begin_ambient_camera_rotation(rate=1)  # Adjust rate for desired speed

        # Optional: Keep the scene running for a certain time to observe the rotation
        self.wait(2)  # Wait for 20 seconds while the camera rotates

        # If you want to stop the rotation after some time, you can use:
        # self.stop_ambient_camera_rotation()

        # Enter interactive mode (if needed)
        self.interactive_embed()


# To render the scene
if __name__ == "__main__":
    scene = SurfacePlot3D()
    scene.render()
