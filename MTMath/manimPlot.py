import manim as M
from contiPotential import numericContiPotential
import numpy as np

phi, _, _ = numericContiPotential()


class FunctionSlicesThroughCube(M.ThreeDScene):
    def func(self, x, y, z):
        # Define your function here
        return phi(
            x,
            y,
            z,
        )

    def construct(self):
        # Set up 3D axes
        axes = M.ThreeDAxes(
            x_range=[0, 1, 0.1],
            y_range=[0, 1, 0.1],
            z_range=[0, 1, 0.1],
        )

        # Create a 3D volume surface for the function
        surface = M.Surface(
            lambda u, v: axes.c2p(u, v, self.func(u, v, 0.5)),
            u_range=[0, 1],
            v_range=[0, 1],
            fill_opacity=0.5,
            checkerboard_colors=[M.BLUE_D, M.BLUE_E],
            resolution=(20, 20),
        )

        # Add surface to the scene
        self.set_camera_orientation(phi=75 * M.DEGREES, theta=45 * M.DEGREES)
        self.add(axes, surface)

        # Create and animate slices along the z-axis
        for z in np.linspace(0, 1, 10):
            # Each slice represented by a new plane at z
            slice_plane = M.Surface(
                lambda u, v: axes.c2p(u, v, z),
                u_range=[0, 1],
                v_range=[0, 1],
                fill_opacity=0.2,
                color=M.YELLOW,
                resolution=(10, 10),
            )

            # Show each slice and then fade it out
            self.play(M.FadeIn(slice_plane), run_time=0.3)
            self.play(M.FadeOut(slice_plane), run_time=0.3)

        # Add camera rotation for perspective
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(2)
        self.stop_ambient_camera_rotation()


v = FunctionSlicesThroughCube()
v.construct()
