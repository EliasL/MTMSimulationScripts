from manim import *


# Custom Scene to display the surface
class SurfacePlot3D(ThreeDScene):
    def construct(self):
        # Setup the axes
        axes = ThreeDAxes()

        # Define the surface using the matrices of X, Y, Z
        # Assuming your X, Y, Z are functions of two parameters (u, v)
        def parametric_surface(u, v):
            # These are your mappings of (u,v) to X, Y, Z
            x = u  # Replace with your actual mapping from X matrix
            y = v  # Replace with your actual mapping from Y matrix
            z = u**2 + v**2  # Replace with actual Z matrix values

            return np.array([x, y, z])

        # Create the surface using the parametric equation
        surface = Surface(
            lambda u, v: parametric_surface(u, v),
            u_range=[-3, 3],  # Adjust according to your X and Y ranges
            v_range=[-3, 3],  # Adjust according to your X and Y ranges
            resolution=(21, 21),  # Adjust for higher resolution if needed
        )

        # Surface color and opacity
        surface.set_style(fill_opacity=0.8, stroke_color=WHITE)
        surface.set_fill_by_value(axes=axes, colors=[BLUE, GREEN, YELLOW])

        # Set camera orientation to view the surface
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Add axes and surface to the scene
        self.add(axes, surface)
        self.wait(2)


# To render the scene
if __name__ == "__main__":
    from manim import config

    config.media_width = "50%"  # Adjust the resolution of rendered video
    scene = SurfacePlot3D()
    scene.render()
