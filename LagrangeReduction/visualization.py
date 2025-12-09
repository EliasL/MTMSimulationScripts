import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QFontDatabase
import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QApplication
import matplotlib
from concurrent.futures import ThreadPoolExecutor
from .LagrangeReduction import (
    C2PoincareDisk,
    F2C,
    constrainDeterminant,
    generate_matrix,
    lagrange_reduction,
    lagrange_reduction_visualization,
)
from .vectorPair import VectorPair
from MTMath.plotEnergy import (
    generate_energy_grid,
    generate_cauchy_stress_grid,
    generate_piola_stress_grid,
    drawPoincareGrid,
)
from MTMath.contiPotential import ContiEnergy, SShear

# Suppress scientific notation in NumPy arrays
np.set_printoptions(suppress=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO Make point in the configuration space dragable,
# And super importantly, show all possible F configurations
# TODO Show caucy stress instead of energy in the configuration space


COOLWARM_LUT = (
    matplotlib.colormaps["coolwarm"](np.linspace(0, 1, 256))[:, :3] * 255
).astype(np.uint8)
VIRIDIS_LUT = (
    matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255
).astype(np.uint8)


class LagrangeReductionVisualization(QtWidgets.QWidget):
    from PyQt5 import QtCore

    energyComputed = pyqtSignal(object)  # carry the ndarray as a plain Python object

    def __init__(self):
        # Create two separate executors for quick and high-resolution updates
        self._energy_executor_quick = ThreadPoolExecutor(max_workers=1)
        self._energy_executor_highres = ThreadPoolExecutor(max_workers=1)
        super().__init__()
        self._target_version = 0
        self._heatmap_update_pending = False
        self._deferred_timer = QTimer(self)
        self._deferred_timer.setSingleShot(True)
        self._deferred_timer.timeout.connect(self._doDeferredUpdates)

        # Colors and line size
        self.background_line_color = np.array([100, 100, 100])
        self.handleColor = "#008B8B"
        self.reducedColor = "#FF6347"
        self.elasticReducedColor = "#0073FF"
        self.lineSize = 2
        self.markerSize = 15
        self.vectorWidth = 8

        # Default energy parameters
        self.currentBeta = -0.25
        self.volumetricEnergy = True
        self.energy_lim = [0, 0.37]
        self.energyFunc = ContiEnergy  # SuperSimple

        # Div
        self.showHistory = False
        self.showStress = True
        self.stress_type = "cauchy"  # "cauchy", "PK1" or "PK2" (PK=Piola-Kirchhoff)
        # "det", "trace","N1", "J2", "sqrtJ2", or "i,j" for components
        self.stress_mode = "trace"
        self.stressLim = (-0.2, 0.2)
        self.showCircles = True
        self.showRightOrth = False

        # Basic widget setup
        self.setWindowTitle("Lagrange reduction with Poincaré Disk")
        self.resize(1300, 650)

        # defines MainRow, LeftColumn and TableRow layouts (l_)
        self.initLayout()

        self.setLayout(self.l_MainCol)

        # Set up the views and plots
        self.setupLRView(self.l_WindowRow)  # Lagrange reduction
        self.setupGVView(self.l_WindowRow)  # Grid visualization
        # self.setupCSView(self.l_MainRow)  # Configuration space
        self.setupPoincareCSView(self.l_WindowRow)

        # Add markers
        self.mkMarkers()
        for plot in [self.PCS_plot]:
            plot.addItem(self.reduced_marker)
            plot.addItem(self.normal_marker)
            plot.addItem(self.elastic_reduced_marker)

        # Set up table
        self.setUpTables(self.l_MatrixRow)

        # Draw background elements
        self.drawBackground()

        # Set shearVelocity for controlling vectors with arrow keys
        self.shearVelocity = np.eye(2)

        # Timer elements for making animations
        self.time = 0
        self.timer = QTimer(self)

        # Connect events
        self.LR_plot.scene().sigMouseMoved.connect(self.mouseMove)
        self.GV_plot.scene().sigMouseMoved.connect(self.mouseMove)
        self.w_LR.keyPressEvent = self.keyPressEvent
        self.w_LR.keyReleaseEvent = self.keyReleaseEvent
        self.timer.timeout.connect(self.moveVector)
        self.timer.start(40)  # Every 20 milliseconds
        # Rotation state (Home + Left/Right)
        self.home_held = False
        self.alt_held = False
        self.meta_held = False
        self.rotation_step_small = np.pi / 180  # 1° per press
        self.rotation_step_large = 5 * np.pi / 180  # 5° when holding Shift
        self.energyComputed.connect(self.updateEnergyHeatmap, Qt.QueuedConnection)
        # Connect the signal once after setting up the plot
        self.GV_plot.getViewBox().sigRangeChanged.connect(self.onViewRangeChanged)

        # Hide Lagrange reduction by default
        self.elastic_reduced_marker.setVisible(
            not self.elastic_reduced_marker.isVisible()
        )
        self.LR_VP.setVisible(reduced=False)
        self.GV_VP.setVisible(reduced=False)
        self.reduced_marker.setVisible(False)

        self.show()

    def initLayout(self):
        """
        https://asciiflow.com/#/share/eJzlVEsKwjAQvUqZtSsF0V7Ajd3oNpuhjSVQo8RaqiKIaxdddNHz9STGX4mfGmksUiyBZiZ5L2%2FCy2yA45SCzZdB0IIAV1SADRsCERULNuME7HaLQCz%2F%2FW5XzlanTK8jZyGNQxkQyNNDnu6aORJCeJ7um1lCIpXf9BuWkBjAq2GTi2xF%2F956%2FsoWXucLpvpZB4J5VsQWSwzYGkP5VAr0EH2B3KePrA4NBXOrah1Rb%2Byez7nPj%2BfoUqsqq8EN%2FD1ravDeMgN4NWx2r99E%2FK9GpujXdhxtV%2FrChhcuKfxR6iht%2BJ7VQdlFYgWH3FNpPBapIeOT2QesNWjVOkzr4y9sKNPXRPtfSyKwhe0RUuICCA%3D%3D)        Approximate layout
        ┌─────────────────────────────────────────────────────────────┐
        │┌───────────────────────────────────────────────────────────┐│
        ││┌───────────────────┐┌──────────────────┐┌────────────────┐││
        │││                   ││                  ││                │││
        │││                   ││                  ││                │││
        │││Grid visualization ││  Lagrange        ││  Metric        │││
        │││                   ││  RedSction       ││  Space         │││
        │││                   ││                  ││                │││
        │││                   ││                  ││                │││
        │││                   ││                  ││                │││
        │││                   ││                  ││                │││
        ││└───────────────────┘└──────────────────┘└────────────────┘││
        │└───────────────────────────────────────────────────────────┘│
        │┌──────────┐┌─────────┐┌─────────┐┌─────────┐                │
        ││          ││         ││         ││         │                │
        ││  Matrix  ││   and   ││   div   ││  info   │                │
        ││          ││         ││         ││         │                │
        │└──────────┘└─────────┘└─────────┘└─────────┘                │
        └─────────────────────────────────────────────────────────────┘

        In words:
        - Column of two elements:
            - Row of three element:
                - Grid Visualization
                - Lagrange Reduction
                - Metric Space
            - Row of info
                - Matrixes and table stuff

        We now try to create this with the QtWidgets layouts
        """

        self.l_MainCol = QtWidgets.QVBoxLayout()
        self.l_WindowRow = QtWidgets.QHBoxLayout()
        self.l_MatrixRow = QtWidgets.QHBoxLayout()

        self.l_MainCol.addLayout(self.l_WindowRow)
        self.l_MainCol.addLayout(self.l_MatrixRow)

    def setupGVView(self, layout):
        # Grid visualization
        self.w_GV = pg.GraphicsLayoutWidget()

        layout.insertWidget(0, self.w_GV)

        self.GV_plot = self.w_GV.addPlot()
        self.GV_plot.setLabels(left="Y", bottom="X")
        self.GV_plot.setTitle("Grid visualization", **{"color": "#FFF", "size": "20pt"})
        self.GV_plot.setAspectLocked()
        # Set fixed margins for the ViewBox
        s = 2
        self.GV_plot.setRange(xRange=[-s, s], yRange=[-s, s])

        LR_grid = pg.GridItem()
        self.GV_plot.addItem(LR_grid)
        # Create scatter plot item
        self.scatter = pg.ScatterPlotItem()
        self.GV_plot.addItem(self.scatter)
        self.updateGVSpheres(init=True)

        # Copies VP
        self.GV_VP = VectorPair(
            self.GV_plot,
            colorS=self.reducedColor,
            colorH=self.handleColor,
            handelable=True,
            width=self.vectorWidth,
        )
        # self.GV_VP.setVisible(reduced=False)

    # Grid view
    def updateGVSpheres(self, init=False):
        if init:
            e1 = np.array([0, 1])
            e2 = np.array([1, 0])
        else:
            # Obtain the basis vectors from VectorPair
            e1 = self.VP.pos1()
            e2 = self.VP.pos2()
            # e1 = self.reduced_marker.pos()

        s = 5
        N = 100
        # Define the range for the grid
        x_range = np.arange(-N, N)  # Change these ranges as needed
        y_range = np.arange(-N, N)

        # Create the grid of combinations using numpy's broadcasting
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid = grid_x[..., np.newaxis] * e1 + grid_y[..., np.newaxis] * e2

        # Reshape the grid to a 2D array where each row is a point in the grid
        positions = grid.reshape(-1, 2)
        filtered_positions = positions[np.all(np.abs(positions) <= s, axis=1)]
        # Add points to scatter plot
        self.scatter.setData(pos=filtered_positions, size=10, symbol="o", brush="w")

    # Lagrange reduction view
    def setupLRView(self, layout):
        self.w_LR = pg.GraphicsLayoutWidget()

        layout.insertWidget(1, self.w_LR)

        self.LR_plot = self.w_LR.addPlot()
        self.LR_plot.setLabels(left="Y", bottom="X")
        self.LR_plot.setTitle("F and reduced F", **{"color": "#FFF", "size": "20pt"})
        self.LR_plot.setAspectLocked()
        s = 1.5
        self.LR_plot.setRange(xRange=[-s, s], yRange=[-s, s])

        # Set fixed margins for the ViewBox
        self.LR_plot.showAxes(False, size=(45, 35))

        LR_grid = pg.GridItem()
        self.LR_plot.addItem(LR_grid)

        self.LR_VP = VectorPair(
            self.LR_plot,
            colorS=self.reducedColor,
            colorH=self.handleColor,
            width=self.vectorWidth,
        )

        # Hide by default
        self.w_LR.setVisible(False)

    def setupPoincareCSView(self, layout):
        # For the second view configuration space (Poincaré Disk)
        self.w_PCS = pg.GraphicsLayoutWidget()
        layout.addWidget(self.w_PCS)

        self.PCS_plot = self.w_PCS.addPlot()
        nbs = "\u00a0"  # non-breaking-space
        self.PCS_plot.setLabels(
            left=f"← Large angle {nbs * 7} T(Length ratio and θ - π/2) {nbs * 7} Small angle →",
            bottom="T(Length ratio)",
        )
        self.PCS_plot.setTitle(
            "Configuration space in Poincaré Disk", **{"color": "#FFF", "size": "20pt"}
        )
        self.PCS_plot.setAspectLocked()
        s = 1
        self.PCS_plot.setRange(xRange=[-s, s], yRange=[-s, s])

        self.PCS_plot.showAxes(False)

    def mkMarkers(self):
        self.normal_marker = pg.ScatterPlotItem(
            pos=np.array([(0, 0)]),
            size=self.markerSize,
            brush=pg.mkBrush(self.handleColor),  # Fill color
            pen=pg.mkPen(color="white", width=2),  # Outline color and width
            zorder=5,
        )

        self.reduced_marker = pg.ScatterPlotItem(
            pos=np.array([(0, 0)]),
            size=self.markerSize,
            brush=pg.mkBrush(self.reducedColor),  # Fill color
            pen=pg.mkPen(color="white", width=2),  # Outline color and width
        )
        self.elastic_reduced_marker = pg.ScatterPlotItem(
            pos=np.array([(0, 0)]),
            size=self.markerSize,
            brush=pg.mkBrush(self.elasticReducedColor),  # Fill color
            pen=pg.mkPen(color="white", width=2),  # Outline color and width
        )

    def setUpTables(self, layout):
        # Create a frame to hold the matrix displays
        matrix_frame = QtWidgets.QFrame()
        matrix_frame.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Raised)
        matrix_frame.setLineWidth(1)
        frame_layout = QtWidgets.QHBoxLayout(matrix_frame)
        layout.addWidget(matrix_frame)

        # Create matrix displays
        matrices = [
            {"name": "F", "var_name": "F_display", "tooltip": "Deformation Gradient"},
            {
                "name": "C",
                "var_name": "C_display",
                "tooltip": "Right Cauchy-Green Tensor",
            },
            {
                "name": "C_R",
                "var_name": "C_R_display",
                "tooltip": "Reduced Right Cauchy-Green Tensor",
            },
            {
                "name": "M",
                "var_name": "M_display",
                "tooltip": "Lagrange reduction matrix",
            },
            {
                "name": "P",
                "var_name": "P_display",
                "tooltip": "First Piola-Kirchhoff stress tensor",
            },
        ]

        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        # Create custom widgets for each matrix
        for matrix in matrices:
            matrix_widget = QtWidgets.QGroupBox(matrix["name"])
            matrix_widget.setAlignment(Qt.AlignCenter)
            matrix_widget.setToolTip(matrix["tooltip"])
            matrix_layout = QtWidgets.QVBoxLayout(matrix_widget)

            # Create a QLabel for the matrix display
            matrix_label = QtWidgets.QLabel()
            matrix_label.setAlignment(Qt.AlignCenter)
            matrix_label.setStyleSheet("""
                font-size: 12pt;
                padding: 8px;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
            """)
            matrix_label.setFont(fixed_font)
            # Set initial text to identity matrix

            matrix_layout.addWidget(matrix_label)
            frame_layout.addWidget(matrix_widget)

            # Store the label in the class for later updates
            setattr(self, matrix["var_name"], matrix_label)

        # Create a QLabel for determinant and eigenvalues
        info_frame = QtWidgets.QFrame()
        info_layout = QtWidgets.QVBoxLayout(info_frame)

        # Determinant display
        self.det_label = QtWidgets.QLabel()
        self.det_label.setAlignment(Qt.AlignLeft)
        info_layout.addWidget(self.det_label)

        # Determinant display
        self.energy_label = QtWidgets.QLabel()
        self.energy_label.setAlignment(Qt.AlignLeft)
        self.det_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        info_layout.addWidget(self.energy_label)

        # Eigenvalues display (m values)
        self.m_label = QtWidgets.QLabel()
        self.m_label.setAlignment(Qt.AlignLeft)
        self.m_label.setStyleSheet("font-size: 11pt;")
        self.m_label.setFont(fixed_font)
        info_layout.addWidget(self.m_label)

        # Eigenvalues display (m values)
        self.angle_label = QtWidgets.QLabel()
        self.angle_label.setAlignment(Qt.AlignLeft)
        self.angle_label.setStyleSheet("font-size: 11pt;")
        self.angle_label.setFont(fixed_font)
        info_layout.addWidget(self.angle_label)

        # Add the info frame to the layout
        layout.addWidget(info_frame)
        self.updateInfoDisplay()

    # Add a method to update the display
    def updateInfoDisplay(
        self,
        F=np.eye(2),
        C=np.eye(2),
        C_R=np.eye(2),
        M=np.eye(2),
        P=np.eye(2),
        ms=[],
        m1=0,
        m2=0,
        m3=0,
    ):
        # Update F matrix display
        if F is not None:
            self.F_display.setText(
                f"{F[0, 0]: .2f}  {F[0, 1]: .2f}\n{F[1, 0]: .2f}  {F[1, 1]: .2f}"
            )
            # Calculate and display determinant
            det_F = np.linalg.det(F)
            self.det_label.setText(f"det(F) = {det_F:.2f}")
            if abs(det_F - 1) < 0.0001:
                # Set color to red while preserving other style settings:
                self.det_label.setStyleSheet(
                    "color: black; font-weight: bold; font-size: 11pt;"
                )
            else:
                # Set back to the default color (e.g., black)
                self.det_label.setStyleSheet(
                    "color: red; font-weight: bold; font-size: 11pt;"
                )

            E = self.energyFunc.energy_from_F(
                F,
                self.currentBeta,
                K=4 if self.volumetricEnergy else 0,
                zeroReference=True,
            )
            self.energy_label.setText(f"E = {E:.3f}")

        # Update C matrix display
        if C is not None:
            self.C_display.setText(
                f"{C[0, 0]: .2f}  {C[0, 1]: .2f}\n{C[1, 0]: .2f}  {C[1, 1]: .2f}"
            )

        # Update C_R matrix display
        if C_R is not None:
            self.C_R_display.setText(
                f"{C_R[0, 0]: .2f}  {C_R[0, 1]: .2f}\n{C_R[1, 0]: .2f}  {C_R[1, 1]: .2f}"
            )

        # Update M matrix display
        if M is not None:
            self.M_display.setText(
                f"{M[0, 0]: .2f}  {M[0, 1]: .2f}\n{M[1, 0]: .2f}  {M[1, 1]: .2f}"
            )
        # Update P matrix display
        if P is not None:
            self.P_display.setText(
                f"{P[0, 0]: .2f}  {P[0, 1]: .2f}\n{P[1, 0]: .2f}  {P[1, 1]: .2f}"
            )

        max_numb = 50  # Max number of numbers per line
        joined_ms = "".join(map(str, ms))
        ms_with_newlines = "\n".join(
            [joined_ms[i : i + max_numb] for i in range(0, len(joined_ms), max_numb)]
        )
        self.m_label.setText(f"m₁: {m1}  m₂: {m2}  m₃: {m3} \ns:{ms_with_newlines}")

        # Calculate and display angle
        # The diagonal of C gives the dot product of the basis vectors
        if C is not None:
            angle = np.arccos(C[0, 1] / (np.sqrt(C[0, 0]) * np.sqrt(C[1, 1])))
            degrees = np.degrees(angle)
            self.angle_label.setText(f"Angle (θ): {degrees:.2f}°")

    def _get_or_create_background_image(
        self, fName, generator_fn, force_generate=False
    ):
        # Ensure an image exists on disk at fName and return it as a pg.ImageItem.

        dirpath = os.path.dirname(fName)
        os.makedirs(dirpath, exist_ok=True)

        if force_generate or not os.path.isfile(fName):
            generator_fn()
            if not os.path.isfile(fName):
                print(f"NOT SAVED: {fName}")

        return self.loadImage(fName)

    def drawLagrangeReductionBackground(self):
        # We want to visualize where the lagrange reduction occurs when moving either vector
        # We solve this by creating two heatmaps and changing which heatmap is in front depending on what
        # vector was moved last.

        # Dimensions of the data
        ppu = 1200  # Pixels per unit
        width, height = (
            4,
            4,
        )  # Does not work with different width height for some reason
        loops = 10
        folder = "precomputedLagrangeBackgrounds"
        fNames = [
            f"{SCRIPT_DIR}/{folder}/{width},{height},{ppu},{loops},{v},LRBackround.png"
            for v in ["v1", "v2"]
        ]

        def _gen_lr_background(fName, v2_is_fixed):
            def _inner():
                img = lagrange_reduction_visualization(
                    width, height, ppu, v2_is_fixed=v2_is_fixed, loops=loops
                )
                img.save(fName)

            return _inner

        # Use the shared helper to obtain cached or newly generated images
        LR_heatmapImage1 = self._get_or_create_background_image(
            fNames[0], _gen_lr_background(fNames[0], True)
        )
        LR_heatmapImage2 = self._get_or_create_background_image(
            fNames[1], _gen_lr_background(fNames[1], False)
        )

        # Separate instances for the grid visualization view
        GV_heatmapImage1 = self.loadImage(fNames[0])
        GV_heatmapImage2 = self.loadImage(fNames[1])

        self.LR_bg1 = self.drawHeatMap(
            LR_heatmapImage1, -10, self.LR_plot, width, height
        )
        self.LR_bg2 = self.drawHeatMap(
            LR_heatmapImage2, -1, self.LR_plot, width, height
        )

        self.GV_bg1 = self.drawHeatMap(
            GV_heatmapImage1, -10, self.GV_plot, width, height
        )
        self.GV_bg2 = self.drawHeatMap(
            GV_heatmapImage2, -1, self.GV_plot, width, height
        )
        self.GV_bg1.setOpacity(0)  # Set opacity to 0 to hide it
        self.GV_bg2.setOpacity(0)  # Set opacity to 0 to hide it

    @staticmethod
    def fig_to_pg(fig):
        # Ensure the figure renders exactly the pixels we expect
        fig.canvas.draw()

        # Extract RGBA buffer directly
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))

        # Matplotlib gives ARGB; convert to RGBA
        # buf[..., 0] = A, 1 = R, 2 = G, 3 = B
        rgba = np.empty_like(buf)
        rgba[..., 0] = buf[..., 1]  # R
        rgba[..., 1] = buf[..., 2]  # G
        rgba[..., 2] = buf[..., 3]  # B
        rgba[..., 3] = buf[..., 0]  # A

        shiftx = -1
        shifty = 0
        if not (shiftx == 0 and shifty == 0):
            # Allocate output filled with zeros
            shifted = np.zeros_like(rgba)

            # Compute valid source & destination ranges
            src_y0 = max(0, -shifty)
            src_y1 = min(rgba.shape[0], rgba.shape[0] - shifty)
            dst_y0 = max(0, shifty)
            dst_y1 = min(rgba.shape[0], rgba.shape[0] + shifty)

            src_x0 = max(0, -shiftx)
            src_x1 = min(rgba.shape[1], rgba.shape[1] - shiftx)
            dst_x0 = max(0, shiftx)
            dst_x1 = min(rgba.shape[1], rgba.shape[1] + shiftx)

            # Copy data into shifted buffer
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = rgba[src_y0:src_y1, src_x0:src_x1]
        else:
            shifted = rgba

        # Create pg.ImageItem
        img_item = pg.ImageItem(shifted.swapaxes(1, 0))

        return img_item

    @staticmethod
    def compute_stress_scalar_field(stress, mode="det"):
        if mode == "det":
            # det over last two axes
            return np.linalg.det(stress)

        elif mode in ("I1", "trace"):
            # trace over last two axes
            return np.trace(stress, axis1=-2, axis2=-1)
        elif mode in ("N1", "-trace"):
            # sigma_xx - sigma_yy
            return stress[..., 0, 0] - stress[..., 1, 1]

        elif mode in ("J2", "sqrtJ2"):
            # mean (hydrostatic) stress in 2D: (sigma_xx + sigma_yy) / 2
            I1 = np.trace(stress, axis1=-2, axis2=-1)
            mean = I1 / 2.0  # shape (...,)

            # deviatoric stress: s_ij = sigma_ij - mean * delta_ij
            s = stress - mean[..., None, None] * np.eye(2, dtype=stress.dtype)

            # J2 = 1/2 * s : s  (double contraction over last two axes)
            J2 = 0.5 * np.sum(s * s, axis=(-2, -1))  # shape (...,)

            if mode == "J2":
                return J2
            else:  # "sqrtJ2"
                with np.errstate(invalid="ignore"):
                    return np.sqrt(np.clip(J2, 0.0, None))

        else:
            try:
                if isinstance(mode, str):
                    mode = tuple(int(i) for i in mode.strip("()").split(","))
                i, j = mode
                return stress[..., i, j]
            except Exception:
                raise ValueError("Unknown stress mode:", mode)

    def generateEnergyBackground(self, ppu, beta, fName, stress_mode, stressLim):
        from matplotlib import pyplot as plt

        dpi = 400  # any value, just keep it consistent
        fig = plt.figure(figsize=(ppu / dpi, ppu / dpi), dpi=dpi)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # Transparent background so only grid lines are opaque
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        drawPoincareGrid(
            ax,
            grid_size=ppu,
            depth=6,
            c=(0.2, 0.2, 0.2, 0.4),
            linewidth=ppu / (4 * dpi),
        )

        if not self.showStress:
            field = generate_energy_grid(
                resolution=ppu, beta=beta, K=0, zeroReference=True, eps=1e-2
            )
        else:
            if self.stress_type == "cauchy":
                stress = generate_cauchy_stress_grid(
                    resolution=ppu, beta=beta, eps=1e-2
                )
            elif self.stress_type == "PK1":
                stress = generate_piola_stress_grid(
                    beta=beta, resolution=ppu, second_PK=False, eps=1e-2
                )
            elif self.stress_type == "PK2":
                stress = generate_piola_stress_grid(
                    beta=beta, resolution=ppu, second_PK=True, eps=1e-2
                )
            else:
                raise ValueError("Unknown stress type:", self.stress_type)

            with np.errstate(invalid="ignore"):
                field = self.compute_stress_scalar_field(stress, mode=stress_mode)
                field = np.clip(field, *stressLim)

        ax.imshow(
            field,
            origin="lower",
            cmap="coolwarm",
        )
        mappable = ax.images[-1]

        # [left, bottom, width, height] in *figure* coordinates
        # height = 0.20 -> 20% of the image height
        cax = fig.add_axes([0.98, 0.02, 0.015, 0.20])

        cb = plt.colorbar(mappable, cax=cax, orientation="vertical")

        # Transparent background
        cax.set_facecolor((1, 1, 1, 0.0))
        text_color = "#777777"
        # White outline
        cb.outline.set_edgecolor(text_color)
        for spine in cax.spines.values():
            spine.set_edgecolor(text_color)

        textSize = 2 * ppu / dpi

        cb.ax.yaxis.tick_left()  # Put ticks + labels on the left
        cb.ax.yaxis.set_label_position("left")

        cb.ax.tick_params(
            labelsize=textSize,
            length=1,
            width=0.4,
            color=text_color,
            left=True,  # ticks on left
            right=False,
        )

        for label in cb.ax.get_yticklabels():
            label.set_ha("right")  # horizontal alignment: pull toward bar
            label.set_va("center")  # vertical alignment is usually fine
        cb.outline.set_linewidth(0.2)

        for label in cb.ax.get_yticklabels():
            label.set_color(text_color)

        # The "×10⁻¹⁰" offset text: gray and small
        offset_text = cb.ax.yaxis.get_offset_text()
        offset_text.set_color(text_color)
        offset_text.set_ha("right")
        offset_text.set_va("top")
        offset_text.set_size(textSize)

        # Ensure directory exists and save the figure for caching
        dirpath = os.path.dirname(fName)
        os.makedirs(dirpath, exist_ok=True)
        fig.savefig(
            fName,
            dpi=dpi,
            transparent=True,
        )

        # Convert the matplotlib figure directly to a pg.ImageItem
        energyImage = self.fig_to_pg(fig)

        plt.close(fig)

        if os.path.isfile(fName):
            print(f"Saved to {fName}")
        else:
            print("NOT SAVED!")
        return energyImage

    def drawEnergyBackground(
        self,
    ):
        ppu = 2001  # Pixels per unit
        folder = "precomputedEnergyBackgrounds"

        # Set quantity name for caching
        if self.showStress:
            quantity = (
                f"_{self.stress_type}_stress_{self.stress_mode}_clip{self.stressLim[1]}"
            )
        else:
            quantity = "energy"

        self.triangularEnergy = None
        self.squareEnergy = None
        self.angleRegionImage = None

        # Generate energy images for triangular and square shapes
        for shape, beta, opacity in zip(["triangular", "square"], [4, -0.25], [0, 1]):
            fName = f"{SCRIPT_DIR}/{folder}/{ppu}_{shape}_{quantity}_Poincare_LRBackround.png"

            forceGenerate = ppu <= 500  # Always regenerate for small ppu
            energyImage = self._get_or_create_background_image(
                fName,
                lambda: self.generateEnergyBackground(
                    ppu, beta, fName, self.stress_mode, self.stressLim
                ),
                force_generate=forceGenerate,
            )

            # Apply LUT for both cached and newly generated images
            energyImage.setLookupTable(COOLWARM_LUT)

            if opacity == 1:
                self.currentBeta = beta

            # Dynamically assign attributes for triangularEnergy and squareEnergy
            setattr(
                self,
                f"{shape}Energy",
                self.drawHeatMap(energyImage, opacity, self.PCS_plot, 1, 1),
            )

    @staticmethod
    def loadImage(fileName):
        img = QImage(fileName)

        if img.format() != QImage.Format_RGBA8888:
            img = img.convertToFormat(QImage.Format_RGBA8888)

        imgArray = pg.imageToArray(img, copy=True)
        imgArray = imgArray.astype(np.uint8, copy=False)

        imgArray = np.fliplr(imgArray)  # Fix vertical flip

        return pg.ImageItem(imgArray)

    def _snapshot_energy_inputs(self, resolution):
        # Called on the GUI thread only
        lastUsed = self.GV_VP.lastDragged
        last_used_is_e1 = lastUsed is self.GV_VP.e1
        dragged_pos = np.array(lastUsed.head.pos())
        not_dragged_pos = (
            np.array(self.GV_VP.e2.head.pos())
            if last_used_is_e1
            else np.array(self.GV_VP.e1.head.pos())
        )
        x_range, y_range = self.GV_plot.viewRange()

        # Precompute grid sizes to avoid any decisions in worker
        xResolution = int(resolution)
        yResolution = int(
            xResolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
        )

        return {
            "last_used_is_e1": last_used_is_e1,
            "dragged_pos": dragged_pos,
            "fixed_pos": not_dragged_pos,
            "x_range": tuple(x_range),
            "y_range": tuple(y_range),
            "xResolution": xResolution,
            "yResolution": yResolution,
        }

    @staticmethod
    def _build_F_grid_from_snapshot(snap):
        x0, x1 = snap["x_range"]
        y0, y1 = snap["y_range"]
        xR = snap["xResolution"]
        yR = snap["yResolution"]

        # avoid 0 boundary as you did
        eps = 1 - 0.00001
        x_vals, y_vals = np.meshgrid(
            np.linspace(x0 * eps, x1, xR), np.linspace(y0 * eps, y1, yR)
        )
        grid_positions = np.stack([x_vals, y_vals], axis=-1)

        F_grid = np.zeros((*grid_positions.shape[:-1], 2, 2))
        if snap["last_used_is_e1"]:
            # First column varies with grid, second fixed
            F_grid[..., :, 0] = grid_positions
            F_grid[..., :, 1] = snap["fixed_pos"]
        else:
            F_grid[..., :, 0] = snap["fixed_pos"]
            F_grid[..., :, 1] = grid_positions

        return F_grid

    @staticmethod
    def getPlotRange(plot):
        view_range = plot.viewRange()
        x_range = view_range[0]
        y_range = view_range[1]
        # Calculate the rectangle parameters
        x = x_range[0]
        y = y_range[0]
        rect_width = x_range[1] - x_range[0]
        rect_height = y_range[1] - y_range[0]
        return x, y, rect_width, rect_height

    def drawHeatMap(self, heatmap, opacity, plot, width=None, height=None):
        heatmap.setOpacity(opacity)
        heatmap.setZValue(-5)  # Ensure it's in the background
        if width is None or height is None:
            rect = pg.QtCore.QRectF(*self.getPlotRange(plot))
            heatmap.setRect(rect)
        else:
            heatmap.setRect(pg.QtCore.QRectF(-width, -height, 2 * width, 2 * height))
        plot.addItem(heatmap)
        return heatmap

    def updateFEnergyBackground(self):
        self._target_version += 1
        if not self._heatmap_update_pending:
            self._heatmap_update_pending = True
            # run after the current paint cycle
            self._deferred_timer.start(0)

    def _doDeferredUpdates(self):
        self._heatmap_update_pending = False
        # Quick + highres after paint returns
        self._scheduleEnergyUpdate(resolution=100, is_highres=False)
        self._scheduleEnergyUpdate(resolution=700, is_highres=True)

    def _scheduleEnergyUpdate(self, resolution, is_highres):
        executor = (
            self._energy_executor_highres if is_highres else self._energy_executor_quick
        )
        snap = self._snapshot_energy_inputs(resolution)

        # capture all scalars used in the worker
        scalars = dict(
            beta=self.currentBeta,
            K=(4 if self.volumetricEnergy else 0),
            energy_lim=tuple(self.energy_lim),
        )
        version = self._target_version
        executor.submit(
            self._processEnergyUpdate_worker, snap, scalars, version, is_highres
        )

    def _processEnergyUpdate_worker(self, snap, scalars, version, is_highres):
        if version < self._target_version:
            return
        if is_highres:
            from time import sleep

            sleep(0.01)
            if version < self._target_version:
                return

        F_grid = self._build_F_grid_from_snapshot(snap)
        r = min(snap["x_range"][1] - snap["x_range"][0], 0.03)

        with np.errstate(over="ignore", invalid="ignore"):
            energy_grid = self.energyFunc.energy_from_F(
                F_grid,
                scalars["beta"],
                K=scalars["K"],
                zeroReference=True,
                accuracy=1 - r,
            )

        energy_grid = np.clip(energy_grid, *scalars["energy_lim"]).transpose()
        if version < self._target_version:
            return
        self.energyComputed.emit(energy_grid)  # object payload (queued)

    def updateEnergyHeatmap(self, energy_grid):
        image_attr = "PCS_Energy"
        energyImage = pg.ImageItem(energy_grid)
        energyImage.setLookupTable(COOLWARM_LUT)

        if not hasattr(self, image_attr) or getattr(self, image_attr) is None:
            heatmap = self.drawHeatMap(energyImage, 1, self.GV_plot)
            setattr(self, image_attr, heatmap)
        else:
            heatmap = getattr(self, image_attr)
            heatmap.setImage(energy_grid)
            rect = pg.QtCore.QRectF(*self.getPlotRange(self.GV_plot))
            heatmap.setRect(rect)

        heatmap.setLevels(self.energy_lim)

    def drawBackground(self):
        self.drawLagrangeReductionBackground()
        self.drawEnergyBackground()

    def mouseMove(self, pos):
        for VP in [self.LR_VP, self.GV_VP]:
            dragged_vector, not_dragged_vector = VP.dragging_vector()

            if dragged_vector:
                self.VP = VP
                VP.check_move()
                # Do something with the dragged vector, for instance, adjust the other vector to conserve volume.
                # Check if Shift is held
                shift_held = QApplication.keyboardModifiers() & Qt.ShiftModifier

                if shift_held:
                    constrainDeterminant(dragged_vector, not_dragged_vector)
                    self.updateFEnergyBackground()
                if VP.dragingVectorChanged:
                    self.updateFEnergyBackground()
                    if dragged_vector == VP.e1:
                        self.LR_bg1.setZValue(-1)  # show v1
                        self.LR_bg2.setZValue(-10)
                        self.GV_bg1.setZValue(-1)  # show v1
                        self.GV_bg2.setZValue(-10)
                    else:
                        self.LR_bg1.setZValue(-10)
                        self.LR_bg2.setZValue(-1)  # show v2
                        self.GV_bg1.setZValue(-10)
                        self.GV_bg2.setZValue(-1)  # show v2

                self.updateMarkers()
                self.updateGVSpheres()

    def moveVector(self):
        # self.time += 0.01
        # self.GV_VP.e2.head.setPos(np.sin(self.time), 1)
        # self.VP = self.GV_VP
        if not np.all(self.shearVelocity == np.eye(2)):
            self.applyTransformation(self.shearVelocity)

    def applyTransformation(self, transform, roundToInt=False):
        for VP in [self.LR_VP, self.GV_VP]:
            if not self.alt_held:
                VP.applyBasisTransformation(transform, roundToInt)
            else:
                VP.applyPointTransformation(transform, roundToInt)
        self.updateMarkers()
        self.updateGVSpheres()
        self.updateFEnergyBackground()

    def drawHistory(
        self,
        history,
        color=None,
        width=2,
        zValue=3,
        clear=True,
        arrows=True,
        dashed=False,
    ):
        """
        Draw the reduction path in the Poincaré disk.

        Parameters
        ----------
        history : sequence of 2x2 matrices or (N,2,2) ndarray
            The successive C matrices visited during the algorithm.
        color : any pyqtgraph color (tuple/str), optional
            Line/arrow color. Defaults to a muted gray if None.
        width : int, optional
            Line width.
        zValue : int, optional
            Z stacking value so the path sits above backgrounds.
        clear : bool, optional
            If True, remove previously drawn history items first.
        """
        # Lazily create a bucket to track added items so we can remove them later
        if not hasattr(self, "_history_items"):
            self._history_items = []

        if clear:
            # Remove any previously drawn history graphics
            for it in self._history_items:
                try:
                    self.PCS_plot.removeItem(it)
                except Exception:
                    pass
            self._history_items.clear()

        if history is None:
            return

        C = np.asarray(history)
        if C.ndim == 2:  # single matrix -> nothing to connect
            C = C[None, ...]

        if len(C) < 2:
            # Still draw the single point if you like; here we just exit quietly
            return

        # Map C path to (x, y) in the Poincaré disk
        xs, ys = C2PoincareDisk(C)
        xs = np.asarray(xs).ravel()
        ys = np.asarray(ys).ravel()

        # Defaults
        if color is None:
            color = (160, 160, 160)
        pen = pg.mkPen(
            color=color, width=width, style=Qt.DashLine if dashed else Qt.SolidLine
        )

        # Draw a single polyline for the whole path (nice for panning/zooming)
        poly = self.PCS_plot.plot(xs, ys, pen=pen)
        poly.setClipToView(True)
        poly.setDownsampling(False)
        poly.setSkipFiniteCheck(True)
        poly.setZValue(zValue)
        self._history_items.append(poly)
        if arrows:
            # Draw arrows for *each* segment to indicate direction
            for i in range(1, len(xs)):
                x0, y0 = xs[i - 1], ys[i - 1]
                x1, y1 = xs[i], ys[i]

                # Short segment line (helps if poly is downsampled)
                seg = self.PCS_plot.plot([x0, x1], [y0, y1], pen=pen)
                seg.setZValue(zValue)
                self._history_items.append(seg)

                # Arrow at the end of the segment
                # Angle expected by ArrowItem: use 180 - atan2deg, matching drawLine()
                angle_deg = np.degrees(np.arctan2(y1 - y0, x1 - x0))
                arrow = pg.ArrowItem(
                    pos=(x1, y1),
                    headLen=width + 8,
                    angle=180 - angle_deg,
                    brush=color,
                    pen=(0, 0, 0),
                )
                arrow.setZValue(zValue)
                self.PCS_plot.addItem(arrow)
                self._history_items.append(arrow)

    def updateMarkers(self):
        # Normal lagrange reduction
        # rePos1, rePos2, R_C, m, m1, m2, m3 = fast_lagrange_reduction(
        #    self.VP.pos1(), self.VP.pos2())
        # in case no vetor pair is defined
        if not hasattr(self, "VP"):
            self.VP = self.LR_VP
        rePos1, rePos2, C_R, C_E_R, M, m1, m2, m3, ms, history1, history2 = (
            lagrange_reduction(self.VP.pos1(), self.VP.pos2())
        )
        # rePos1, rePos2, m1, m2, m3 = old_lagrange_reduction(
        #     self.VP.pos1(), self.VP.pos2())

        # if not (np.allclose(_rePos1, rePos1) and np.allclose(_rePos2, rePos2) and
        #         np.allclose(_m, m) and _m1 == m1 and _m2 == m2 and _m3 == m3):
        #     print('Fast does not work')
        #     print('Fast does not work')
        #     print(f"_rePos1: {_rePos1}, rePos1: {rePos1}")
        #     print(f"_rePos2: {_rePos2}, rePos2: {rePos2}")
        #     print(f"_m: {_m}, m: {m}")
        #     print(f"_m1: {_m1}, m1: {m1}")
        #     print(f"_m2: {_m2}, m2: {m2}")
        #     print(f"_m3: {_m3}, m3: {m3}")

        self.VP.setPosForSquare(rePos1, rePos2)
        self.GV_VP.copyVP(self.VP)
        self.LR_VP.copyVP(self.VP)
        self.LR_plot.update()

        # Update marker positions
        F, C = generate_matrix(self.VP.pos1(), self.VP.pos2())
        normal_pos = C2PoincareDisk(C)
        self.normal_marker.setData(pos=np.array([normal_pos]))

        reduced_pos = C2PoincareDisk(C_R)
        self.reduced_marker.setData(pos=np.array([reduced_pos]))

        elastic_reduced_pos = C2PoincareDisk(C_E_R)
        self.elastic_reduced_marker.setData(pos=np.array([elastic_reduced_pos]))

        # Calculate P
        F_grid = F[np.newaxis, np.newaxis, :, :]  # shape (1, 1, 2, 2)
        P = self.energyFunc.P_from_F(F_grid, beta=self.currentBeta, K=4)[0, 0]
        # Update table
        self.updateInfoDisplay(F, C, C_R, M, P, ms, m1, m2, m3)
        # Draw the histories (first clears, second overlays)
        if self.showHistory:
            self.drawHistory(
                history1, color=self.reducedColor, width=3, zValue=4, clear=True
            )
            self.drawHistory(
                history2, color=self.elasticReducedColor, width=2, zValue=4, clear=False
            )
        else:
            self.drawHistory([], clear=True)

        self.makeCircles(F)

    def makeCircles(self, F):
        if not self.showCircles:
            return

        def Sx(g):
            return SShear(g, 0)

        def Sy(g):
            return SShear(g, np.pi / 2)

        def Sxy(g):
            return SShear(g, np.pi / 4)

        def Sxy2(g):
            return SShear(g, 3 * np.pi / 4)

        moves = (Sx, Sy, Sxy, Sxy2)
        c1 = "#06923E"
        c2 = "#F4991A"
        colors = (
            c1,
            c1,
            c2,
            c2,
        )
        h = 20  # nr of energy well jumps
        q = 200  # quality of curve
        u = np.linspace(-1, 1, q)
        p = 2.0  # 1 = linear, >1 = more points near zero
        vals = np.sign(u) * (np.abs(u) ** p) * h
        for i, M, c in zip(range(len(moves)), moves, colors):
            if self.alt_held:
                history = [F2C(M(j) @ F) for j in vals]
                self.drawHistory(
                    history,
                    color=c,
                    clear=False,
                    arrows=False,
                    dashed=i % 2 == 1,
                    width=3,
                )
            else:
                history = [F2C(F @ M(j)) for j in vals]
                self.drawHistory(
                    history,
                    color=c,
                    clear=False,
                    arrows=False,
                    dashed=i % 2 == 1,
                    width=3,
                )
        if self.showRightOrth:
            gamma = F[0, 1]
            theta_orth = orth_theta_ref0(gamma)

            history = [F2C(F @ SShear(j, theta_orth)) for j in vals]
            self.drawHistory(
                history, color="green", clear=False, arrows=False, dashed=True
            )

    def onViewRangeChanged(self, view, range):
        self.updateFEnergyBackground()

    def keyPressEvent(self, event):
        # Track Home key pressed state for rotation mode
        if event.key() == Qt.Key_Home:
            self.home_held = True
        self.alt_held = event.modifiers() & Qt.AltModifier  # Check if Alt is held
        self.meta_held = event.modifiers() & Qt.MetaModifier

        # Update axis-locking for the currently dragged vector
        self._update_dragged_vector_axis_locks(event)

        # Rotation mode (Home + Left/Right)
        if self._handle_rotation_keys(event):
            return

        # Shear shortcuts (Y/U, I/O)
        if self._handle_shear_shortcuts(event):
            return

        # View/layout toggles (R, F, P, G)
        self._handle_view_toggles(event)

        # Energy/background-related toggles (T, S, C, L, A, V, B, H)
        self._handle_energy_and_background_toggles(event)

        # Arrow-key shears (Up/Down/Left/Right)
        self._handle_arrow_shear(event)

        # Final common updates
        self._post_keypress_updates()

    def keyReleaseEvent(self, event):
        # Reset axis locks for dragged vectors when X/Y are released
        self._reset_axis_locks_if_needed(event)

        if event.key() == Qt.Key_Home:
            self.home_held = False

        self.alt_held = event.modifiers() & Qt.AltModifier  # Check if Alt is held
        self.meta_held = event.modifiers() & Qt.MetaModifier

        # Reset continuous shear velocity when any key is released
        self.shearVelocity = np.eye(2)

        self._post_keypress_updates()

    def _update_dragged_vector_axis_locks(self, event):
        """Update axis lock flags for the currently dragged vector based on X/Y keys."""
        if hasattr(self, "VP"):
            dragged_vector, _ = self.VP.dragging_vector()
        else:
            dragged_vector = None

        if not dragged_vector:
            return

        if event.key() == Qt.Key_X:
            dragged_vector.moveInY = False
        else:
            dragged_vector.moveInY = True

        if event.key() == Qt.Key_Y:
            dragged_vector.moveInX = False
        else:
            dragged_vector.moveInX = True

    def _handle_rotation_keys(self, event):
        """Handle rotation when Home is held and Left/Right are pressed.

        Returns True if the event was handled and no further processing is needed.
        """
        if not getattr(self, "home_held", False):
            return False

        shift_held_local = event.modifiers() & Qt.ShiftModifier
        step = (
            self.rotation_step_large if shift_held_local else self.rotation_step_small
        )

        if event.key() == Qt.Key_Left:
            angle = -step  # rotate backwards
        elif event.key() == Qt.Key_Right:
            angle = step  # rotate forwards
        else:
            return False

        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        self.applyTransformation(rot)
        return True

    def _handle_shear_shortcuts(self, event):
        """Handle Y/U and I/O shear shortcuts.

        Returns True if the event was handled and keyPressEvent should return.
        """
        shift_held = event.modifiers() & Qt.ShiftModifier
        gamma = None
        transform = None

        if event.key() == Qt.Key_Y:
            gamma = -1 if shift_held else -0.1
            transform = np.array(
                [[1 - 0.5 * gamma, 0.5 * gamma], [-0.5 * gamma, 1 + 0.5 * gamma]]
            )
        elif event.key() == Qt.Key_U:
            gamma = 1 if shift_held else 0.1
            transform = np.array(
                [[1 - 0.5 * gamma, 0.5 * gamma], [-0.5 * gamma, 1 + 0.5 * gamma]]
            )
        elif event.key() == Qt.Key_I:
            gamma = -1 if shift_held else -0.1
            transform = np.array([[1 + gamma, 0], [0, 1 / (1 + gamma)]])
        elif event.key() == Qt.Key_O:
            gamma = 1 if shift_held else 0.1
            transform = np.array([[1 + gamma, 0], [0, 1 / (1 + gamma)]])

        if gamma is None:
            return False

        self.applyTransformation(transform)
        return True

    def _handle_view_toggles(self, event):
        """Handle view/layout related toggles (R, F, P, G)."""
        if event.key() == Qt.Key_R:
            for vp in [self.GV_VP, self.LR_VP]:
                vp.e1.head.setPos(1, 0)
                vp.e2.head.setPos(0, 1)
            s = 2
            self.GV_plot.setRange(xRange=[-s, s], yRange=[-s, s])
            return True

        if event.key() == Qt.Key_F:
            self.w_LR.setVisible(not self.w_LR.isVisible())
            return True
        if event.key() == Qt.Key_P:
            self.w_PCS.setVisible(not self.w_PCS.isVisible())
            return True
        if event.key() == Qt.Key_G:
            self.w_GV.setVisible(not self.w_GV.isVisible())
            return True

        return False

    def _handle_energy_and_background_toggles(self, event):
        """Handle energy/background related toggles (T, S, C, L, A, V, B, H)."""
        k = event.key()

        if k == Qt.Key_T:
            self.triangularEnergy.setOpacity(1)
            self.squareEnergy.setOpacity(0)
            self.currentBeta = 4
            return True

        if k == Qt.Key_S:
            self.triangularEnergy.setOpacity(0)
            self.squareEnergy.setOpacity(1)
            self.currentBeta = -0.25
            return True

        if k == Qt.Key_C:
            self.showCircles = not self.showCircles
            return True

        if k == Qt.Key_L:
            self.elastic_reduced_marker.setVisible(
                not self.elastic_reduced_marker.isVisible()
            )
            self.LR_VP.setVisible(reduced=not self.LR_VP.isVisible(reduced=True))
            self.GV_VP.setVisible(reduced=not self.GV_VP.isVisible(reduced=True))
            self.reduced_marker.setVisible(not self.reduced_marker.isVisible())
            return True

        if k == Qt.Key_A:
            self.angleRegionImage.setVisible(not self.angleRegionImage.isVisible())
            return True

        if k == Qt.Key_V:
            self.volumetricEnergy = not self.volumetricEnergy
            self.updateFEnergyBackground()
            self.updateMarkers()
            return True

        if k == Qt.Key_B:
            # Toggle GV backgrounds
            if self.GV_bg1.opacity() > 0:
                self.GV_bg1.setOpacity(0)
                self.GV_bg2.setOpacity(0)
            else:
                self.GV_bg1.setOpacity(1)
                self.GV_bg2.setOpacity(1)
            return True

        if k == Qt.Key_H:
            self.showHistory = not self.showHistory
            return True

        return False

    def _handle_arrow_shear(self, event):
        """Handle arrow-key shear (Up/Down/Left/Right)."""
        shift_held = event.modifiers() & Qt.ShiftModifier

        upShear = np.array([[1, 0], [1, 1]])
        downShear = np.array([[1, 0], [-1, 1]])
        leftShear = np.array([[1, -1], [0, 1]])
        rightShear = np.array([[1, 1], [0, 1]])
        shearDirection = None

        if event.key() == Qt.Key_Up:
            shearDirection = upShear
        elif event.key() == Qt.Key_Down:
            shearDirection = downShear
        elif event.key() == Qt.Key_Left:
            shearDirection = leftShear
        elif event.key() == Qt.Key_Right:
            shearDirection = rightShear

        if shearDirection is None:
            return False

        shearStep = 1 if shift_held else 0.1
        step_adjusted_shear = np.eye(2) + (shearDirection - np.eye(2)) * shearStep

        if shift_held:
            self.applyTransformation(step_adjusted_shear)  # Integer shear
        else:
            self.shearVelocity = step_adjusted_shear  # Continuous shear

        return True

    def _post_keypress_updates(self):
        """Common updates after handling a key press."""
        self.updateMarkers()
        self.LR_plot.update()
        self.updateGVSpheres()
        self.updateFEnergyBackground()

    def _reset_axis_locks_if_needed(self, event):
        """Reset axis locks on key release of X/Y."""
        if not hasattr(self, "VP"):
            return

        if event.key() == Qt.Key_X:
            self.VP.e1.moveInY = True
            self.VP.e2.moveInY = True

        if event.key() == Qt.Key_Y:
            self.VP.e1.moveInX = True
            self.VP.e2.moveInX = True


def orth_theta_ref0(gamma):
    return -np.atan(
        gamma - np.sqrt(gamma**4 + 3 * gamma**2 + 1) / np.sqrt(gamma**2 + 1)
    )


# def orth_theta_ref0(gamma):
#     gamma = np.asarray(gamma)

#     term1 = np.arctan((2 * gamma * (gamma**2 + 1)) / (3 * gamma**2 + 2))
#     term2 = np.arccos(
#         -(gamma**2) / np.sqrt(4 * gamma**6 + 17 * gamma**4 + 16 * gamma**2 + 4)
#     )

#     # θ⊥ = -1/2 ( term1 ± term2 )
#     theta_plus = -0.5 * (term1 + term2)
#     theta_minus = -0.5 * (term1 - term2)

#     return theta_minus


def runVisualization():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)
    LagrangeReductionVisualization()
    app.exec()
