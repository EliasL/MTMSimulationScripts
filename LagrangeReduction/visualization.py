import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QFontDatabase
import pyqtgraph as pg
import random
import numpy as np
from PyQt5.QtWidgets import QApplication
import matplotlib
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from .LagrangeReduction import (
    C2PoincareDisk,
    CPos,
    CToAngle,
    conTrans,
    constrainDeterminant,
    generate_flood_fill_coordinates,
    generate_matrix,
    lagrange_reduction,
    lagrange_reduction_visualization,
    manhattanDistance,
)
from .vectorPair import VectorPair
from MTMath.plotEnergy import generate_energy_grid, generate_angle_region
from MTMath.contiPotential import ContiEnergy, SuperSimple

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
    energyComputed = pyqtSignal(np.ndarray)

    def __init__(self):
        # Create two separate executors for quick and high-resolution updates
        self._energy_executor_quick = ThreadPoolExecutor(max_workers=1)
        self._energy_executor_highres = ThreadPoolExecutor(max_workers=1)
        self._target_version = 0
        super().__init__()

        # Colors and line size
        self.background_line_color = np.array([100, 100, 100])
        self.handleColor = "#008B8B"
        self.reducedColor = "#FF6347"
        self.elasticReducedColor = "orange"
        self.lineSize = 2
        self.markerSize = 15
        self.vectorWidth = 8

        # Default energy parameters
        self.currentBeta = -0.25
        self.volumetricEnergy = True
        self.energy_lim = [0, 0.37]

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
        self.energyComputed.connect(self.updateEnergyHeatmap)

        # Connect the signal once after setting up the plot
        self.GV_plot.getViewBox().sigRangeChanged.connect(self.onViewRangeChanged)

        # Hide Lagrange reduction by default
        self.elastic_reduced_marker.setVisible(
            not self.elastic_reduced_marker.isVisible()
        )
        self.LR_VP.setVisible(reduced=not self.LR_VP.isVisible(reduced=True))
        self.GV_VP.setVisible(reduced=not self.GV_VP.isVisible(reduced=True))
        self.reduced_marker.setVisible(not self.reduced_marker.isVisible())

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

            E = ContiEnergy.energy_from_F(
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

    def drawMetricSpaceBackground(self):
        # Draw lines
        self.drawCircle(1)
        nr = 1000
        one = np.array([1] * nr)
        zero = np.array([0] * nr)

        # t has many values close to 0, and fewer larger values
        t = np.sinh(np.linspace(np.arcsinh(0.001), np.arcsinh(300), nr))

        # Shearing circles
        self.drawF(one, zero, t, one, width=1, color="#222", zValue=-2)
        self.drawF(one, t, zero, one, width=1, color="#222", zValue=-2)

        """
        See Note 1 at the bottom of the document for more theory on what follows
        """
        depth = 5

        # VERTICAL LINE
        t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2 / np.sqrt(3)), nr))
        # Values from -1<t<1 give complex solutions
        # det=1, C12=C21, C11=C22
        # Vertical Positive
        C_V_P = np.array([[t, np.sqrt(t**2 - 1)], [np.sqrt(t**2 - 1), t]]).transpose(
            2, 0, 1
        )
        self.drawAllVariations(C_V_P, depth)
        # Vertical Negative
        C_V_N = np.array([[t, -np.sqrt(t**2 - 1)], [-np.sqrt(t**2 - 1), t]]).transpose(
            2, 0, 1
        )

        self.drawAllVariations(C_V_N, depth)

        # HORIZONTAL LINE
        # Values from -1<t<1 are outside of the circle
        t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
        # det=1, C12=C21, C12=0
        C_H = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
        self.drawAllVariations(C_H, depth, color="green")

        # FUNDAMENTAL DOMAIN (0.01 to avoid div by 0)
        # https://www.wolframalpha.com/input?i=0%3Ca%3Cd%2C+b%3Da%2F2%2C+++a*d-b*c%3D1%2C+b%3Dc
        t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(2 / np.sqrt(3)), nr))
        # Negative values are outside of the circle
        # det=1, C12=C21,
        C_F = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)

        self.drawAllVariations(C_F, depth, color="red")

    def drawTriangularConfigurationLines(self, C, loops=1, width=1, color=None):
        if color is None:
            color = self.background_line_color

        for x in range(-loops, loops + 1):
            for y in range(-loops, loops + 1):
                C_ = self.CPos(C, x, y)
                self.drawAllVariations(C_, color=color, width=width)

    def drawColorfullTriangularConfigurationLines(
        self, C, conjugations=2, width=10, color=None
    ):
        if color is None:
            color = self.background_line_color

        # The transposing and swaping of the top and bottom rows means that
        # purple is now [1,1], and red is [0,0]. When we then shift everything
        # +1,+1, purple is (0,0) and red is (-1, -1)
        color_matrix = np.array(
            [
                ["red", "pink", "blue"],
                ["yellow", "purple", "orange"],
                ["green", "white", "grey"],
            ]
        ).transpose()
        temp = color_matrix[:, 2].copy()
        color_matrix[:, 2] = color_matrix[:, 0]
        color_matrix[:, 0] = temp

        nr_colors = (2 * conjugations + 1) ** 2
        i = min([nr_colors * 2, 30])

        # We want to record the start and stop possition of each line. We do
        # this using the angle on the unit circle
        CLineAngles = np.zeros((conjugations * 2 + 1, conjugations * 2 + 1, 2))

        for x, y in generate_flood_fill_coordinates(conjugations, manhattanDistance):
            C_ = CPos(C, x, y).copy()
            # print(f"{x}, {y}")
            # print(color_matrix[x+1, y+1])
            # m = len(C_)//2 # Middle
            # print(C_[0],C_[m], C_[-1])
            if conjugations == 1:
                color = color_matrix[x + 1, y + 1]
            else:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
            self.drawAllVariations(C_, color=color, width=i, showArrows=False)
            i = i - 2
            i = max([i, 1])

            CLineAngles[x + conjugations, y + conjugations, :] = CToAngle(C_)
        return CLineAngles

    def drawAllVariations(self, C, depth=0, **kwards):
        nr = len(C)
        one = np.array([1] * nr)
        zero = np.array([0] * nr)

        m1 = np.array([[one, zero], [zero, -one]]).transpose(2, 0, 1)
        m2 = np.array([[zero, one], [one, zero]]).transpose(2, 0, 1)
        m3 = np.array([[one, -one], [zero, one]]).transpose(2, 0, 1)
        # m3Inv = np.linalg.inv(m3)

        def up(C):
            return conTrans(C, m3)

        def right(C):
            return conTrans(C, m3.transpose(0, 2, 1))

        # kwards['color']=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # kwards['width']=(1+depth)*10
        self.drawC(C, **kwards)
        self.drawC(conTrans(C, m1), **kwards)
        self.drawC(conTrans(C, m2), **kwards)
        self.drawC(conTrans(conTrans(C, m1), m2), **kwards)
        if depth > 0:
            self.drawAllVariations(up(C), depth - 1, **kwards)
            self.drawAllVariations(right(C), depth - 1, **kwards)

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

        if os.path.isfile(fNames[0]) and os.path.isfile(fNames[1]):
            LR_heatmapImage1 = self.loadImage(fNames[0])
            LR_heatmapImage2 = self.loadImage(fNames[1])
            GV_heatmapImage1 = self.loadImage(fNames[0])
            GV_heatmapImage2 = self.loadImage(fNames[1])
        else:
            LR_heatmapImage1 = lagrange_reduction_visualization(
                width, height, ppu, v2_is_fixed=True, loops=loops
            )
            LR_heatmapImage1.save(fNames[0])
            LR_heatmapImage2 = lagrange_reduction_visualization(
                width, height, ppu, v2_is_fixed=False, loops=loops
            )
            LR_heatmapImage2.save(fNames[1])

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

    def drawEnergyBackground(self):
        ppu = 2000  # Pixels per unit
        folder = "precomputedEnergyBackgrounds"
        self.triangularEnergy = None
        self.squareEnergy = None
        self.angleRegionImage = None

        # Generate energy images for triangular and square shapes
        for shape, beta, offset in zip(["triangular", "square"], [4, -0.25], [-10, -1]):
            fName = f"{SCRIPT_DIR}/{folder}/{ppu},{shape},Poincare,LRBackround.png"
            # Check if file exists, if not generate and save
            if os.path.isfile(fName):
                energyImage = self.loadImage(fName)
            else:
                energy_grid = generate_energy_grid(
                    resolution=ppu, beta=beta, K=0, zeroReference=True
                ).transpose()
                energyImage = pg.ImageItem(energy_grid)
                energyImage.setLookupTable(COOLWARM_LUT)
                energyImage.save(fName)
            if offset == -1:
                self.currentBeta = beta

            # Dynamically assign attributes for triangularEnergy and squareEnergy
            setattr(
                self,
                f"{shape}Energy",
                self.drawHeatMap(energyImage, offset, self.PCS_plot, 1, 1),
            )

        # Generate angle region image with a different colormap
        fName_angle = f"{SCRIPT_DIR}/{folder}/{ppu},angleRegion,LRBackround.png"
        if os.path.isfile(fName_angle) and False:
            angleRegionImage = self.loadImage(fName_angle)
        else:
            angle_region_data = generate_angle_region(resolution=ppu).transpose()
            angleRegionImage = pg.ImageItem(angle_region_data)
            angleRegionImage.setLookupTable(VIRIDIS_LUT)
            angleRegionImage.save(fName_angle)

        angleRegionImage.setOpacity(0.5)
        self.angleRegionImage = self.drawHeatMap(
            angleRegionImage, 0, self.PCS_plot, 1, 1
        )
        self.angleRegionImage.setVisible(False)

    @staticmethod
    def loadImage(fileName):
        img = QImage(fileName)
        img = img.convertToFormat(QImage.Format_RGBA64)
        imgArray = pg.imageToArray(img, copy=True)
        imageItem = pg.ImageItem(imgArray)
        imageItem.setAutoDownsample(False)
        return imageItem

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

    def drawHeatMap(self, heatmap, zValue, plot, width=None, height=None):
        heatmap.setZValue(zValue)
        if width is None or height is None:
            rect = pg.QtCore.QRectF(*self.getPlotRange(plot))
            heatmap.setRect(rect)
        else:
            heatmap.setRect(pg.QtCore.QRectF(-width, -height, 2 * width, 2 * height))
        plot.addItem(heatmap)
        return heatmap

    def getFGrid(self, resolution=100):
        lastUsed = self.GV_VP.lastDragged
        not_dragged_vector = (
            self.GV_VP.e2 if lastUsed is self.GV_VP.e1 else self.GV_VP.e1
        )

        lastUsed = np.array(lastUsed.head.pos())
        not_dragged_vector_pos = np.array(not_dragged_vector.head.pos())
        # Extract the current view range from the grid view plot
        x_range, y_range = self.GV_plot.viewRange()

        # Optionally, adjust resolution based on the view size
        xResolution = resolution
        yResolution = int(
            xResolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
        )
        # We create some offsets to not calculate thing too close to zero
        # when using nice ranges in the beginning
        eps = 1 - 0.00001
        # Create the meshgrid using the extracted ranges
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_range[0] * eps, x_range[1], xResolution),
            np.linspace(y_range[0] * eps, y_range[1], yResolution),
        )
        grid_positions = np.stack([x_vals, y_vals], axis=-1)
        # Check if lastUsed is e1 (first column) or e2 (second column)
        if lastUsed is self.GV_VP.e1:
            # First column is variable, second is fixed
            F_grid = np.zeros((*grid_positions.shape[:-1], 2, 2))
            F_grid[..., :, 0] = grid_positions  # First column varies with grid
            F_grid[..., :, 1] = not_dragged_vector_pos  # Second column is fixed
        else:
            # First column is fixed, second is variable
            F_grid = np.zeros((*grid_positions.shape[:-1], 2, 2))
            F_grid[..., :, 0] = not_dragged_vector_pos  # First column is fixed
            F_grid[..., :, 1] = grid_positions  # Second column varies with grid

        return F_grid

    def updateFEnergyBackground(self):
        self._target_version += 1
        # Quick update
        self._scheduleEnergyUpdate(resolution=100, is_highres=False)
        # High-resolution update
        self._scheduleEnergyUpdate(resolution=700, is_highres=True)

    def _scheduleEnergyUpdate(self, resolution, is_highres):
        executor = (
            self._energy_executor_highres if is_highres else self._energy_executor_quick
        )
        executor.submit(
            self._processEnergyUpdate, resolution, self._target_version, is_highres
        )

    def _processEnergyUpdate(self, resolution, version, is_highres):
        if version < self._target_version:
            return
        if is_highres:
            # We wait a bit to see if the user has perhaps already changed the view
            sleep(0.01)
            if version < self._target_version:
                return

        F_grid = self.getFGrid(resolution)
        # We only need high accuracy if the zoom level is high
        x_range, _ = self.GV_plot.viewRange()
        r = np.diff(x_range)[0]
        r = min(r, 0.03)
        with np.errstate(over="ignore", invalid="ignore"):
            energy_grid = SuperSimple.energy_from_F(
                F_grid,
                self.currentBeta,
                K=4 if self.volumetricEnergy else 0,
                zeroReference=True,
                accuracy=1 - r,
            )
        energy_grid = np.clip(energy_grid, *self.energy_lim).transpose()

        # Check if this result is outdated (for highres only)
        if version < self._target_version:
            return  # discard outdated results

        self.energyComputed.emit(energy_grid)

    def updateEnergyHeatmap(self, energy_grid):
        image_attr = "PCS_Energy"
        energyImage = pg.ImageItem(energy_grid)
        energyImage.setLookupTable(COOLWARM_LUT)

        if not hasattr(self, image_attr) or getattr(self, image_attr) is None:
            heatmap = self.drawHeatMap(energyImage, -1, self.GV_plot)
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
        self.drawMetricSpaceBackground()

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
            VP.applyTransformation(transform, roundToInt)
        self.updateMarkers()
        self.updateGVSpheres()
        self.updateFEnergyBackground()

    def updateMarkers(self):
        # Normal lagrange reduction
        # rePos1, rePos2, R_C, m, m1, m2, m3 = fast_lagrange_reduction(
        #    self.VP.pos1(), self.VP.pos2())
        # in case no vetor pair is defined
        if not hasattr(self, "VP"):
            self.VP = self.LR_VP
        rePos1, rePos2, C_R, C_E_R, M, m1, m2, m3, ms = lagrange_reduction(
            self.VP.pos1(), self.VP.pos2()
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
        P = ContiEnergy.P_from_F(F, M, self.currentBeta, K=4)

        # Update table
        self.updateInfoDisplay(F, C, C_R, M, P, ms, m1, m2, m3)

    def onViewRangeChanged(self, view, range):
        self.updateFEnergyBackground()

    def keyPressEvent(self, event):
        if hasattr(self, "VP"):
            dragged_vector, not_dragged_vector = self.VP.dragging_vector()
        else:
            dragged_vector = None
        if dragged_vector:
            if event.key() == Qt.Key_X:
                dragged_vector.moveInY = False
            else:
                dragged_vector.moveInY = True

            if event.key() == Qt.Key_Y:
                dragged_vector.moveInX = False
            else:
                dragged_vector.moveInX = True

        if event.key() == Qt.Key_R:
            for vp in [self.GV_VP, self.LR_VP]:
                vp.e1.head.setPos(1, 0)
                vp.e2.head.setPos(0, 1)
            s = 2
            self.GV_plot.setRange(xRange=[-s, s], yRange=[-s, s])
            self.updateMarkers()
            self.LR_plot.update()
            self.updateGVSpheres()
            self.updateFEnergyBackground()

        if event.key() == Qt.Key_T:
            self.triangularEnergy.setZValue(-1)
            self.squareEnergy.setZValue(-10)
            self.currentBeta = 4
        if event.key() == Qt.Key_S:
            self.triangularEnergy.setZValue(-10)
            self.squareEnergy.setZValue(-1)
            self.currentBeta = -0.25
        if event.key() == Qt.Key_F:
            self.w_LR.setVisible(not self.w_LR.isVisible())
        if event.key() == Qt.Key_P:
            self.w_PCS.setVisible(not self.w_PCS.isVisible())
        if event.key() == Qt.Key_G:
            self.w_GV.setVisible(not self.w_GV.isVisible())
        if event.key() == Qt.Key_L:
            self.elastic_reduced_marker.setVisible(
                not self.elastic_reduced_marker.isVisible()
            )
            self.LR_VP.setVisible(reduced=not self.LR_VP.isVisible(reduced=True))
            self.GV_VP.setVisible(reduced=not self.GV_VP.isVisible(reduced=True))
            self.reduced_marker.setVisible(not self.reduced_marker.isVisible())
        if event.key() == Qt.Key_A:
            self.angleRegionImage.setVisible(not self.angleRegionImage.isVisible())
        if event.key() == Qt.Key_V:
            self.volumetricEnergy = not self.volumetricEnergy
            self.updateFEnergyBackground()
            self.updateMarkers()
        if event.key() == Qt.Key_B:
            # Check if the ImageItem is currently visible by checking its opacity
            if self.GV_bg1.opacity() > 0:
                self.GV_bg1.setOpacity(0)  # Set opacity to 0 to hide it
                self.GV_bg2.setOpacity(0)  # Set opacity to 0 to hide it
            else:
                self.GV_bg1.setOpacity(1)  # Set opacity to 1 to show it
                self.GV_bg2.setOpacity(1)  # Set opacity to 1 to show it

        # here we want to handle arrow key presses. If the user presses up,
        # we should apply a simple shear transformation upwards.
        # If the shift key is down, we should perform an integer shear.
        # Otherwise, we should set some shear velocity that is applied every frame.
        shift_held = event.modifiers() & Qt.ShiftModifier  # Check if Shift is held
        alt_held = event.modifiers() & Qt.AltModifier  # Check if Alt is held

        upShear = np.array([[1, 0], [1, 1]])
        downShear = np.array([[1, 0], [-1, 1]])
        leftShear = np.array([[1, -1], [0, 1]])
        rightShear = np.array([[1, 1], [0, 1]])
        shearDirection = None

        if shift_held and alt_held:
            shearStep = 0.5
        elif shift_held:
            shearStep = 1
        elif alt_held:
            shearStep = 0.01
        else:
            shearStep = 0.1

        if event.key() == Qt.Key_Up:
            shearDirection = upShear
        if event.key() == Qt.Key_Down:
            shearDirection = downShear
        if event.key() == Qt.Key_Left:
            shearDirection = leftShear
        if event.key() == Qt.Key_Right:
            shearDirection = rightShear

        if shearDirection is not None:
            step_adjusted_shear = np.eye(2) + (shearDirection - np.eye(2)) * shearStep

            if shift_held:
                self.applyTransformation(step_adjusted_shear)  # Integer shear
            elif shearDirection is not None:
                self.shearVelocity = step_adjusted_shear  # Continuous shear

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_X:
            self.VP.e1.moveInY = True
            self.VP.e2.moveInY = True

        if event.key() == Qt.Key_Y:
            self.VP.e1.moveInX = True
            self.VP.e2.moveInX = True

        self.shearVelocity = np.eye(2)

    def drawLine(
        self, x, y, color=None, width=1, dashed=False, zValue=-1, showArrows=False
    ):
        dashPattern = [10, 10] if dashed else None
        color = color if color is not None else self.background_line_color
        line = self.PCS_plot.plot(
            x, y, pen=pg.mkPen(color=color, width=width, dash=dashPattern)
        )
        line.setZValue(zValue)
        if showArrows and len(x) > 1 and len(y) > 1:
            # Calculate angle for the arrow at the end of the line

            angle = self.angleBetweenPoints((x[-1], y[-1]), (x[-2], y[-2]))

            # Create and add an arrow at the end of the lineangle at the end of a line python
            endArrow = pg.ArrowItem(
                pos=(x[-1], y[-1]),
                headLen=width + 10,
                angle=180 - angle,
                brush=color,
                pen=(0, 0, 0),
            )
            self.PCS_plot.addItem(endArrow)

    def drawEllipse(self, a, b, x=0, y=0, color=None):
        """
        Draw an ellipse with semi-major axis 'a' and semi-minor axis 'b'.
        (h,k) is the center of the ellipse.
        """
        t = np.linspace(0, np.pi * 2, 100)
        x = x + a * np.sin(t)
        y = y + b * np.cos(t)
        self.drawLine(x, y, color)

    def drawCircle(self, r, x=0, y=0, color=None):
        """
        Draw a circle of radius 'r' centered at (x,y).
        """
        self.drawEllipse(r, r, x, y, color)

    def drawF(self, F11, F12, F21, F22, **kwargs):
        F = np.array([[F11, F12], [F21, F22]]).transpose(2, 0, 1)
        C = F.transpose(0, 2, 1) @ F
        self.drawC(C, **kwargs)

    def drawC(self, C, C12=None, C22=None, **kwargs):
        if C12 is not None and C22 is not None:
            # Assuming C is C11 here
            C = np.array([[C, C12], [C12, C22]]).transpose(2, 0, 1)
        # If C is already the array, it will be used as is
        pos = C2PoincareDisk(C)
        self.drawLine(pos[0], pos[1], **kwargs)

    def dim_colormap(self, colormap, factor=0.5, neutral=(0, 0, 0), alpha=1):
        """
        Dim a colormap by interpolating its colors with a neutral color.

        Parameters:
        - colormap: The original colormap.
        - factor: A factor between 0 (fully neutral) and 1 (original colormap).
        - neutral: A neutral color to interpolate with, e.g., (0, 0, 0) for black.

        Returns:
        - A new dimmed colormap.
        """
        for i, color in enumerate(colormap.color):
            r = color[0] * factor + neutral[0] * (1 - factor)
            g = color[1] * factor + neutral[1] * (1 - factor)
            b = color[2] * factor + neutral[2] * (1 - factor)
            colormap.color[i] = (r, g, b, alpha)

        return colormap


def runVisualization():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)
    LagrangeReductionVisualization()
    app.exec()
