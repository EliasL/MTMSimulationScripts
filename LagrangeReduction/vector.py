import numpy as np
from pyqtgraph import UIGraphicsItem, QtCore, ROI, Point, QtGui
from PyQt5.QtCore import QPointF
import pyqtgraph.functions as fn


class Pos(UIGraphicsItem):
    def __init__(self, pos=(0, 0)):
        super(Pos, self).__init__()

        if isinstance(pos, tuple) and len(pos) == 2:
            # If pos is a tuple of (x, y) coordinates, create a QPointF object
            self.setPos(QtCore.QPointF(*pos))
        elif isinstance(pos, QtCore.QPointF):
            # If pos is already a QPointF object, use it directly
            self.setPos(pos)
        else:
            raise ValueError("Invalid pos argument. Use a tuple (x, y) or QPointF.")


class Vector(ROI):
    r"""
    ROI subclass with one fixed, and one freely-moving handles defining a line.

    ============== =============================================================
    **Arguments**
    positions      (list of two length-2 sequences) The endpoints of the line
                   segment. Note that, unlike the handle positions specified in
                   other ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(
        self,
        positions=(None, None),
        pos=None,
        handleable=(False, True),
        width=1,
        **args,
    ):
        if pos is None:
            pos = [0, 0]

        ROI.__init__(self, pos, [1, 1], **args)
        if len(positions) > 2:
            raise Exception(
                "LineSegmentROI must be defined by exactly 2 positions. For more points, use PolyLineROI."
            )

        self.head = None
        self.root = None

        if handleable[0]:
            self.root = self.addFreeHandle(positions[0])
        else:
            self.root = Pos(positions[0])
            self.handles.append({"item": self.root})

        if handleable[1]:
            self.head = self.addFreeHandle(positions[1])
            self.head.mouseClickEvent = self.mouseClickEvent
        else:
            self.head = Pos(positions[1])
            self.handles.append({"item": self.head})

        self.translatable = False

        # axies that the vector is allowed to move in
        self.moveInX = True
        self.moveInY = True
        self.lastX = positions[1][0]
        self.lastY = positions[1][1]

        # line width
        self.width = width

    @property
    def endpoints(self):
        # must not be cached because self.handles may change.
        return [h["item"] for h in self.handles]

    def listPoints(self):
        return [p["item"].pos() for p in self.handles]

    def getState(self):
        state = ROI.getState(self)
        state["points"] = [Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        state = ROI.saveState(self)
        state["points"] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self, state):
        ROI.setState(self, state)
        p1 = [
            state["points"][0][0] + state["pos"][0],
            state["points"][0][1] + state["pos"][1],
        ]
        p2 = [
            state["points"][1][0] + state["pos"][0],
            state["points"][1][1] + state["pos"][1],
        ]
        self.movePoint(self.getHandles()[0], p1, finish=False)
        self.movePoint(self.getHandles()[1], p2)

    def mouseClickEvent(self, ev):
        pos = self.head.pos()
        if self.moveInX:
            self.lastX = pos[0]
        if not self.moveInY:
            self.lastY = pos[1]

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Set the pen's width (adjust the value as needed)
        self.currentPen.setWidth(self.width)
        p.setPen(self.currentPen)
        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()

        p.drawLine(h1, h2)

        # Calculate the direction vector
        direction = h2 - h1
        length = np.linalg.norm(direction)

        # Define arrowhead parameters
        arrow_size = min(
            [0.1 * length, 0.3]
        )  # You can adjust this to control the size of the arrowhead
        arrow_angle = np.arctan2(
            direction.y(), direction.x()
        )  # Calculate the angle using arctan2

        # Calculate the points for the arrowhead
        arrow_p1 = h2
        arrow_p2 = h2 - QtCore.QPointF(
            arrow_size
            * np.cos(
                arrow_angle + np.pi / 6
            ),  # Adding pi/6 to the angle for the arrowhead
            arrow_size
            * np.sin(
                arrow_angle + np.pi / 6
            ),  # Adding pi/6 to the angle for the arrowhead
        )
        arrow_p3 = h2 - QtCore.QPointF(
            arrow_size
            * np.cos(
                arrow_angle - np.pi / 6
            ),  # Subtracting pi/6 for the other side of the arrowhead
            arrow_size
            * np.sin(
                arrow_angle - np.pi / 6
            ),  # Subtracting pi/6 for the other side of the arrowhead
        )

        # Draw the arrowhead
        p.drawLine(arrow_p1, arrow_p2)
        p.drawLine(arrow_p1, arrow_p3)

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        p = QtGui.QPainterPath()

        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        dh = h2 - h1
        if dh.length() == 0:
            return p
        pxv = self.pixelVectors(dh)[1]
        if pxv is None:
            return p

        pxv *= 4

        p.moveTo(h1 + pxv)
        p.lineTo(h2 + pxv)
        p.lineTo(h2 - pxv)
        p.lineTo(h1 - pxv)
        p.lineTo(h1 + pxv)
        return p

    def getArrayRegion(
        self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, **kwds
    ):
        """
        Use the position of this ROI relative to an imageItem to pull a slice
        from an array.

        Since this pulls 1D data from a 2D coordinate system, the return value
        will have ndim = data.ndim-1

        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
        imgPts = [self.mapToItem(img, h.pos()) for h in self.endpoints]

        d = Point(imgPts[1] - imgPts[0])
        o = Point(imgPts[0])
        rgn = fn.affineSlice(
            data,
            shape=(int(d.length()),),
            vectors=[Point(d.norm())],
            origin=o,
            axes=axes,
            order=order,
            returnCoords=returnMappedCoords,
            **kwds,
        )

        return rgn

    def applyTransformation(self, transformation, roundToInt=False):
        # Ensure transformation is a 2x2 NumPy array
        pos = self.head.pos()  # QPointF
        pos_array = np.array([pos.x(), pos.y()])  # Convert to NumPy array
        new_pos = np.dot(transformation, pos_array)  # Apply transformation
        if roundToInt:
            # round, but avoid rounding to zero in either of the components

            new_pos = np.round(new_pos)

        self.head.setPos(QPointF(new_pos[0], new_pos[1]))  # Convert back
        self.update()
