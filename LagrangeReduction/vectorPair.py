from .vector import Vector


class VectorPair:
    def __init__(self, view, colorS="y", colorH="r", handelable=True, width=2) -> None:
        self.r1 = Vector(
            [(0, 0), (1, 0)],
            handleable=(False, False),
            pen=colorS,
            width=int(width * 0.7),
        )
        view.addItem(self.r1)
        self.r2 = Vector(
            [(0, 0), (0, 1)],
            handleable=(False, False),
            pen=colorS,
            width=int(width * 0.7),
        )
        view.addItem(self.r2)
        self.r3 = Vector(
            [(1, 0), (1, 1)],
            handleable=(False, False),
            pen=colorS,
            width=int(width * 0.7),
        )
        view.addItem(self.r3)
        self.r4 = Vector(
            [(0, 1), (1, 1)],
            handleable=(False, False),
            pen=colorS,
            width=int(width * 0.7),
        )
        view.addItem(self.r4)

        self.e1 = Vector(
            [(0, 0), (1, 0)], handleable=(False, handelable), pen=colorH, width=width
        )
        view.addItem(self.e1)
        self.e2 = Vector(
            [(0, 0), (0, 1)], handleable=(False, handelable), pen=colorH, width=width
        )
        view.addItem(self.e2)

        self.color = colorS
        self.view = view

        self.lastDragged = self.e1  # Random default value
        self.dragingVectorChanged = False

    def pos1(self):
        return self.e1.head.pos()

    def pos2(self):
        return self.e2.head.pos()

    def copyVP(self, VP):
        if VP is not self:
            self.e1.head.setPos(VP.e1.head.pos())
            self.e2.head.setPos(VP.e2.head.pos())
            self.setPosForSquare(VP.r1.head.pos(), VP.r2.head.pos())

    def setPosForSquare(self, p1, p2):
        self.r1.head.setPos(p1[0], p1[1])
        self.r2.head.setPos(p2[0], p2[1])
        self.r3.root.setPos(self.r1.head.pos())
        self.r3.head.setPos(self.r1.head.pos() + self.r2.head.pos())
        self.r4.root.setPos(self.r2.head.pos())
        self.r4.head.setPos(self.r2.head.pos() + self.r1.head.pos())

    def dragging_vector(self):
        if self.e1.head.isMoving:
            self.dragingVectorChanged = self.lastDragged != self.e1
            self.lastDragged = self.e1
            return self.e1, self.e2
        elif self.e2.head.isMoving:
            self.dragingVectorChanged = self.lastDragged != self.e2
            self.lastDragged = self.e2
            return self.e2, self.e1
        return None, None

    def setVisible(self, both=None, main=None, reduced=None):
        if both is not None:
            reduced = both
            main = both
        if reduced is not None:
            self.r1.setVisible(reduced)
            self.r2.setVisible(reduced)
            self.r3.setVisible(reduced)
            self.r4.setVisible(reduced)
        if main is not None:
            self.e1.setVisible(reduced)
            self.e2.setVisible(reduced)

    def isVisible(self, both=None, main=None, reduced=None):
        if both is not None:
            return self.r1.isVisible() and self.e1.isVisible()
        if reduced is not None:
            return self.r1.isVisible()
        if main is not None:
            return self.e1.isVisible()
        raise ValueError("No vector specified")

    def check_move(self):
        # This function constrains the movement to only move in the x or y axis
        # if the flags to do so are set.
        for vec in [self.e1, self.e2]:
            pos = vec.head.pos()
            if not vec.moveInX:
                vec.head.setPos(vec.lastX, pos[1])
            else:
                vec.lastX = pos[0]
            if not vec.moveInY:
                vec.head.setPos(pos[0], vec.lastY)
            else:
                vec.lastY = pos[1]

    def applyTransformation(self, transformation, roundToInt=False):
        # transformation is a 2x2 matrix
        self.e1.applyTransformation(transformation, roundToInt)
        self.e2.applyTransformation(transformation, roundToInt)
