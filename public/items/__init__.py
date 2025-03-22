from janim.imports import *
import numpy as np

__all__ = ["PositionedVItem", "createEmptyDot"]


class PositionedVItem(VItem):
    def getPosition(self) -> np.ndarray:
        raise NotImplementedError

    def setPosition(self, p: Vect) -> Self:
        self.points.shift(np.array(p) - self.getPosition())
        return self


def createEmptyDot() -> Dot:
    return Dot(radius=0.001).color.set(alpha=0).r.stroke.set(alpha=0).r
