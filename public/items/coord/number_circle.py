from janim.imports import *

from .. import PositionedVItem
from ...utils.geometry_utils import udvec
from ...utils.number_theory_utils import grange

__all___ = ["NumberCircle"]

CCW = 1
CW = -1
OUTSIDE = 1
INSIDE = -1


def _defaultCreateLabel(x: int | float):
    if isinstance(x, int):
        src = str(x)
    else:
        src = f"{x:.2f}"
    i_text = (
        TypstMath(src)(VItem)
        .set_stroke_background(True)
        .stroke.set(color=BLACK, alpha=0.75)
        .r.radius.set(0.05)
        .r.depth.set(-1)
        .r
    )
    return i_text


class NumberCircle(Arc, PositionedVItem):
    def __init__(
        self,
        radius: float = 2.5,
        startAngle: float = PI / 2,
        angle: float = TAU,
        direction: int = CW,
        valueRange=(0, 1),
    ):
        self._r = radius
        self._valueRange = valueRange
        self._startAngle = startAngle
        self._angle = angle
        self._direction = direction = CCW if direction >= 0 else CW
        super().__init__(startAngle, angle, radius)
        self.radius.set(0.01)
        if direction < 0:
            self.points.reverse()

    def getPosition(self):
        return self.get_arc_center()

    def n2a(self, number: float, diff: bool = False) -> float:
        start, stop = self._valueRange
        if start == stop:
            return self._startAngle
        t = (number - start) / (stop - start)
        angleDiff = t * self._angle * self._direction
        if diff:
            return angleDiff
        return angleDiff + self._startAngle

    def n2v(self, number: float) -> float:
        return udvec(self.n2a(number))

    def n2p(self, number: float) -> np.ndarray:
        position = self.getPosition()
        return position + self.n2v(number) * self._r

    def createTick(self, number: float, size: float = 0.1, add: bool = True) -> Line:
        tick = (
            Line(LEFT * size, RIGHT * size)
            .points.shift(RIGHT * self._r)
            .rotate(self.n2a(number), about_point=ORIGIN)
            .shift(self.getPosition())
            .r.radius.set(self.radius.get())
            .r
        )
        if add:
            self.add(tick)
        return tick

    def createTicks(
        self,
        step: float = 1,
        disp: float = 0,
    ):
        ticks = Group()
        start, stop = self._valueRange
        for value in grange(start, stop, step, disp, includeEnd=True):
            if value >= stop:
                break
            tick = self.createTick(value, add=False)
            ticks.add(tick)
        self.add(ticks)
        return ticks

    def addLabel(
        self,
        number: float,
        i_label: Points,
        side: int = OUTSIDE,
        buff: float = 0.1,
        labelSizeMask: Vect = UR,
        autoRotate: bool = False,
        correctDownLabels: bool = True,
        add: bool = True,
    ):
        if autoRotate:
            angle = self.n2a(number)
            i_label.points.move_to(self.getPosition()).shift(
                UP * (self.radius + buff + i_label.points.box.height)
            ).rotate(self.n2a(number) - PI / 2, about_point=self.getPosition())
            if correctDownLabels and angle % TAU > PI:
                i_label.points.rotate(PI)
        else:
            labelSizeMask = np.sign(labelSizeMask)
            side = OUTSIDE if side >= 0 else INSIDE
            labelSize = np.array(
                (
                    i_label.points.box.width,
                    i_label.points.box.height,
                    i_label.points.box.depth,
                )
            )
            d = np.linalg.norm(labelSize * labelSizeMask)
            i_label.points.move_to(self.n2p(number)).shift(
                side * (d / 2 + buff) * self.n2v(number)
            )
        if add:
            self.add(i_label)
        return i_label

    def addLabels[LabelT](
        self,
        step: float = 1,
        labelGenerator: Callable[[float], LabelT] = _defaultCreateLabel,
        disp: float = 0,
        add: bool = True,
        **kwargs,
    ):
        labels = Group[LabelT]()
        start, stop = self._valueRange
        for value in grange(start, stop, step, disp, includeEnd=False):
            label = labelGenerator(value)
            label = self.addLabel(value, label, add=False, **kwargs)
            labels.add(label)
        if add:
            self.add(labels)
        return labels

    def createRay(self, number: float, add: bool = True, **kwargs) -> Line:
        i_line = Line(self.getPosition(), self.n2p(number), **kwargs)
        if add:
            self.add(i_line)
        return i_line

    def createRays(
        self, step: float = 1, disp: float = 0, add: bool = True, **kwargs
    ) -> Group[Line]:
        start, stop = self._valueRange
        rays = Group[Line]()
        for value in grange(start, stop, step, disp, includeEnd=True):
            ray = self.createRay(value, add=False, **kwargs)
            rays.add(ray)
        if add:
            self.add(rays)
        return rays

    def createArc(
        self, start: float, diff: float, buff=0.25, add=True, **kwargs
    ) -> Arc:
        startAngle = self.n2a(start)
        start_, stop_ = self._valueRange
        i_arc = (
            Arc(
                radius=self._r + buff,
                start_angle=startAngle,
                angle=diff / (stop_ - start_) * TAU * self._direction,
                **kwargs,
            )
            .points.shift(self.getPosition())
            .r.color.set(YELLOW)
            .r.depth.set(1)
            .r
        )
        if add:
            self.add(i_arc)
        return i_arc

    def createSector(self, start: float, diff: float, add=True, **kwargs):
        startAngle = self.n2a(start)
        start_, stop_ = self._valueRange
        i_sector = (
            Sector(
                arc_center=self.getPosition(),
                radius=self._r,
                start_angle=startAngle,
                angle=diff / (stop_ - start_) * TAU * self._direction,
                **kwargs,
            )
            .fill.set(color=WHITE, alpha=0.25)
            .r.radius.set(0.01)
            .r.depth.set(1)
            .r
        )
        if add:
            self.add(i_sector)
        return i_sector
