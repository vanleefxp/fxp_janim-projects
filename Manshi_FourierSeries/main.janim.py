import math
from functools import lru_cache, partial
from collections.abc import Mapping
from pathlib import Path
from numbers import Rational
from fractions import Fraction as Q

from janim.imports import *
import numpy as np
from egcd import egcd
import pyrsistent as pyr

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
arrowConfig = dict(center_anchor="front", body_length=0.15, back_width=0.15)
config = Config(
    font=[
        "NewComputerModern10",
        "FandolSong",
    ],
    typst_shared_preamble=(DIR / "../assets/typst/manshi_preamble.typ").read_text(),
)


def circle3pData(p1: Vect, p2: Vect, p3: Vect) -> tuple[Vect, float]:
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3

    a, b, c = 2 * (x2 - x1), 2 * (y2 - y1), (x2**2 - x1**2) + (y2**2 - y1**2)
    d, e, f = 2 * (x3 - x2), 2 * (y3 - y2), (x3**2 - x2**2) + (y3**2 - y2**2)

    delta = a * e - b * d

    h = (c * e - b * f) / delta
    k = (a * f - c * d) / delta
    r = np.sqrt((x1 - h) ** 2 + (y1 - k) ** 2)

    return np.array((h, k, 0)), r


def circle3p(p1: Vect, p2: Vect, p3: Vect) -> Circle:
    center, radius = circle3pData(p1, p2, p3)
    return Circle(radius).points.shift(center).r


def perpPoint(p1: Vect, p2: Vect, p3: Vect) -> Vect:
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v23 = p3 - p2
    v21 = p1 - p2
    dot = np.dot(v23, v21)
    mag = np.dot(v23, v23)
    t = dot / mag
    p4 = p2 + t * v23
    return p4


def crtUnitVecs(divisors: Iterable[int]):
    prod = math.prod(divisors)
    for d in divisors:
        otherProd = prod // d
        _, coef, _ = egcd(otherProd, d)
        yield (coef * otherProd) % prod


def solveCRT(divisors: Iterable[int], remainders: Iterable[int]) -> tuple[int, int]:
    prod = math.prod(divisors)
    result = 0
    for d, r in zip(divisors, remainders):
        otherProd = prod // d
        _, coef, _ = egcd(otherProd, d)
        result += r * coef * otherProd
    result %= prod
    return result, prod


def cubeEdges(nDim=3) -> Iterable[tuple[int]]:
    for startVertice in range(1 << nDim):
        for dim in range(nDim):
            endVertice = startVertice | (1 << dim)
            if endVertice > startVertice:
                yield (startVertice, endVertice)


def toDigitArray(n: int, digits: int, base: int = 2) -> np.ndarray[int]:
    res = np.empty(digits, dtype=int)
    for i in range(digits):
        res[i] = n % base
        n //= base
    return res


def cubeEdgeLines(p0: Vect, p1: Vect) -> Group[Line]:
    nDim = len(p0)
    points = np.array((p0, p1))
    r = np.arange(nDim)
    i_group = Group()
    for startVertice, endVertice in cubeEdges(nDim):
        startPoint = points[toDigitArray(startVertice, nDim), r]
        endPoint = points[toDigitArray(endVertice, nDim), r]
        i_group.add(Line(startPoint, endPoint))
    return i_group


@lru_cache(maxsize=1 << 10)
def charCount(src: str, textType: type[Text | TypstDoc] = TypstMath):
    src = src.strip()
    if len(src) == 0:
        return 0
    return len(textType(src))


def _createPolynomialCoef(n: int) -> str:
    return f"a_({n})"


def createPolynomialTerm(n: int, sym: str = "x") -> str:
    if n == 0:
        return ""
    elif n == 1:
        return sym
    else:
        return f"{sym}^({n})"


def polyTermCharCount(n: int, sym: str = "x") -> int:
    if n == 0:
        return 0
    elif n == 1:
        return charCount(sym)
    else:
        return charCount(sym) + len(str(n))


_polynomialDefaultWidths = pyr.m(
    eq=0.6,
    symbol=1,
    coef=0.4,
    term=0.4,
    sign=0.5,
    ellipsis=1,
)

_polynomialDefaultAligns = pyr.m(
    coef=-1,
    term=-1,
    symbol=1,
    sign=0,
    ellipsis=-1,
    eq=0,
)


def halign[I = VItem](item: I, x0: float, w: float, alignment=-1) -> I:
    if alignment < 0:
        x = x0
    elif alignment == 0:
        x = x0 + w / 2
    else:
        x = x0 + w
    return item.points.move_to(
        x * RIGHT, coor_mask=RIGHT, aligned_edge=alignment * RIGHT
    ).r


def createEmptyItem() -> Dot:
    return Dot(radius=0, fill_alpha=0)


class MarkedTypstMath(TypstMath, MarkedItem):
    def __init__(self, text, *args, **kwargs) -> None:
        # 在输入前面增加一个 ".", 用于确定文本基线的位置
        super().__init__(". " + text, *args, **kwargs)
        y0 = self[0].points.box.bottom[1]
        self.remove(self[0])  # 移除增加的的点
        if len(self) > 0:
            x0 = self.points.box.left[0]
            self.mark.set_points(((x0, y0, 0),))
        else:
            self.add(createEmptyItem())
            self.mark.set_points(((0, y0, 0),))
        self.mark.set(ORIGIN)


type Sgn = Literal[1, -1]


class PolyTermOptBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def sign(self) -> Sgn:
        raise NotImplementedError

    @property
    @abstractmethod
    def order(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def coefSrc(self) -> str:
        raise NotImplementedError

    @property
    def omit(self) -> bool:
        return False


class PolyTermOpt(PolyTermOptBase):
    def __new__(cls, sign: Sgn, coefSrc: str, order: int, omit: bool = False) -> Self:
        self = super().__new__(cls)
        self._sign = sign
        self._coefSrc = coefSrc
        self._order = order
        self._omit = omit
        return self

    @property
    def sign(self) -> Sgn:
        return self._sign

    @property
    def coefSrc(self) -> str:
        return self._coefSrc

    @property
    def order(self) -> int:
        return self._order

    @property
    def omit(self) -> bool:
        return self._omit


class NumberedPolyTermOpt(PolyTermOptBase):
    def __new__(cls, order, symbol: str = "a") -> Self:
        self = super().__new__(cls)
        self._order = order
        self._symbol = symbol
        return self

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def order(self) -> int:
        return self._order

    @property
    def sign(self) -> Sgn:
        return 1

    @property
    def coefSrc(self) -> str:
        return f"{self.symbol}_({self.order})"


class RationalPolyTermOpt[N = Rational](PolyTermOptBase):
    def __new__(cls, coef: N, order: int) -> Self:
        self = super().__new__(cls)
        self._coef = coef
        self._order = order
        return self

    @property
    def coef(self) -> N:
        return self._coef

    @property
    def sign(self) -> Sgn:
        return 1 if self.coef.numerator >= 0 else -1

    @property
    def order(self) -> int:
        return self._order

    @property
    def coefSrc(self) -> str:
        if self.order != 0 and abs(self.coef) == 1:
            return ""  # 非常数项不写系数 1
        return str(abs(self.coef))

    @property
    def omit(self) -> bool:
        return self.coef == 0


class PolynomialText(Group[VItem], MarkedItem):
    def __init__(
        self,
        terms: Iterable[PolyTermOptBase] | None = None,
        unknownSymbol: str = "x",
        nameSymbol: str = "f(x)",
        typstConfig: Mapping[str, Any] = pyr.m(),
        widths: Mapping[str, Any] = _polynomialDefaultWidths,
        aligns: Mapping[str, Any] = _polynomialDefaultAligns,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if terms is None:
            terms = (NumberedPolyTermOpt(i) for i in range(7))
        if widths is not _polynomialDefaultWidths:
            widths = _polynomialDefaultWidths | widths
        if aligns is not _polynomialDefaultAligns:
            aligns = _polynomialDefaultAligns | aligns

        # 多项式符号
        i_symbol = self._i_symbol = halign(
            MarkedTypstMath(nameSymbol, **typstConfig),
            -widths["eq"] - widths["symbol"],
            widths["symbol"],
            aligns["symbol"],
        )
        self.add(*i_symbol)

        # 等号
        i_eq = self._i_eq = halign(
            MarkedTypstMath("=", **typstConfig),
            -widths["eq"],
            widths["eq"],
            aligns["eq"],
        )
        self.add(*i_eq)

        i_coefs = self._i_coefs = Group()  # 系数
        i_terms = self._i_terms = Group()  # 项
        i_signs = self._i_adds = Group()  # 加号

        pos = 0
        for i, termOpt in enumerate(terms):
            omit = termOpt.omit
            # 符号
            i_signs.add(
                i_sign := halign(
                    MarkedTypstMath(
                        ""
                        if omit
                        else "-"
                        if termOpt.sign < 0
                        else ""
                        if i == 0
                        else "+",
                        **typstConfig,
                    ),
                    pos,
                    widths["sign"],
                    aligns["sign"],
                )
            )
            pos += widths["sign"]
            self.add(*i_sign)

            # 系数
            i_coefs.add(
                i_coef := halign(
                    MarkedTypstMath("" if omit else termOpt.coefSrc, **typstConfig),
                    pos,
                    widths["coef"],
                    aligns["coef"],
                )
            )
            pos += widths["coef"]
            self.add(*i_coef)

            # 项
            i_terms.add(
                i_term := halign(
                    MarkedTypstMath(
                        ""
                        if omit
                        else createPolynomialTerm(termOpt.order, unknownSymbol),
                        **typstConfig,
                    ),
                    pos,
                    widths["term"],
                    aligns["term"],
                )
            )
            pos += widths["term"]
            self.add(*i_term)

        i_signs.add(
            i_sign := halign(
                MarkedTypstMath("+", **typstConfig),
                pos,
                widths["sign"],
                aligns["sign"],
            )
        )
        pos += widths["sign"]
        self.add(*i_sign)

        i_ellipsis = self._i_ellipsis = halign(
            MarkedTypstMath("...", **typstConfig),
            pos,
            widths["ellipsis"],
            aligns["ellipsis"],
        )
        self.add(*i_ellipsis)

        self.mark.set_points((ORIGIN,))

    @property
    def i_symbol(self) -> VItem:
        return self._i_symbol

    @property
    def i_eq(self) -> VItem:
        return self._i_eq

    @property
    def i_coefs(self) -> Group[VItem]:
        return self._i_coefs

    @property
    def i_terms(self) -> Group[VItem]:
        return self._i_terms

    @property
    def i_signs(self) -> Group[VItem]:
        return self._i_adds

    @property
    def i_ellipsis(self) -> VItem:
        return self._i_ellipsis


class TL_Coord(Timeline):
    CONFIG = config

    def construct(self):
        u = 0.75
        coords = np.array(
            (
                (4, 3),
                (-3, 4),
                (-5, -2),
                (5, -3),
            )
        )
        startCoord = endCoord = coords[0]
        i_coord = NumberPlane(
            x_range=(-10, 10, 1),
            y_range=(-7, 7, 1),
            background_line_style={"stroke_alpha": 0.75},
            unit_size=u,
        )
        originPoint = i_coord.c2p(0, 0)
        startPoint = i_coord.c2p(*startCoord)
        x0, y0, _ = originPoint
        x1, y1, _ = startPoint

        i_dot = Dot(startPoint, radius=0.05).depth.set(-1).r
        i_lineToX = Line(startPoint, (x1, y0, 0)).stroke.set(color=RED).r
        i_lineToY = Line(startPoint, (x0, y1, 0)).stroke.set(color=GREEN).r

        def createCoordText(coord):
            x, y = coord
            return (
                Text(f"({x:.2f}, {y:.2f})")
                .points.next_to(i_coord.c2p(*coord), aligned_edge=DL)
                .r
            )

        i_coordText = createCoordText(startCoord)

        self.play(Create(i_coord), duration=2)
        self.forward(1)
        self.play(FadeIn(i_dot))
        self.play(Write(i_coordText))
        self.play(Create(i_lineToX))
        self.play(Create(i_lineToY))

        def getUpdaterFns(startCoord, endCoord):
            def updateDot(i_: Dot, params: UpdaterParams) -> None:
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                i_.points.move_to(i_coord.c2p(*coord))

            def updateLineToX(i_: Line, params: UpdaterParams) -> None:
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                x, y = coord
                i_.points.put_start_and_end_on(i_coord.c2p(x, y), i_coord.c2p(x, 0))

            def updateLineToY(i_: Line, params: UpdaterParams) -> None:
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                x, y = coord
                i_.points.put_start_and_end_on(i_coord.c2p(x, y), i_coord.c2p(0, y))

            def updateCoordText(params: UpdaterParams):
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                return createCoordText(coord)

            return updateDot, updateLineToX, updateLineToY, updateCoordText

        def animatePointMove(startCoord, endCoord):
            updaterFns = getUpdaterFns(startCoord, endCoord)
            self.play(
                i_dot.anim.points.move_to(i_coord.c2p(*coord)),
                *(
                    GroupUpdater(i_, updater)
                    for i_, updater in zip(
                        (i_dot, i_lineToX, i_lineToY),
                        updaterFns[:-1],
                    )
                ),
                ItemUpdater(i_coordText, updaterFns[-1]),
                duration=1.5,
            )

        for coord in coords[1:]:
            endCoord = coord
            animatePointMove(startCoord, endCoord)
            startCoord = endCoord

        endCoord = coords[0]
        animatePointMove(startCoord, endCoord)

        self.forward(2)

        i_ux = Line(originPoint, i_coord.c2p(1, 0)).stroke.set(color=RED).r
        i_uy = Line(originPoint, i_coord.c2p(0, 1)).stroke.set(color=GREEN).r

        i_ux.add_tip(fill_color=RED, **arrowConfig)
        i_uy.add_tip(fill_color=GREEN, **arrowConfig)

        self.forward(1)
        self.play(
            Uncreate(i_lineToX),
            Uncreate(i_lineToY),
            FadeOut(i_coordText),
            FadeOut(i_dot),
        )

        startCoord = endCoord

        i_lineOnX = (
            Line(originPoint, i_coord.c2p(startCoord[0], 0)).stroke.set(color=RED).r
        )
        i_lineOnY = (
            Line(originPoint, i_coord.c2p(0, startCoord[1])).stroke.set(color=GREEN).r
        )
        i_tickOnX = Line(ORIGIN, UP * 0.15).points.move_to(originPoint).r
        i_tickOnY = Line(ORIGIN, RIGHT * 0.15).points.move_to(originPoint).r
        i_xText = Text("0.00").points.next_to(originPoint, DOWN).r
        i_yText = Text("0.00").points.next_to(originPoint, LEFT).r

        def updateLineOnX(i_: Line, params: UpdaterParams):
            t = params.alpha
            endX = startCoord[0]
            i_.points.put_start_and_end_on(originPoint, i_coord.c2p(endX * t, 0))

        def updateTickOnX(i_: Line, params: UpdaterParams):
            t = params.alpha
            endX = startCoord[0]
            i_.points.move_to(i_coord.c2p(endX * t, 0))

        def updateLineOnY(i_: Line, params: UpdaterParams):
            t = params.alpha
            endY = startCoord[1]
            i_.points.put_start_and_end_on(originPoint, i_coord.c2p(0, endY * t))

        def updateTickOnY(i_: Line, params: UpdaterParams):
            t = params.alpha
            endY = startCoord[1]
            i_.points.move_to(i_coord.c2p(0, endY * t))

        def updateXText(params: UpdaterParams):
            t = params.alpha
            endX = startCoord[0]
            x = endX * t
            return Text(f"{x:.2f}").points.next_to(i_coord.c2p(x, 0), DOWN).r

        def updateYText(params: UpdaterParams):
            t = params.alpha
            endY = startCoord[1]
            y = endY * t
            return Text(f"{y:.2f}").points.next_to(i_coord.c2p(0, y), LEFT).r

        self.play(
            GroupUpdater(i_lineOnX, updateLineOnX),
            GroupUpdater(i_tickOnX, updateTickOnX),
            ItemUpdater(i_xText, updateXText),
            duration=1.5,
        )
        self.play(
            GroupUpdater(i_lineOnY, updateLineOnY),
            GroupUpdater(i_tickOnY, updateTickOnY),
            ItemUpdater(i_yText, updateYText),
            duration=1.5,
        )
        i_lineToX.points.reverse()
        i_lineToY.points.reverse()
        self.play(Create(i_lineToX), Create(i_lineToY))
        self.play(Write(i_coordText), FadeIn(i_dot))

        self.forward(1)
        self.play(Uncreate(i_lineOnX), Uncreate(i_lineOnY))
        self.play(Create(i_ux), duration=1)
        self.play(Create(i_uy), duration=1)

        i_perp = RightAngle(i_ux, i_uy, length=0.15)
        self.play(FadeIn(i_perp))

        self.forward(2)


class TL_Geometry_Pythagorean(Timeline):
    def construct(self):
        self.camera.points.shift((3, 0.1, 0))
        theta = PI / 3
        r = 1.25

        alpha = theta / 2
        c = 2 * r
        a = c * np.sin(alpha)
        b = c * np.cos(alpha)

        trianglePoints = (
            (r, 0, 0),
            (r * np.cos(theta), r * np.sin(theta), 0),
            (-r, 0, 0),
        )
        points1 = (
            trianglePoints[0],
            (r, -2 * r, 0),
            (-r, -2 * r, 0),
            trianglePoints[-1],
        )
        points2 = (
            trianglePoints[0],
            (r + a * np.cos(alpha), a * np.sin(alpha), 0),
            (
                r + a * np.cos(alpha) - a * np.sin(alpha),
                a * np.sin(alpha) + a * np.cos(alpha),
                0,
            ),
            trianglePoints[1],
        )
        points3 = (
            trianglePoints[-1],
            (-r - b * np.sin(alpha), b * np.cos(alpha), 0),
            (
                -r - b * np.sin(alpha) + b * np.cos(alpha),
                b * np.cos(alpha) + b * np.sin(alpha),
                0,
            ),
            trianglePoints[1],
        )

        i_triangle = Polygon(*trianglePoints).radius.set(0.015).r
        i_square1 = Polyline(*points1).radius.set(0.015).r
        i_square2 = Polyline(*points2).radius.set(0.015).r
        i_square3 = Polyline(*points3).radius.set(0.015).r

        i_square1_1 = Polygon(*points1).stroke.set(GREEN).r.fill.set(GREEN, alpha=0.5).r
        i_square2_1 = (
            Polygon(*points2)
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r.points.reverse()
            .r
        )
        i_square3_1 = Polygon(*points3).stroke.set(RED).r.fill.set(RED, alpha=0.5).r

        i_line1 = Line(points1[0], points3[1]).radius.set(0.015).r
        i_line2 = Line(points1[2], points3[-1]).radius.set(0.015).r
        i_line3 = Line(points1[1], points2[-1]).radius.set(0.015).r
        i_line4 = Line(points1[-1], points2[1]).radius.set(0.015).r
        i_line5 = (
            Line(trianglePoints[1], (r * np.cos(theta), -2 * r, 0)).radius.set(0.015).r
        )

        i_triangle1_1 = (
            Polygon(trianglePoints[1], points3[1], trianglePoints[-1])
            .stroke.set(RED)
            .r.fill.set(RED, alpha=0.5)
            .r
        )
        i_triangle1_2 = (
            Polygon(trianglePoints[0], points3[1], trianglePoints[-1])
            .stroke.set(RED)
            .r.fill.set(RED, alpha=0.5)
            .r
        )
        i_triangle1_3 = (
            Polygon(points1[2], (r * np.cos(theta), 0, 0), trianglePoints[-1])
            .stroke.set(RED)
            .r.fill.set(RED, alpha=0.5)
            .r
        )

        i_triangle2_1 = (
            Polygon(trianglePoints[0], points2[1], points2[-1])
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r
        )
        i_triangle2_2 = (
            Polygon(trianglePoints[0], points2[1], trianglePoints[-1])
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r
        )
        i_triangle2_3 = (
            Polygon(trianglePoints[0], (r * np.cos(theta), 0, 0), points1[1])
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r
        )

        self.play(Create(i_triangle, auto_close_path=False), duration=0.5)
        self.play(
            AnimGroup(
                *(
                    Create(i_, auto_close_path=False, duration=0.5)
                    for i_ in (
                        i_square1,
                        i_square2,
                        i_square3,
                        i_line1,
                        i_line2,
                        i_line3,
                        i_line4,
                        i_line5,
                    )
                ),
                lag_ratio=0.25,
            )
        )
        self.play(FadeIn(i_triangle1_1), duration=0.5)
        self.play(Transform(i_triangle1_1, i_triangle1_2))
        self.play(
            Rotate(i_triangle1_2, -PI / 2, about_point=trianglePoints[-1]),
            duration=0.75,
        )

        self.play(FadeIn(i_triangle2_1), duration=0.5)
        self.play(Transform(i_triangle2_1, i_triangle2_2))
        self.play(
            Rotate(i_triangle2_2, PI / 2, about_point=trianglePoints[0]),
            duration=0.75,
        )

        self.play(
            Transform(i_triangle1_2, i_triangle1_3),
            Transform(i_triangle2_2, i_triangle2_3),
        )
        self.play(FadeOut(i_triangle1_3), FadeOut(i_triangle2_3), duration=0.5)
        self.play(FadeIn(i_square2_1), FadeIn(i_square3_1), duration=0.5)
        self.play(
            Transform(i_square2_1, i_square1_1, hide_src=False),
            Transform(i_square3_1, i_square1_1, hide_src=False),
            duration=1.5,
        )
        self.forward_to(12)


class TL_Geometry_NinePointCircle(Timeline):
    def construct(self):
        self.camera.points.shift((-2.25, -0.1, 0))
        trianglePoints = np.array(
            (
                (-2, -2.5, 0),
                (3.5, -2, 0),
                (0, 2, 0),
            )
        )
        midpoints = (trianglePoints + np.roll(trianglePoints, 1, axis=0)) / 2
        perpPoints = np.array(
            [
                perpPoint(*np.roll(trianglePoints, -i, axis=0))
                for i in range(len(trianglePoints))
            ]
        )
        center, radius = circle3pData(*midpoints)
        perpMidpoints = -midpoints + 2 * center

        i_triangle = Polygon(*trianglePoints).radius.set(0.015).r
        i_midTriangle = Polygon(*midpoints).radius.set(0.015).r
        i_circ = Circle(radius).points.shift(center).r.stroke.set(RED).r
        i_perpLines = Group(
            *(
                Line(p1, p2).radius.set(0.015).r
                for p1, p2 in zip(trianglePoints, perpPoints)
            )
        )
        i_diameters = Group(
            *(
                Line(p1, p2).radius.set(0.015).r
                for p1, p2 in zip(midpoints, perpMidpoints)
            )
        )
        i_center = Dot(center, radius=0.05)

        self.play(Create(i_triangle, auto_close_path=False), duration=1)
        self.play(
            AnimGroup(
                Create(i_midTriangle, auto_close_path=False),
                Create(i_circ, auto_close_path=False),
                lag_ratio=0.25,
            )
        )
        self.play(Create(i_perpLines))

        self.play(
            AnimGroup(
                *(Flash(point) for point in np.concat((midpoints, perpPoints), axis=0)),
                lag_ratio=0.5,
            )
        )
        self.play(FadeIn(i_center))
        self.play(Create(i_diameters))
        self.play(
            AnimGroup(
                *(Flash(point) for point in perpMidpoints),
                lag_ratio=0.5,
            )
        )

        self.forward_to(12)


class TL_Geometry(Timeline):
    def construct(self):
        tl1 = TL_Geometry_Pythagorean().build().to_item(keep_last_frame=True).show()
        tl2 = TL_Geometry_NinePointCircle().build().to_item(keep_last_frame=True).show()
        self.forward_to(max(tl1.end, tl2.end))


class TL_CRT_3d(Timeline):
    CONFIG = config

    def __init__(self, showUvecText=False, fadeOut=True, *args, **kwargs):
        self._showUvecText = showUvecText
        self._fadeOut = fadeOut
        super().__init__(*args, **kwargs)

    def construct(self):
        showUvecText = self._showUvecText
        fadeOut = self._fadeOut

        self.camera.points.rotate(PI / 3, axis=RIGHT).rotate(PI / 3, axis=OUT).scale(
            1.25
        ).shift((-1, 1, 2.5))
        divisors = (3, 5, 7)
        remainders = (1, 2, 6)
        colors = (RED, GREEN, BLUE)
        gcd = math.prod(divisors)
        unitVecs = tuple(crtUnitVecs(divisors))

        endpoints = (RIGHT, UP, OUT)
        axisLablePlacements = (DOWN, LEFT, LEFT)
        uvecTextPlacements = (DOWN, UP, DOWN)

        i_vecs = Group(
            *(
                Vector(endpoint, color=color)
                for endpoint, color in zip(endpoints, colors)
            )
        )
        i_axisLabels = Group(
            *(
                Text(
                    f"mod {d}",
                    stroke_radius=0.01,
                    stroke_color=WHITE,
                    stroke_alpha=1,
                    depth=-1,
                )
                .points.next_to(endpoint, buff=0.1, direction=direction)
                .r
                for d, endpoint, direction in zip(
                    divisors, endpoints, axisLablePlacements
                )
            )
        )
        if showUvecText:
            i_uvecTexts = Group(
                *(
                    Text(
                        str(uvec),
                        stroke_alpha=1,
                        stroke_color=color,
                        fill_color=color,
                        stroke_radius=0.01,
                        depth=-2,
                    )
                    .points.next_to(i_axisLabel, placement, buff=0.1)
                    .r
                    for uvec, i_axisLabel, placement, color in zip(
                        unitVecs, i_axisLabels, uvecTextPlacements, colors
                    )
                )
            )

        i_lineToAxes = Group(
            *(
                Line(ORIGIN, endpoint * remainder, color=color)
                for color, endpoint, remainder in zip(colors, endpoints, remainders)
            )
        )
        i_cubeEdges = cubeEdgeLines(ORIGIN, remainders)
        i_dot = Dot(radius=0.05)

        for i_line in i_cubeEdges:
            i_line.stroke.set(alpha=0.5).r.depth.set(2)

        self.play(
            AnimGroup(*(Create(i_) for i_ in i_vecs), lag_ratio=0.5), duration=1.5
        )
        self.play(Write(i_axisLabels))
        if showUvecText:
            self.play(FadeIn(i_uvecTexts))

        self.play(
            *(i_.anim.stroke.set(alpha=0.5).r.fill.set(alpha=0.5) for i_ in i_vecs),
            FadeIn(i_dot),
        )

        i_linesToCreate = Group(
            Group(),
            Group(i_cubeEdges[3]),
            Group(i_cubeEdges[4], i_cubeEdges[6], i_cubeEdges[7]),
        )
        transforms = (
            (),
            (Transform(i_cubeEdges[0], i_cubeEdges[5]),),
            (
                Transform(i_cubeEdges[3], i_cubeEdges[10], hide_src=False),
                Transform(i_cubeEdges[5], i_cubeEdges[11], hide_src=False),
                Transform(i_cubeEdges[1], i_cubeEdges[9]),
                Transform(i_cubeEdges[0], i_cubeEdges[8]),
            ),
        )

        def createCoordText(point) -> Text:
            return (
                Text(
                    f"({
                        ', '.join(
                            map(
                                lambda x, c: f'<sc {c}><fc {c}>{x}</fc></sc>',
                                np.round(point).astype(int),
                                colors,
                            )
                        )
                    })",
                    format=Text.Format.RichText,
                    stroke_alpha=1,
                    stroke_radius=0.005,
                    depth=-1,
                )
                .points.next_to(ORIGIN, UP, buff=0.1)
                .rotate(PI / 4, axis=RIGHT, about_point=ORIGIN)
                .shift(point)
                .r
            )

        def createResultText(point) -> Text:
            resultValue = np.dot(np.round(point).astype(int), unitVecs) % gcd
            return (
                Text(f"{resultValue}")
                .points.next_to(ORIGIN, DR, buff=0)
                .shift(RIGHT * 0.2)
                .rotate(PI / 4, axis=RIGHT, about_point=ORIGIN)
                .shift(point)
                .r.stroke.set(color=WHITE, alpha=1)
                .r.radius.set(0.005)
                .r.depth.set(-1)
                .r
            )

        def createTextUpdaterFn(currentPoint, shift, textFactory):
            def updaterFn(params: UpdaterParams):
                t = params.alpha
                point = currentPoint + t * shift
                return textFactory(point)

            return updaterFn

        def createVecUpdaterFn(currentPoint, shift):
            def updaterFn(params: UpdaterParams):
                t = params.alpha
                point = currentPoint + t * shift
                return Vector(point, stroke_color=PINK, fill_color=PINK)

            return updaterFn

        i_coordText = createCoordText(ORIGIN)
        i_resultText = createResultText(ORIGIN)
        i_vec = Vector(ORIGIN)

        self.play(Write(i_coordText), duration=0.5)
        currentPoint = ORIGIN
        for remainder, endpoint, i_line, i_createLines, transform in zip(
            remainders, endpoints, i_lineToAxes, i_linesToCreate, transforms
        ):
            shift = remainder * endpoint
            self.play(
                Create(i_line),
                *(Create(i_) for i_ in i_createLines),
                *transform,
                i_dot.anim.points.shift(shift),
                ItemUpdater(
                    i_coordText,
                    createTextUpdaterFn(currentPoint, shift, createCoordText),
                ),
                ItemUpdater(
                    i_resultText,
                    createTextUpdaterFn(currentPoint, shift, createResultText),
                ),
                ItemUpdater(i_vec, createVecUpdaterFn(currentPoint, shift)),
                duration=np.sqrt(remainder),
            )
            currentPoint = currentPoint + shift
        self.play(
            *(i_.anim.stroke.set(alpha=1).r.fill.set(alpha=1) for i_ in i_vecs),
        )
        self.forward(1)
        if fadeOut:
            self.play(*map(partial(FadeOut, root_only=True), self.visible_items()))


class TL_CRT_RemainderAdd(Timeline):
    CONFIG = config

    def __init__(self, divisor=3, counts=(22, 34), *args, **kwargs):
        self._divisor = divisor
        self._counts = counts
        super().__init__(*args, **kwargs)

    def construct(self):
        divisor = self._divisor
        counts = self._counts
        dotColors = (BLUE, GREEN)
        rems = tuple(i % divisor for i in counts)
        sumRem = sum(rems) % divisor

        gridConfig = pyr.m(h_buff=0.3, v_buff=0.5, fill_rows_first=False)
        i_pile0 = (
            (Dot(fill_color=dotColors[0]) * counts[0])
            .points.arrange_in_grid(n_rows=divisor, **gridConfig)
            .r
        )
        i_pile1 = (
            (Dot(fill_color=dotColors[1]) * counts[1])
            .points.arrange_in_grid(n_rows=divisor, **gridConfig)
            .r
        )
        Group(i_pile0, i_pile1).points.arrange_in_grid(n_rows=1, h_buff=1).shift(UP)
        i_modText0 = (
            TypstMath(
                f'#text($x$, rgb("{dotColors[0]}")) equiv {rems[0]} quad '
                f'mod #text(${divisor}$, rgb("{RED}"))'
            )
            .points.next_to(i_pile0, DOWN, buff=1.5)
            .r
        )
        i_modText1 = (
            TypstMath(
                f'#text($y$, rgb("{dotColors[1]}")) equiv {rems[1]} quad '
                f'mod #text(${divisor}$, rgb("{RED}"))'
            )
            .points.next_to(i_pile1, DOWN, buff=1.5)
            .r
        )

        self.play(
            AnimGroup(*(FadeIn(i_) for i_ in i_pile0), duration=1.5, lag_ratio=0.5)
        )
        self.play(
            Write(i_modText0),
            ShowPassingFlashAround(i_pile0[-rems[0] :], time_width=3, duration=2),
        )
        self.forward(1)

        self.play(
            AnimGroup(*(FadeIn(i_) for i_ in i_pile1), duration=1.5, lag_ratio=0.5)
        )
        self.play(
            Write(i_modText1),
            ShowPassingFlashAround(i_pile1[-rems[1] :], time_width=3, duration=2),
        )
        self.forward(1)

        i_combinedPile = (
            Group(*i_pile0, *i_pile1)
            .copy()
            .points.arrange_in_grid(n_rows=divisor, **gridConfig)
            .shift(UP)
            .r
        )
        i_modText = (
            TypstMath(
                f'#text($x$, rgb("{dotColors[0]}")) + #text($y$, rgb("{dotColors[1]}")) '
                f'equiv {sumRem} quad mod #text(${divisor}$, rgb("{RED}"))'
            )
            .points.next_to(i_combinedPile, DOWN, buff=1.5)
            .r
        )

        divisorCharCount = len(str(divisor))
        remCharCounts = tuple(len(str(r)) for r in rems)
        sumRemCharCount = len(str(sumRem))

        self.prepare(
            Transform(i_modText0[0], i_modText[0]),  # x
            Transform(i_modText1[0], i_modText[2]),  # y
            Transform(i_modText0[1], i_modText[3]),  # 同余符号
            Transform(i_modText1[1], i_modText[3]),  # 同余符号
            Transform(
                i_modText0[2 : 2 + remCharCounts[0]], i_modText[4 : 4 + sumRemCharCount]
            ),  # 余数 1
            Transform(
                i_modText1[2 : 2 + remCharCounts[0]], i_modText[4 : 4 + sumRemCharCount]
            ),  # 余数 2
            FadeIn(i_modText[1]),  # 加号
            Transform(
                i_modText0[-3 - divisorCharCount :], i_modText[-3 - divisorCharCount :]
            ),  # mod
            Transform(
                i_modText1[-3 - divisorCharCount :], i_modText[-3 - divisorCharCount :]
            ),  # mod,
            duration=2,
            at=1,
        )
        self.play(
            AnimGroup(
                *(
                    Transform(i_pile0[i], i_combinedPile[i], path_arc=PI / 2)
                    for i in range(counts[0] - 1, -1, -1)
                ),
                lag_ratio=0.1,
                duration=2,
            ),
            AnimGroup(
                *(
                    Transform(
                        i_pile1[i], i_combinedPile[i + counts[0]], path_arc=PI / 2
                    )
                    for i in range(counts[1])
                ),
                lag_ratio=0.1,
                duration=2,
            ),
        )
        self.forward(1)
        self.play(
            ShowPassingFlashAround(i_combinedPile[-sumRem:], time_width=3), duration=2
        )
        self.forward(1)
        self.play(FadeOut(i_combinedPile), FadeOut(i_modText), duration=0.5)


class TL_CRT(Timeline):
    CONFIG = config

    def construct(self):
        divisors = (3, 5, 7)
        remainders = (1, 2, 6)
        colors = (RED, GREEN, BLUE)
        sol_n, sol_gcd = solveCRT(divisors, remainders)
        unitVecs = tuple(crtUnitVecs(divisors))
        dotColor = BLUE

        gridConfig = pyr.m(fill_rows_first=False, h_buff=0.2, v_buff=0.4)
        rectConfig = pyr.m(
            fill_color=GREY,
            fill_alpha=0.95,
            stroke_color=GREY,
            stroke_alpha=0,
        )
        showCols = 5

        i_title = Title("韩信点兵", font_size=36)
        i_dots = (
            (Dot(fill_color=dotColor) * sol_n)
            .points.arrange_in_grid(n_rows=divisors[0], **gridConfig)
            .r
        )
        i_modTexts = Group(
            *(
                TypstMath(
                    f'n equiv #text(${r}$, fill: rgb("{color}")) quad '
                    f'mod #text(${d}$, fill: rgb("{color}"))'
                )
                for d, r, color in zip(divisors, remainders, colors)
            )
        )

        def createSurroundingRect(d, r, i_dots=i_dots):
            i_rect = SurroundingRect(i_dots[d * showCols : -r], **rectConfig)
            i_rect.add(
                Text("?", font_size=48, stroke_color=BLACK, stroke_alpha=1)
                .set_stroke_background(True)
                .points.move_to(i_rect)
                .r.radius.set(0.04)
                .r
            )
            return i_rect

        d, r, i_text = (
            divisors[0],
            remainders[0],
            i_modTexts[0].points.next_to(i_dots, DOWN, buff=0.5).r,
        )
        i_rect = createSurroundingRect(d, r)

        # 展示第一种排列方式
        self.play(Write(i_title))
        self.forward(2)
        self.play(
            AnimGroup(*(FadeIn(i_) for i_ in i_dots), lag_ratio=0.5, duration=1.5),
            FadeIn(i_rect, duration=1.5),
        )
        self.play(
            ShowPassingFlashAround(i_dots[-r:], time_width=3, duration=2),
            Write(i_text, duration=1),
        )
        self.forward(2)
        self.play(FadeOut(i_text))

        # 更换到其他排列方式
        for d, r, i_text in zip(divisors[1:], remainders[1:], i_modTexts[1:]):
            i_dots_cp = i_dots.copy().points.arrange_in_grid(n_rows=d, **gridConfig).r
            i_text.points.next_to(i_dots_cp, DOWN, buff=0.5)
            i_newRect = createSurroundingRect(d, r, i_dots=i_dots_cp)
            i_newRect.add(Text("?", font_size=48).points.move_to(i_rect).r)
            self.play(
                AnimGroup(
                    *(
                        Transform(item1, item2)
                        for item1, item2 in zip(i_dots, i_dots_cp)
                    ),
                    # lag_ratio=0.01,
                    duration=2,
                ),
                Transform(i_rect, i_newRect, duration=2),
            )
            i_dots = i_dots_cp
            self.play(
                ShowPassingFlashAround(i_dots[-r:], time_width=3, duration=2),
                Write(i_text, duration=1),
            )
            i_rect = i_newRect
            self.forward(2)
            self.play(FadeOut(i_text))
        self.play(FadeOut(i_rect))

        # 数一遍总数
        i_count = Text("0").points.next_to(i_dots, DOWN, buff=0.5).r
        self.show(i_count)
        for i, i_dot in enumerate(i_dots):
            i_count.become(Text(str(i + 1)).points.next_to(i_dots, DOWN, buff=0.5).r)
            i_dot.fill.set(WHITE)
            self.forward(0.03)
        self.forward(0.5)
        self.play(*(i_.anim.fill.set(dotColor) for i_ in i_dots))
        self.forward(2)

        # 总数变成坐标
        i_countAsCoord = (
            Text(
                f"{sol_n} ← ({', '.join(f'<c {color}>{r}</c>' for color, r in zip(colors, remainders))})",
                format=Text.Format.RichText,
            )
            .points.move_to((-3, 1.8, 0))
            .r
        )
        i_modTexts.points.arrange_in_grid(
            n_cols=1, v_buff=0.5, aligned_edge=LEFT
        ).move_to((-3, -0.8, 0))
        self.play(
            FadeOut(i_title),
            TransformMatchingShapes(i_count, i_countAsCoord[: len(str(sol_n))]),
            FadeOut(i_dots),
            FadeIn(i_modTexts),
        )

        # 展示 3D 向量示意图

        i_tl = TL_CRT_3d().build().to_item().show()
        i_tlClipped = TransformableFrameClip(i_tl, offset=(0.15, 0.02))
        self.prepare(
            FadeOut(i_tl),
            FadeOut(i_tlClipped),
            FadeOut(i_modTexts),
            FadeOut(i_countAsCoord),
            at=i_tl.end - self.current_time - 1,
            duration=1,
        )
        self.show(i_tlClipped)
        self.forward_to(i_tl.end)
        self.hide(i_tl, i_tlClipped)

        i_solText = (
            TypstMath(f"n equiv {sol_n} quad mod {sol_gcd}")
            .points.move_to((0, -2.75, 0))
            .r
        )

        def playTimeline(tl):
            i_tl = tl.build().to_item(keep_last_frame=True).show()
            self.forward_to(i_tl.end)
            self.hide(i_tl)

        playTimeline(TL_CRT_RemainderAdd())
        playTimeline(TL_CRT_RemainderAdd(divisor=5, counts=(48, 31)))
        # playTimeline(TL_CRT_RemainderAdd(divisor=7, counts=(60, 62)))

        i_koujue = (
            TypstText(
                (
                    f'#text(fill: rgb("{RED}"))[*三*]人同行#text(fill: rgb("{RED}"))[*七十*]稀，'
                    f'#text(fill: rgb("{GREEN}"))[*五*]树梅花#text(fill: rgb("{GREEN}"))[*廿一*]枝。 \\\n'
                    f'#text(fill: rgb("{BLUE}"))[*七*]子团圆正#text(fill: rgb("{BLUE}"))[*半月*]，'
                    "除*百零五*便得知。"
                ),
            )
            .points.shift(UP * 2.75)
            .r
        )

        tableElements = {}
        for i, (divisor, uvec, color) in enumerate(zip(divisors, unitVecs, colors)):
            i_modText = Text(
                f"mod <c {color}>{divisor}</c>", format=Text.Format.RichText
            )
            i_uvecText = Text(f"{uvec}").color.set(color).r
            tableElements[f"divisor_{i}"] = i_modText
            tableElements[f"uvec_{i}"] = i_uvecText

            for j in range(len(divisors)):
                if i == j:
                    i_remText = Text("1").color.set(color).r
                else:
                    i_remText = Text("0")
                tableElements[f"rem_{i}_{j}"] = i_remText

        i_table = (
            TypstDoc(
                f"""
                #table(
                    [], {", ".join(f"divisor_{i}" for i in range(len(divisors)))},
                    {"\n".join(f"uvec_{i}, " + (", ".join(f"rem_{i}_{j}" for j in range(len(divisors)))) + "," for i in range(len(divisors)))}
                    columns: {len(divisors) + 1},
                    inset: (x: 20pt, y: 10pt),
                )
                """,
                vars=tableElements,
            )
            .points.to_center()
            .shift(DOWN * 0.25)
            .r
        )

        for i_ in tableElements.values():
            i_table.remove(i_)

        i_solProcessText = (
            TypstMath(
                f"n equiv {
                    '+'.join(
                        f'#text(fill: rgb("{color}"))[${uvec}$] times {rem}'
                        for uvec, rem, color in zip(unitVecs, remainders, colors)
                    )
                } "
                f'quad mod " " {
                    " times ".join(
                        f'#text(fill: rgb("{color}"))[${d}$]'
                        for d, color in zip(divisors, colors)
                    )
                }'
            )
            .points.shift((-1.8, -2.75, 0))
            .r
        )
        i_solText.points.shift(LEFT * 1.8)
        self.play(Write(i_koujue))
        self.play(Create(i_table))

        for i in range(len(divisors)):
            self.play(Write(tableElements[f"uvec_{i}"]), duration=0.5)
            self.forward(0.5)

        self.play(
            *(FadeIn(i_) for k, i_ in tableElements.items() if "divisor" in k),
            duration=0.5,
        )
        self.forward(0.5)

        for i in range(len(divisors)):
            for j in range(len(divisors)):
                self.show(tableElements[f"rem_{i}_{j}"])
                self.forward(0.1)
        self.forward(2)

        for i_ in tableElements.values():
            i_table.add(i_)

        i_modTexts.points.arrange_in_grid(
            n_cols=1, v_buff=0.5, aligned_edge=LEFT
        ).move_to((4.5, -0.25, 0))

        self.play(
            Group(i_table, i_koujue).anim(duration=1).points.shift(LEFT * 1.8),
            FadeIn(i_modTexts, duration=1),
        )
        self.forward(0.5)
        self.play(Write(i_solProcessText))
        self.forward(1)
        self.play(FadeOut(i_modTexts, duration=1))

        i_tl = (
            TL_CRT_3d(showUvecText=True, fadeOut=False)
            .build()
            .to_item(keep_last_frame=True)
            .show()
        )
        i_tlClipped = TransformableFrameClip(i_tl, offset=(0.25, 0.04))
        self.show(i_tlClipped)
        # self.play(i_text2.anim(duration=2).points.shift(LEFT * 1.8))
        self.forward_to(i_tl.end)

        self.forward(1)

        self.forward(1)
        self.play(Transform(i_solProcessText, i_solText))
        self.forward(2)


class TL_TalorSeries(Timeline):
    def __init__(
        self,
        coefsFactory=lambda: it.cycle((1, -1)),
        resultFn=lambda x: 1 / (x + 1),
        resultGraphConfig=pyr.m(discontinuities=(-1,)),
        maxDeg=7,
        *args,
        **kwargs,
    ):
        self._coefsFactory = coefsFactory
        self._resultFn = resultFn
        self._maxDeg = maxDeg
        self._resultGraphConfig = resultGraphConfig
        super().__init__(*args, **kwargs)

    def construct(self):
        coefsFactory = self._coefsFactory
        resultFn = self._resultFn
        maxDeg = self._maxDeg

        i_coord = Axes(
            num_sampled_graph_points_per_tick=5, axis_config=dict(tick_size=0.05)
        )

        self.show(i_coord)
        # self.play(FadeIn(i_coord))
        self.forward(2)

        polys = tuple(
            np.polynomial.Polynomial(tuple(it.islice(coefsFactory(), i + 1)))
            for i in range(maxDeg)
        )
        i_resultGraph = i_coord.get_graph(
            resultFn,
            stroke_color=BLUE,
            **self._resultGraphConfig,
        )
        i_polyGraph = i_coord.get_graph(polys[0], stroke_color=RED)
        self.play(FadeIn(i_resultGraph))
        self.play(Create(i_polyGraph))
        for poly in polys[1:]:
            self.play(
                Transform(
                    i_polyGraph,
                    i_polyGraph := i_coord.get_graph(poly, stroke_color=RED),
                ),
                duration=1,
            )
            self.forward(0.5)


class TL_Polynomial(Timeline):
    def construct(self):
        i_poly1 = (
            PolynomialText(
                nameSymbol="1 / (x + 1)",
                terms=(RationalPolyTermOpt(1 - (i % 2) * 2, i) for i in range(7)),
                aligns=dict(coef=0),
            )
            .points.to_center()
            .shift(UP * 2.75)
            .r
        )
        i_poly2 = (
            PolynomialText(
                nameSymbol='integral_0^x ("d"t) / (t + 1)',
                terms=(
                    RationalPolyTermOpt(Q(1 - (i % 2) * 2, i + 1), i + 1)
                    for i in range(7)
                ),
                aligns=dict(coef=0),
            )
            .mark.set(i_poly1.mark.get())
            .r.points.shift(DOWN * 1.25)
            .r
        )
        i_poly2NewSymbol = (
            TypstMath("ln(x + 1)").points.next_to(i_poly2.i_eq, LEFT, buff=0.15).r
        )
        i_integral = (
            TypstMath('integral_0^x ("d" t) / (t + 1) = ln(x + 1)')
            .points.shift((-2.5, -2, 0))
            .r
        )
        i_deriv = (
            TypstMath('("d") / ("d" x) ln(x + 1) = 1 / (x + 1)')
            .points.shift((2.5, -2, 0))
            .r
        )

        i_tl1 = (
            TL_TalorSeries(
                coefsFactory=lambda: it.cycle((1, -1)),
                resultFn=lambda x: 1 / (x + 1),
                resultGraphConfig=pyr.m(discontinuities=(-1,)),
                maxDeg=7,
            )
            .build()
            .to_item(keep_last_frame=True)
        )
        i_tl1Clipped = TransformableFrameClip(
            i_tl1, offset=(-0.225, -0.375), clip=(0.35, 0.1, 0.35, 0.45)
        )

        i_tl2 = (
            TL_TalorSeries(
                coefsFactory=lambda: it.chain(
                    (0,), ((-1 if i % 2 == 0 else 1) / i for i in it.count(1))
                ),
                resultFn=lambda x: np.log(x + 1),
                resultGraphConfig=pyr.m(x_range=(-0.99, 8)),
                maxDeg=8,
            )
            .build()
            .to_item(keep_last_frame=True)
        )
        i_tl2Clipped = TransformableFrameClip(
            i_tl2, offset=(0.125, -0.275), clip=(0.45, 0.2, 0.25, 0.35)
        )

        self.show(i_tl1, i_tl2, i_tl1Clipped, i_tl2Clipped)

        self.play(Write(i_poly1))
        self.forward(1)
        self.play(
            *(FadeIn(i_poly2.i_symbol[i]) for i in (0, 2, 3, 4, 5)),
            Transform(i_poly1.i_symbol[2:5], i_poly2.i_symbol[6:9], hide_src=False),
            Transform(i_poly1.i_symbol[1], i_poly2.i_symbol[1], hide_src=False),
            Transform(i_poly1.i_eq, i_poly2.i_eq, hide_src=False),
        )

        self.forward(1)
        for i_coef1, i_term1, i_sign1, i_coef2, i_term2, i_sign2 in zip(
            i_poly1.i_coefs,
            i_poly1.i_terms,
            i_poly1.i_signs,
            i_poly2.i_coefs,
            i_poly2.i_terms,
            i_poly2.i_signs,
        ):
            self.play(
                Transform(i_sign1, i_sign2, hide_src=False, flatten=True),
                Transform(i_coef1, i_coef2, hide_src=False, flatten=True),
                Transform(i_term1, i_term2, hide_src=False, flatten=True),
                duration=0.5,
            )
            self.forward(0.5)

        self.play(
            Transform(i_poly1.i_signs[-1], i_poly2.i_signs[-1], hide_src=False),
            Transform(i_poly1.i_ellipsis, i_poly2.i_ellipsis, hide_src=False),
            duration=0.5,
        )
        self.forward(1)
        self.play(Transform(i_poly2.i_symbol, i_poly2NewSymbol))
        self.play(Write(i_integral))
        self.play(Write(i_deriv))

        self.forward(2)


class TL_Test(Timeline):
    def construct(self):
        i_circ1 = Circle().stroke.set(alpha=0.5).r
        i_circ2 = Circle(2).points.shift(RIGHT * 3).r.stroke.set(alpha=0.5).r
        self.play(Create(i_circ1))
        self.play(Transform(i_circ1, i_circ2, hide_src=False))

        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
######################################
