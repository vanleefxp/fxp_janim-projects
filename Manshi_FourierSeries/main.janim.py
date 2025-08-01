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
import fantazia.synth.waveform as w

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


def nvec2d(v: Vect) -> Vect:
    x, y, *_ = v
    return np.array((-y, x, 0))


def unvec2d(v: Vect) -> Vect:
    nvec = nvec2d(v)
    return nvec / np.linalg.norm(nvec)


@lru_cache(maxsize=1 << 10)
def charCount(src: str, textType: type[Text | TypstDoc] = TypstMath):
    src = src.strip()
    if len(src) == 0:
        return 0
    return len(textType(src))


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


class MarkedText(Text, MarkedItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mark.set_points((ORIGIN,))


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


class RationalPolyTermOpt[N: Rational](PolyTermOptBase):
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


class FloatPolyTermOpt(PolyTermOptBase):
    def __new__(cls, coef: float, order: int, digits: int = 2) -> Self:
        self = super().__new__(cls)
        self._coef = coef
        self._order = order
        self._digits = digits
        return self

    @property
    def coef(self) -> float:
        return self._coef

    @property
    def digits(self) -> int:
        return self._digits

    @property
    def sign(self) -> Sgn:
        return 1 if self.coef >= 0 else -1

    @property
    def order(self) -> int:
        return self._order

    @property
    def coefSrc(self) -> str:
        return f"{self.coef:.{self.digits}f}"


class PolynomialText(Group[VItem], MarkedItem):
    def __init__(
        self,
        terms: Iterable[PolyTermOptBase] | None = None,
        unknownSymbol: str = "x",
        nameSymbol: str = "f(x)",
        typstConfig: Mapping[str, Any] = pyr.m(),
        widths: Mapping[str, Any] = _polynomialDefaultWidths,
        aligns: Mapping[str, Any] = _polynomialDefaultAligns,
        showEllipsis: bool = True,
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

        (
            self._widths,
            self._aligns,
        ) = widths, aligns

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
        i_signs = self._i_adds = Group()  # 符号
        self._coefPos = coefPos = []

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
            coefPos.append(pos)
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

        i_ellipsis = self._i_ellipsis = halign(
            MarkedTypstMath("...", **typstConfig),
            pos,
            widths["ellipsis"],
            aligns["ellipsis"],
        )
        if showEllipsis:
            self.add(*i_sign, *i_ellipsis)

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

    def putSymbol[I: MarkedItem](self, item: I) -> I:
        widths, aligns = self._widths, self._aligns
        halign(
            item,
            -widths["eq"] - widths["symbol"],
            widths["symbol"],
            aligns["symbol"],
        )
        item.mark.set(item.mark.get() + self.mark.get())
        return item

    def putCoef[I: MarkedItem](self, idx, item: I) -> I:
        widths, aligns = self._widths, self._aligns
        halign(
            item,
            self._coefPos[idx],
            widths["coef"],
            aligns["coef"],
        )
        item.mark.set(item.mark.get() + self.mark.get())
        return item


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


class PolyDiagram(Axes):
    def __init__(
        self,
        degree: int = 0,
        x_extent=3,
        y_extent=2.5,
        num_sampled_graph_points_per_tick=100,
        *args,
        **kwargs,
    ):
        super().__init__(
            x_range=(-x_extent, x_extent),
            y_range=(-y_extent, y_extent),
            num_sampled_graph_points_per_tick=num_sampled_graph_points_per_tick,
            *args,
            **kwargs,
            depth=-1,
        )

        for i_axis in self.get_axes():
            i_axis.ticks.set(stroke_radius=0.015)

        self._degree = degree

        x_right = x_extent
        if degree != 0:
            x_right = min(x_right, y_extent ** (1 / degree))

        i_border = SurroundingRect(
            self,
            buff=0,
            color=WHITE,
            fill_color=BLACK,
            fill_alpha=1,
            stroke_radius=0.015,
        )
        self._i_graph = i_graph = self.get_graph(
            lambda x: x**degree,
            x_range=(-x_right, x_right),
            color=RED,
            depth=-2,
            bind=False,
        )
        self.i_formula = i_formula = (
            MarkedTypstMath(
                "1" if degree == 0 else "x" if degree == 1 else f"x^{degree}",
                depth=-4,
            )
            .mark.set(i_border.points.box.get(DL) + (0.2, 0.2, 0))
            .r
        )
        i_formula.add(
            SurroundingRect(
                i_formula,
                buff=0.15,
                stroke_alpha=0,
                fill_color=BLACK,
                fill_alpha=0.5,
                depth=-3,
            )
        )

        self.add(i_border, i_graph, i_formula)

    @property
    def i_graph(self) -> ParametricCurve:
        return self._i_graph


class TL_Polynomial_Diagrams(Timeline):
    def construct(self):
        i_polyDiagrams = (
            Group(
                *(
                    PolyDiagram(
                        degree=i,
                        axis_config=dict(tick_size=0.05),
                        x_axis_config=dict(unit_size=0.5),
                        y_axis_config=dict(unit_size=0.5),
                    )
                    for i in range(8)
                )
            )
            .points.arrange_in_grid(n_cols=4, h_buff=0.5, v_buff=0.5)
            .to_center()
            .shift(UP * 0.5)
            .r
        )
        # self.show(i_polyDiagrams)
        self.play(
            AnimGroup(
                *(GrowFromPoint(i_, ORIGIN) for i_ in i_polyDiagrams),
                lag_ratio=0.1,
                duration=2.5,
            )
        )
        self.forward(2)
        self.play(
            AnimGroup(
                *(FadeOut(i_) for i_ in i_polyDiagrams), lag_ratio=0.1, duration=1
            )
        )


class TL_Polynomial(Timeline):
    CONFIG = config

    def construct(self):
        poly = np.polynomial.Polynomial((8, -2, -9, 2, 1))

        i_coord = (
            NumberPlane(
                x_range=(-8, 8, 1),
                y_range=(-64, 98, 16),
                depth=3,
                y_axis_config=dict(unit_size=1 / 16),
                background_line_style={"stroke_alpha": 0.75},
            )
            .points.shift((0.75, -0.25, 0))
            .r
        )
        i_graph = i_coord.get_graph(lambda _: 0, color=RED, depth=2)
        self.play(Create(i_coord))
        self.forward(1)

        i_diagrams = (
            Group(
                *(
                    PolyDiagram(
                        i,
                        x_extent=3,
                        y_extent=2.5,
                        x_axis_config=dict(unit_size=0.4),
                        y_axis_config=dict(unit_size=0.4),
                        axis_config=dict(tick_size=0.05),
                    )
                    for i in range(len(poly))
                )
            )
            .points.arrange_in_grid(n_rows=1, h_buff=0.25)
            .to_border(UP)
            .shift(UP * 0.25)
            .r
        )

        def createCoefText(coef: float, diagram: PolyDiagram) -> Text:
            i_text = (
                Text(
                    f"{coef:.2f}".replace("-", "\u2212"),
                    depth=-4,
                    stroke_alpha=1,
                    stroke_color=WHITE,
                    stroke_radius=0.005,
                )
                .points.scale(0.8)
                .next_to(diagram.points.box.get(DR), UL, buff=0.15)
                .r
            )
            i_text.add(
                SurroundingRect(
                    i_text,
                    buff=0.1,
                    stroke_alpha=0,
                    stroke_color=BLACK,
                    fill_color=BLACK,
                    fill_alpha=0.5,
                    depth=-3,
                )
            )
            return i_text

        i_coefTexts = Group(
            *(createCoefText(poly.coef[i], i_diagrams[i]) for i in range(len(poly)))
        )
        self.play(
            AnimGroup(*(FadeIn(i_) for i_ in i_diagrams), lag_ratio=0.25, duration=1)
        )
        self.forward(1)
        self.play(Create(i_graph))

        def createPolyGraphUpdaterFn(deg):
            def updaterFn(params: UpdaterParams) -> ParametricCurve:
                t = params.alpha
                interpPoly = np.polynomial.Polynomial(
                    np.append(poly.coef[:deg], poly.coef[deg] * t)
                )
                return i_coord.get_graph(interpPoly, color=RED, depth=2)

            return updaterFn

        def createCoefTextUpdateFn(deg):
            def updaterFn(params: UpdaterParams) -> Text:
                t = params.alpha
                coef = poly.coef[deg] * t
                return createCoefText(coef, i_diagrams[deg])

            return updaterFn

        for i in range(len(poly)):
            self.play(
                ItemUpdater(i_graph, createPolyGraphUpdaterFn(i)),
                ItemUpdater(i_coefTexts[i], createCoefTextUpdateFn(i)),
                duration=2,
            )
        self.forward(2)


class TL_Talor_Diagram(Timeline):
    _defaultPauses = pyr.m(start=0, beforeShowPoly=1)

    def __init__(
        self,
        coefsFactory=lambda: it.cycle((1, -1)),
        resultFn=lambda x: 1 / (x + 1),
        resultGraphConfig=pyr.m(x_range=(-0.8, 5)),
        maxDeg=7,
        cropRadius=(2, 1.5),
        coordShift=(-0.5, -1),
        pauses=_defaultPauses,
        showResultFirst=True,
        *args,
        **kwargs,
    ):
        self._coefsFactory = coefsFactory
        self._resultFn = resultFn
        self._maxDeg = maxDeg
        self._resultGraphConfig = resultGraphConfig
        self._cropRadius = cropRadius
        self._coordShift = coordShift
        self._pauses = dict(self._defaultPauses)
        self._pauses.update(pauses)
        self.showResultFirst = showResultFirst
        super().__init__(*args, **kwargs)

    def construct(self):
        coefsFactory = self._coefsFactory
        resultFn = self._resultFn
        maxDeg = self._maxDeg
        cropRadius = self._cropRadius
        coordShift = self._coordShift
        pauses = self._pauses
        showResultFirst = self.showResultFirst

        i_coord = (
            Axes(num_sampled_graph_points_per_tick=5, axis_config=dict(tick_size=0.05))
            .points.shift((*coordShift, 0))
            .r
        )
        i_cropRect = Rect(
            *np.array(cropRadius) * 2, stroke_radius=0.03, stroke_color=WHITE
        )
        self.forward(pauses["start"])
        self.play(FadeIn(Group(i_cropRect, i_coord)))

        polys = tuple(
            np.polynomial.Polynomial(tuple(it.islice(coefsFactory(), i + 1)))
            for i in range(maxDeg)
        )
        i_resultGraph = i_coord.get_graph(
            resultFn,
            stroke_color=WHITE,
            stroke_alpha=0.5,
            **self._resultGraphConfig,
        )
        polyXRange = (
            i_coord.p2c((-cropRadius[0], 0, 0))[0],
            i_coord.p2c((cropRadius[0], 0, 0))[0],
        )
        i_polyGraph = i_coord.get_graph(polys[0], stroke_color=RED, x_range=polyXRange)

        if showResultFirst:
            self.play(Create(i_resultGraph), duration=1)
            self.forward(pauses["beforeShowPoly"])

        self.play(Create(i_polyGraph), duration=1)
        self.forward(0.5)
        for poly in polys[1:]:
            self.play(
                Transform(
                    i_polyGraph,
                    i_polyGraph := i_coord.get_graph(
                        poly, stroke_color=RED, x_range=polyXRange
                    ),
                ),
                duration=1,
            )
            self.forward(0.5)

        if not showResultFirst:
            self.play(Create(i_resultGraph), duration=1)

    @property
    def cropParams(self) -> tuple[float, float, float, float]:
        crx, cry = self._cropRadius
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius
        print(crx, cry, rx, ry)
        cropX, cropY = (rx - crx) / rx / 2, (ry - cry) / ry / 2
        print(cropX, cropY)
        return (cropX, cropY, cropX, cropY)


class TL_Talor(Timeline):
    def construct(self):
        i_poly1 = (
            PolynomialText(
                nameSymbol="1 / (x + 1)",
                terms=(RationalPolyTermOpt(1 - (i % 2) * 2, i) for i in range(7)),
                aligns=dict(coef=0),
            )
            .points.to_center()
            .shift(UP * 3)
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
        i_convergenceRadius = (
            TypstMath("(-1 < x < 1)", fill_alpha=0.9)(VItem)
            .points.scale(0.8)
            .next_to(i_poly1, DOWN, buff=0.2, aligned_edge=RIGHT)
            .r
        )
        i_poly2NewSymbol = i_poly2.putSymbol(MarkedTypstMath("ln(x + 1)"))
        i_integral = (
            TypstMath('integral_0^x ("d" t) / (t + 1) = ln(x + 1)')
            .points.shift(DOWN * 0.25)
            .r
        )
        i_deriv = (
            TypstMath('("d") / ("d" x) ln(x + 1) = 1 / (x + 1)')
            .points.shift(DOWN * 2.25)
            .r
        )

        arrowRadius = 1.8
        i_arrows = (
            Group(
                Arrow(
                    LEFT * arrowRadius,
                    RIGHT * arrowRadius,
                    tip_kwargs=arrowConfig,
                    stroke_radius=0.015,
                ),
                Arrow(
                    RIGHT * arrowRadius,
                    LEFT * arrowRadius,
                    tip_kwargs=arrowConfig,
                    stroke_radius=0.015,
                ),
            )
            .points.arrange_in_grid(n_cols=1, v_buff=0.1)
            .shift(DOWN * 1.25)
            .r
        )

        tl1 = TL_Talor_Diagram(
            coefsFactory=lambda: it.cycle((1, -1)),
            resultFn=lambda x: 1 / (x + 1),
            resultGraphConfig=pyr.m(x_range=(-0.8, 5)),
            maxDeg=7,
            coordShift=(-0.5, -1),
        )
        tl2 = TL_Talor_Diagram(
            coefsFactory=lambda: it.chain(
                (0,), ((-1 if i % 2 == 0 else 1) / i for i in it.count(1))
            ),
            resultFn=lambda x: np.log(x + 1),
            resultGraphConfig=pyr.m(x_range=(-0.8, 5)),
            maxDeg=8,
            coordShift=(-0.75, 0),
            pauses=dict(
                start=15.5,
            ),
            showResultFirst=False,
        )

        i_tl1 = tl1.build().to_item(keep_last_frame=True)
        i_tl1Clipped = TransformableFrameClip(
            i_tl1, offset=(-0.225, -0.15), clip=tl1.cropParams
        )

        i_tl2 = tl2.build().to_item(keep_last_frame=True)
        i_tl2Clipped = TransformableFrameClip(
            i_tl2, offset=(0.225, -0.15), clip=tl2.cropParams
        )

        self.show(i_tl1, i_tl2, i_tl1Clipped, i_tl2Clipped)

        self.play(
            Write(Group(*i_poly1.i_symbol, i_poly1.i_eq)), FadeIn(i_convergenceRadius)
        )
        self.forward(2)
        for i_sign, i_coef, i_term in zip(
            i_poly1.i_signs, i_poly1.i_coefs, i_poly1.i_terms
        ):
            self.play(*(FadeIn(i_) for i_ in (i_sign, i_coef, i_term)), duration=0.5)
            self.forward(1)
        self.play(FadeIn(i_poly1.i_ellipsis), FadeIn(i_poly1.i_signs[-1]))

        self.forward(1)
        self.play(
            *(FadeIn(i_poly2.i_symbol[i]) for i in (0, 2, 3, 4, 5)),
            Transform(i_poly1.i_symbol[2:5], i_poly2.i_symbol[6:9], hide_src=False),
            Transform(i_poly1.i_symbol[1], i_poly2.i_symbol[1], hide_src=False),
            Transform(i_poly1.i_eq, i_poly2.i_eq, hide_src=False),
            i_convergenceRadius.anim.points.next_to(
                i_poly2, DOWN, buff=0.2, aligned_edge=RIGHT
            ),
        )

        self.forward(1.5)
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
            self.forward(1)

        self.play(
            Transform(i_poly1.i_signs[-1], i_poly2.i_signs[-1], hide_src=False),
            Transform(i_poly1.i_ellipsis, i_poly2.i_ellipsis, hide_src=False),
            duration=0.5,
        )
        self.forward(1)
        self.play(Transform(i_poly2.i_symbol, i_poly2NewSymbol))
        self.forward(2)
        self.play(
            i_tl1Clipped.anim.clip.set(x_offset=-0.29),
            i_tl2Clipped.anim.clip.set(x_offset=0.29),
        )
        self.play(Write(i_integral, duration=1.5), Write(i_arrows[0], duration=1.5))
        self.forward(1)
        self.play(Write(i_deriv[::-1], duration=1.5), Write(i_arrows[1], duration=1.5))

        self.forward(2)


class TL_SineDeriv(Timeline):
    CONFIG = config

    def construct(self):
        def createCoord():
            i_coord = Axes(
                x_range=(-PI * 2, PI * 2, PI / 2),
                y_range=(-1.5, 1.5, 1),
                y_axis_config=dict(unit_size=0.75),
                x_axis_config=dict(unit_size=0.5),
                axis_config=dict(tick_size=0.05),
            )
            for i_axis in i_coord.get_axes():
                i_axis.ticks.set(stroke_radius=0.015)
            i_coord.add(
                SurroundingRect(i_coord, buff=0, color=WHITE, stroke_radius=0.015)
            )
            return i_coord

        i_coords = (
            Group(*(createCoord() for _ in range(4)))
            .points.arrange_in_grid(n_cols=2, h_buff=0.5, v_buff=1)
            .shift(UP * 0.75)
            .r
        )

        i_1 = i_coords[-1]
        i_2 = i_coords[-2]
        i_coords.remove(i_1, i_2)
        i_coords.add(i_1, i_2)

        functions = (np.sin, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x))
        colors = (RED, GREEN, BLUE, PINK)
        i_graphs = Group(
            *(i_.get_graph(f, color=c) for i_, f, c in zip(i_coords, functions, colors))
        )
        i_formulae = Group(
            TypstMath("sin x"),
            TypstMath("(sin x)' = cos x"),
            TypstMath("(sin x)''  = -sin x"),
            TypstMath("(sin x)''' = -cos x"),
        )

        for i_coord, i_formula in zip(i_coords, i_formulae):
            i_formula.points.move_to(i_coord.points.box.bottom).shift(DOWN * 0.5)

        i_lastCoord = None
        i_lastGraph = None

        def createTangentUpdaterFn(item: VItem) -> ItemUpdaterFn:
            def updaterFn(params: UpdaterParams) -> TangentLine:
                return TangentLine(item, params.alpha)

            return updaterFn

        def createDotUpdaterFn(i_graph: VItem) -> GroupUpdaterFn:
            def updaterFn(i_dot: Dot, params: UpdaterParams) -> None:
                i_dot.points.move_to(i_graph.points.pfp(params.alpha))

            return updaterFn

        def createLineUpdaterFn(
            i_graph: VItem, i_coord: CoordinateSystem
        ) -> GroupUpdaterFn:
            def updaterFn(i_line: Line, params: UpdaterParams) -> None:
                pointOnGraph = i_graph.points.pfp(params.alpha)
                x, _ = i_coord.p2c(pointOnGraph)
                pointOnAxis = i_coord.c2p(x, 0)
                i_line.points.put_start_and_end_on(pointOnGraph, pointOnAxis)

            return updaterFn

        for i_coord, i_graph, i_formula in zip(i_coords, i_graphs, i_formulae):
            if i_lastGraph is None:
                self.play(FadeIn(i_coord))
                self.play(Create(i_graph, duration=2), Write(i_formula))
            else:
                self.play(Transform(i_lastCoord, i_coord, hide_src=False, flatten=True))
                i_tangent = TangentLine(i_lastGraph, 0)
                i_dotOnGraph = Dot(i_graph.points.pfp(0), radius=0.06)
                i_dotOnLastGraph = Dot(i_lastGraph.points.pfp(0), radius=0.06)
                i_lineOnGraph = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)
                i_lineOnLastGraph = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)
                self.play(
                    *(
                        FadeIn(i_)
                        for i_ in (
                            i_tangent,
                            i_dotOnGraph,
                            i_dotOnLastGraph,
                            i_lineOnGraph,
                            i_lineOnLastGraph,
                        )
                    ),
                    duration=0.5,
                )
                self.play(
                    Create(i_graph),
                    ItemUpdater(i_tangent, createTangentUpdaterFn(i_lastGraph)),
                    GroupUpdater(i_dotOnGraph, createDotUpdaterFn(i_graph)),
                    GroupUpdater(i_dotOnLastGraph, createDotUpdaterFn(i_lastGraph)),
                    GroupUpdater(i_lineOnGraph, createLineUpdaterFn(i_graph, i_coord)),
                    GroupUpdater(
                        i_lineOnLastGraph, createLineUpdaterFn(i_lastGraph, i_lastCoord)
                    ),
                    duration=3,
                )
                self.play(
                    *(
                        FadeOut(i_)
                        for i_ in (
                            i_tangent,
                            i_dotOnGraph,
                            i_dotOnLastGraph,
                            i_lineOnGraph,
                            i_lineOnLastGraph,
                        )
                    ),
                    FadeIn(i_formula),
                    duration=0.5,
                )
            self.forward(1)
            i_lastGraph = i_graph
            i_lastCoord = i_coord

        i_sinGraph, i_negSinGraph = i_graphs[0], i_graphs[2]

        self.play(i_sinGraph.anim.glow.set(color=colors[0], alpha=0.5, size=0.75))
        self.forward(0.5)
        self.play(i_negSinGraph.anim.glow.set(color=colors[2], alpha=0.5, size=0.75))
        self.forward(0.5)
        self.play(ShowPassingFlashAround(i_formulae[2][-5], time_width=3, duration=2))
        self.forward(2)

        i_dotOnGraph1 = Dot(i_sinGraph.points.pfp(0), radius=0.06)
        i_dotOnGraph2 = Dot(i_negSinGraph.points.pfp(0), radius=0.06)
        i_lineOnGraph1 = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)
        i_lineOnGraph2 = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)

        self.play(
            i_sinGraph.anim.glow.set(alpha=0).r.stroke.set(alpha=0.5),
            i_negSinGraph.anim.glow.set(alpha=0).r.stroke.set(alpha=0.5),
            *(
                FadeIn(i_)
                for i_ in (i_dotOnGraph1, i_dotOnGraph2, i_lineOnGraph1, i_lineOnGraph2)
            ),
            duration=0.5,
        )
        self.forward(0.5)

        i_sinGraphCp = i_sinGraph.copy().stroke.set(alpha=1).r
        i_negSinGraphCp = i_negSinGraph.copy().stroke.set(alpha=1).r

        self.play(
            Create(i_sinGraphCp),
            Create(i_negSinGraphCp),
            GroupUpdater(i_dotOnGraph1, createDotUpdaterFn(i_sinGraph)),
            GroupUpdater(i_dotOnGraph2, createDotUpdaterFn(i_negSinGraph)),
            GroupUpdater(i_lineOnGraph1, createLineUpdaterFn(i_sinGraph, i_coords[0])),
            GroupUpdater(
                i_lineOnGraph2, createLineUpdaterFn(i_negSinGraph, i_coords[2])
            ),
            duration=3,
        )
        self.forward(0.5)
        self.play(
            *(
                FadeOut(i_)
                for i_ in (i_dotOnGraph1, i_dotOnGraph2, i_lineOnGraph1, i_lineOnGraph2)
            ),
            duration=0.5,
        )
        # self.play(Transform(i_graphs[0], i_graphs[2], hide_src=False), duration=2)

        self.forward(2)


class TL_Waveform(Timeline):
    def __init__(self, waveform: w.Waveform = w.square) -> None:
        super().__init__()
        self._waveform = waveform

    def construct(self):
        waveform = self._waveform
        n_harmonics = 15
        smallAmpThresh = 1e-4

        i_coord = (
            Axes(
                x_range=(0, 1, 0.5),
                y_range=(-1, 1, 1),
                axis_config=dict(tick_size=0.05),
                x_axis_config=dict(unit_size=self.config_getter.frame_x_radius * 2),
                y_axis_config=dict(unit_size=2),
                num_sampled_graph_points_per_tick=100,
            )
            .points.move_to(ORIGIN)
            .r
        )
        for i_axis in i_coord.get_axes():
            i_axis.ticks.set(stroke_radius=0.015)
        i_coord.remove(i_coord.get_axes()[1])
        i_waveformGraph = i_coord.get_graph(
            waveform,
            stroke_color="#00ff00",
            glow_color="#00ff00",
            glow_alpha=0.5,
            glow_size=0.5,
            x_range=(-0.01, 1.01),
        )

        self.play(Create(i_coord))
        self.play(Create(i_waveformGraph))
        self.forward(1)
        self.play(FadeOut(i_waveformGraph))
        self.forward(1)

        i_harmonicGraphs = Group(
            *(
                i_coord.get_graph(partial(waveform.h, k), stroke_color="#00ff00")
                for k in range(1, n_harmonics + 1)
            )
        )
        i_lastCumulativeGraph = None
        for k, (coef, i_harmonicGraph) in enumerate(
            zip(it.islice(waveform, 1, None), i_harmonicGraphs), 1
        ):
            if abs(coef) < smallAmpThresh:
                continue
            i_cumulativeGraph = i_coord.get_graph(waveform[:k], stroke_color="#00ff00")
            self.play(Create(i_harmonicGraph), duration=1)
            if i_lastCumulativeGraph is None:
                self.play(Transform(i_harmonicGraph, i_cumulativeGraph), duration=1)
            else:
                self.play(
                    Transform(i_harmonicGraph, i_cumulativeGraph),
                    Transform(i_lastCumulativeGraph, i_cumulativeGraph),
                    duration=1,
                )
            # self.forward(1)
            i_lastCumulativeGraph = i_cumulativeGraph

        self.play(Transform(i_lastCumulativeGraph, i_waveformGraph))
        self.forward(1)

        for i_ in i_harmonicGraphs:
            i_.stroke.set(alpha=0)
        self.show(i_harmonicGraphs)

        distance = 1
        a_flip = Audio(DIR / "assets/sound/flip.mp3")
        a_penClick = Audio(DIR / "assets/sound/pen-click.wav")
        i_cameraInitState = self.camera.store()
        self.play_audio(a_flip)
        self.play(
            *(
                i_graph.update.points.shift(IN * distance * (i + 1))
                .r.stroke.set(alpha=0.5)
                .r
                for i, i_graph in enumerate(i_harmonicGraphs)
            ),
            self.camera.anim.points.rotate(-PI / 6, axis=RIGHT, about_point=ORIGIN)
            .rotate(PI / 6, axis=UP, about_point=ORIGIN)
            .shift((4, 1.72, 1.5))
            .r,
        )

        totalTime = 3
        incFactor = 1.1
        firstTime = totalTime / (incFactor**n_harmonics - 1) * (incFactor - 1)

        self.forward(0.5)
        self.play_audio(a_penClick)
        self.play(i_harmonicGraphs[0].update.stroke.set(alpha=1), duration=0.1)
        self.forward(firstTime)
        for i, (i_lastGraph, i_graph) in enumerate(it.pairwise(i_harmonicGraphs), 1):
            self.play_audio(a_penClick)
            i_lastGraph.stroke.set(alpha=0.5)
            i_graph.stroke.set(alpha=1)
            self.forward(firstTime * incFactor**i)
        self.play(i_graph.update.stroke.set(alpha=0.5), duration=0.1)
        self.forward(2)

        self.play_audio(a_flip)
        self.play(
            self.camera.anim.restore(i_cameraInitState),
            *(
                i_graph.update.points.shift(OUT * distance * (i + 1))
                .r.stroke.set(alpha=0)
                .r
                for i, i_graph in enumerate(i_harmonicGraphs)
            ),
        )
        self.hide(i_harmonicGraphs)

        self.forward(2)


class TL_VecDotProduct(Timeline):
    CONFIG = config

    def construct(self) -> None:
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius
        i_coord = (
            NumberPlane(
                x_range=(-10, 10),
                y_range=(-6, 6),
                background_line_style=dict(stroke_alpha=0.75),
                x_axis_config=dict(unit_size=0.75),
                y_axis_config=dict(unit_size=0.75),
                depth=2,
            )
            .points.shift((-3, 0, 0))
            .r
        )

        i_dotProductFormula = (
            TypstMath(
                f"""
                angle.l
                text(arrow.tr, fill: #rgb("{RED}")),
                text(arrow.br, fill: #rgb("{GREEN}"))
                angle.r =
                """,
                depth=-1,
            )
            .points.to_border(UL)
            .r
        )
        i_dotProductFormula[1].set(
            stroke_alpha=1, stroke_radius=0.005, stroke_color=RED
        )
        i_dotProductFormula[3].set(
            stroke_alpha=1, stroke_radius=0.005, stroke_color=GREEN
        )

        def createDotProductValueText(value: float) -> Text:
            i_dotProductValue = (
                Text(f"{value:.2f}".replace("-", "\u2212"), depth=-1)
                .points.next_to(i_dotProductFormula, RIGHT, buff=0.2)
                .r
            )
            return i_dotProductValue

        i_dotProductValue = createDotProductValueText(0)
        i_bgRect = SurroundingRect(
            Group(i_dotProductFormula, i_dotProductValue),
            buff=0.2,
            fill_color=BLACK,
            fill_alpha=0.75,
            stroke_alpha=0,
            stroke_color=BLACK,
            stroke_radius=0,
            depth=1,
        )

        def createValueUpdaterFn(start1, end1, start2, end2):
            def updaterFn(params: UpdaterParams) -> None:
                t = params.alpha
                p1 = interpolate(start1, end1, t)
                p2 = interpolate(start2, end2, t)
                value = np.dot(p1, p2)
                return createDotProductValueText(value)

            return updaterFn

        def createVecUpdaterFn(start, end):
            start = np.array(start)
            end = np.array(end)

            def updaterFn(item: Arrow, params: UpdaterParams) -> None:
                t = params.alpha
                point = interpolate(start, end, t)
                item.points.put_start_and_end_on(coordOrigin, i_coord.c2p(*point))

            return updaterFn

        origin = np.array((0, 0))
        vec1 = np.array((2, 2))
        vec2 = np.array((4, -1))

        coordOrigin = i_coord.c2p(0, 0)
        i_vec1 = Arrow(coordOrigin, coordOrigin, buff=0, color=RED)
        i_vec2 = Arrow(coordOrigin, coordOrigin, buff=0, color=GREEN)

        def vecAnim(start1, end1, start2, end2):
            return AnimGroup(
                GroupUpdater(i_vec1, createVecUpdaterFn(start1, end1)),
                GroupUpdater(i_vec2, createVecUpdaterFn(start2, end2)),
                ItemUpdater(
                    i_dotProductValue, createValueUpdaterFn(start1, end1, start2, end2)
                ),
            )

        i_formulae = (
            TypstDoc(
                (DIR / "assets/typst/vec-dot-product-properties.typ").read_text(
                    encoding="utf-8"
                ),
                depth=-1,
            )
            .points.move_to((4.8, 0.25, 0))
            .r
        )
        i_formula1 = i_formulae[0:24]
        i_formula2 = i_formulae[24:38]
        i_formula3 = i_formulae[38:]

        i_formulaeBackground = Rect(
            (rx / 3, -ry - 0.25, 0),
            (rx + 0.25, ry + 0.25, 0),
            fill_color=BLACK,
            fill_alpha=0.75,
            stroke_color=WHITE,
            stroke_alpha=0.75,
            stroke_radius=0.01,
            depth=1,
        )

        self.play(Create(i_coord), FadeIn(i_formulaeBackground))

        self.play(
            FadeIn(i_bgRect, duration=1),
            Succession(
                Write(i_dotProductFormula, duration=0.75),
                FadeIn(i_dotProductValue, duration=0.25),
                duration=1,
            ),
        )

        self.show(i_vec2, i_vec1)
        self.play(vecAnim(origin, vec1, origin, vec2), duration=2)
        self.forward(2)

        # 双线性性

        def createMultText(vec, mult, direction=1) -> Text:
            direction = 1 if direction >= 0 else -1
            vec = np.array(vec)
            nvec = unvec2d(vec)
            x, y, *_ = vec
            dirAngle = np.atan2(y, x)
            i_multText = (
                Text(f"{mult:.2f}×".replace("-", "\u2212"), depth=-1)
                .points.move_to(coordOrigin)
                .r
            )
            (
                i_multText.add(
                    SurroundingRect(
                        i_multText,
                        buff=0.1,
                        stroke_color=BLACK,
                        stroke_radius=0,
                        fill_alpha=0.75,
                        fill_color=BLACK,
                    )
                )
                .points.rotate(dirAngle, about_point=coordOrigin)
                .move_to(i_coord.c2p(*(vec * (mult / 2))))
                .shift(nvec * (0.75 * direction))
            )
            return i_multText

        def animateVecScale(idx, mults=(2,)):
            direction = 1 if idx == 0 else -1
            i_vec = i_vec1 if idx == 0 else i_vec2
            i_vecCp = i_vec.copy().set(fill_alpha=0.75, stroke_alpha=0.75)
            vec = vec1 if idx == 0 else vec2

            unvec = unvec2d(vec)
            i_brace = Brace(i_vec, unvec * direction)
            i_multText = createMultText(vec, 1, direction)
            self.play(
                *(FadeIn(i_) for i_ in (i_brace, i_multText, i_vecCp)), duration=0.5
            )
            self.play(
                ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=2)
            )
            self.forward(0.5)
            for lastMult, mult in it.pairwise(it.chain((1,), mults, (1,))):
                self.play(
                    vecAnim(vec1 * lastMult, vec1 * mult, vec2, vec2)
                    if idx == 0
                    else vecAnim(vec1, vec1, vec2 * lastMult, vec2 * mult),
                    DataUpdater(
                        i_brace, lambda i_, _: i_.points.match(i_vec.current())
                    ),
                    ItemUpdater(
                        i_multText,
                        lambda params, lastMult=lastMult, mult=mult: createMultText(
                            vec, interpolate(lastMult, mult, params.alpha), direction
                        ),
                    ),
                    duration=0.75,
                )
                self.play(
                    ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=1)
                )
                self.forward(0.5)
            self.play(
                *(FadeOut(i_) for i_ in (i_brace, i_multText, i_vecCp)), duration=0.5
            )

        animateVecScale(0, (2,))
        self.forward(1)
        animateVecScale(1, (1.5,))
        self.play(Write(i_formula1), duration=1)
        self.forward(2)

        # 对称性
        self.play(ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=2))
        self.forward(0.5)
        self.play(vecAnim(vec1, vec2, vec2, vec1), duration=2)
        self.play(ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=1))
        self.forward(0.5)
        self.play(vecAnim(vec2, vec1, vec1, vec2), duration=2)
        self.play(ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=1))
        self.play(Write(i_formula2), duration=1)
        self.forward(2)

        # 正定性
        self.play(vecAnim(vec1, vec1, vec2, vec1))
        self.play(
            i_vec2.anim.set(stroke_alpha=0, fill_alpha=0),
            i_dotProductFormula[3]
            .anim.points.rotate(PI / 2)
            .r.fill.set(color=RED)
            .r.stroke.set(color=RED),
            duration=0.5,
        )
        self.forward(1)

        targetPoints = np.array(((4, -3), (-4, -2), (-3, 3), vec1))
        for lastP, p in it.pairwise(it.chain((vec1,), targetPoints)):
            self.play(vecAnim(lastP, p, lastP, p), duration=1)
            self.forward(0.5)

        self.play(vecAnim(p, origin, p, origin), duration=2)
        self.play(Write(i_formula3), duration=1)

        self.forward(2)


class TL_Test(Timeline):
    def construct(self):
        i_arrow = Arrow(ORIGIN, ORIGIN, buff=0)
        self.play(Create(i_arrow))
        i_arrow.points.put_start_and_end_on(ORIGIN, np.array((3, 1, 0)))
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
