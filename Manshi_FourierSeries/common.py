from pathlib import Path
from numbers import Rational
from collections.abc import Mapping

from janim.imports import *
from frozendict import frozendict

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
arrowConfig = dict(center_anchor="front", body_length=0.15, back_width=0.15)
config = Config(
    font=[
        "NewComputerModern10",
        "FandolSong",
    ],
    typst_shared_preamble=(DIR / "../assets/typst/manshi_preamble.typ").read_text(),
)


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


def addBgRect[I: VItem](item: I) -> I:
    item.add(
        SurroundingRect(
            item,
            color=BLACK,
            fill_alpha=0.75,
            stroke_alpha=0,
            depth=item.depth.get() + 0.001,
        ),
        insert=True,
    )
    return item


def getVecCreateAnim(item: Arrow, duration=1, arrowFadeInRatio=0.25, **kwargs):
    return AnimGroup(
        Create(item, root_only=True, duration=duration),
        Succession(
            Wait((1 - arrowFadeInRatio) * duration),
            FadeIn(item.tip, duration=arrowFadeInRatio * duration),
        ),
        **kwargs,
    )


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


class PolyDiagram(Axes):
    CONFIG = config

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


type Sgn = Literal[1, -1]

_polynomialDefaultWidths = frozendict(
    eq=0.6,
    symbol=1,
    coef=0.4,
    term=0.4,
    sign=0.5,
    ellipsis=1,
)

_polynomialDefaultAligns = frozendict(
    coef=-1,
    term=-1,
    symbol=1,
    sign=0,
    ellipsis=-1,
    eq=0,
)


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
        typstConfig: Mapping[str, Any] = frozendict(),
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
