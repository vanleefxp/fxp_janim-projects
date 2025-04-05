from collections.abc import Collection
from pathlib import Path
import itertools as it
import operator as op

from janim.imports import *
import numpy as np

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


def createFlag(size=0.35, color=RED, rotation=0):
    i_flagStem = Line((0, 0, 0), (0, size, 0))
    i_flagFace = Polygon((0, size, 0), (0, size * 0.6, 0), (size * 0.6, size * 0.8, 0))
    i_flagStem.radius.set(0.015)
    i_flagStem.stroke.set(color=color)
    i_flagFace.radius.set(0.015)
    i_flagFace.stroke.set(color=color)
    i_flagFace.fill.set(color=color, alpha=1)
    i_flag = Group(i_flagStem, i_flagFace)
    i_flag.points.rotate(rotation, about_point=ORIGIN)
    i_flag.depth.set(-1)
    return i_flag


def createFocusBox(radius=0.16, ratio=0.75):
    i_focusBox = Group()
    i_topRight = Polyline(
        (radius, radius * (1 - ratio), 0),
        (radius, radius, 0),
        (radius * (1 - ratio), radius, 0),
    )
    i_topRight.radius.set(0.01)
    for i in range(4):
        i_corner = i_topRight.copy()
        i_corner.points.rotate(i * PI / 2, about_point=ORIGIN)
        i_focusBox.add(i_corner)
    return i_focusBox


def generateCatalanTable(n: int = 10):
    dp = np.zeros((n + 1, n + 1), dtype=int)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            dp[i, j] = dp[i - 1][j] + dp[i][j - 1]
    return dp


def _powerSeriesTermCharCount(n: int, idxOffset: int = 0):
    # 获取幂级数每一项的长度
    # 返回系数部分和幂次部分分别的长度
    if n == 0:
        return len(str(idxOffset)) + 1, 0
    elif n == 1:
        return len(str(1 + idxOffset)) + 1, 1
    else:
        return len(str(n + idxOffset)) + 1, len(str(n)) + 1


def createPowerSeries(
    n: int = 4,
    prefix: str = "f(x) =",
    coefSym: str = "a",
    termSym: str = "x",
    startPow: int = 0,
    idxOffset: int = 0,
    align: bool = True,
    prefixPad: float = 0.3,
    coefStep: float = 0.4,
    termStep: float = 0.4,
    signStep: float = 0.5,
    startX: float = -4,
    **kwargs,
) -> TypstMath:
    # 创建公式对象
    src = f"{prefix} "
    if n >= 0 and startPow <= 0:
        src += f"{coefSym}_{idxOffset} + "
    if n >= 1 and startPow <= 1:
        src += f"{coefSym}_{1 + idxOffset} {termSym} + "
    for i in range(max(2, startPow), n + startPow + 1):
        src += f"{coefSym}_{i + idxOffset} {termSym}^{i} + "
    src += "..."
    i_text = TypstMath(src, **kwargs)

    # 子对象切分
    i_text.i_coefs = i_coefs = Group()
    i_text.i_terms = i_terms = Group()
    i_text.i_others = i_others = Group()
    i_text.i_gridItems = i_gridItems = Group()
    start = 0
    end = len(TypstMath(prefix))
    # f(x) =
    i_text.i_prefix = i_prefix = i_text[:end]
    i_others.add(i_prefix)
    i_gridItems.add(i_prefix)
    for i in range(startPow, n + startPow + 1):
        start = end
        l1, l2 = _powerSeriesTermCharCount(i, idxOffset)
        end += l1 + l2 + 1
        mid = start + l1
        # 幂级数的系数 a_n
        i_coefs.add(i_text[start:mid])
        i_gridItems.add(i_text[start:mid])
        # 幂级数的指数 x^n
        i_terms.add(i_text[mid : end - 1])
        i_gridItems.add(i_text[mid : end - 1])
        # 加号
        i_others.add(i_text[end - 1])
        i_gridItems.add(i_text[end - 1])

    i_text.i_ellipsis = i_ellipsis = i_text[end:]
    i_gridItems.add(i_ellipsis)

    if align:
        # 手动对齐
        i_text.alignParams = alignParams = lambda: None
        alignParams.prefixPad = prefixPad
        alignParams.coefStep = coefStep
        alignParams.termStep = termStep
        alignParams.signStep = signStep
        alignParams.totalStep = coefStep + termStep + signStep

        # 前缀
        i_gridItems[0].points.set_x(-prefixPad + startX, RIGHT)
        align_x = startX
        for i in range(1, len(i_gridItems)):
            i_gridItems[i].points.set_x(align_x, LEFT)
            match i % 3:
                case 0:  # 系数
                    align_x += coefStep
                case 1:  # x 的幂次
                    align_x += termStep
                case 2:  # 加号
                    align_x += signStep

    return i_text


def createPsBoxes(i_ps: TypstMath, **kwargs):
    i_boxes = Group()
    for i_coef in i_ps.i_coefs:
        i_boxes.add(SurroundingRect(i_coef, **kwargs))
    return i_boxes


def _coefSrcGenerator(
    i: int, coef1Sym: str = "a", coef2Sym: str = "b", useSqr: bool = True
) -> Iterable[str]:
    if coef1Sym == coef2Sym:
        for j in range(i + 1):
            if i == j * 2 and useSqr:
                yield f"{coef1Sym}_{j}^2"
            else:
                yield f"{coef1Sym}_{j} {coef1Sym}_{i-j}"
    else:
        for j in range(i + 1):
            yield f"{coef1Sym}_{j} {coef2Sym}_{i-j}"


def _psProdCoefChCount(i, j, useSqr=False):
    if useSqr and i == j * 2:
        return 2 + len(str(i)), 1
    return 1 + len(str(i)), 1 + len(str(i - j))


def _psProdTermChCount(i):
    if i == 1:
        return 1
    return 1 + len(str(i))


def createPowerSeriesProduct(
    n: int = 8,
    prefix: str = "&f(x) g(x) &&=",
    coef1Sym: str = "a",
    coef2Sym: str = "b",
    termSym: str = "x",
    useSqr: bool = True,
    **kwargs,
):
    # 生成公式对象
    src = prefix
    if n > 0:
        if coef1Sym == coef2Sym and useSqr:
            src += f"&& &&{coef1Sym}_0^2 \\\n&&&+ "
        else:
            src += f"&& &&{coef1Sym}_0 {coef2Sym}_0 \\\n&&&+ "
    if n > 1:
        src += (
            f'&& " " ( " "'
            f'&&{coef1Sym}_0 {coef2Sym}_1 &&+ &&{coef1Sym}_1 {coef2Sym}_0 && ) " " '
            f"&&{termSym} \\\n&&&+ "
        )
    for i in range(2, n + 1):
        src += (
            f'&& " " ( " " &&'
            + " &&+ &&".join(_coefSrcGenerator(i, coef1Sym, coef2Sym, useSqr))
            + (' && " " ) " " ' if i == n else ' && ) " " ')
            + f"&&{termSym}^{i} \\\n&&&+ "
        )
    src += "&& &&......"
    i_text = TypstMath(src, **kwargs)

    # 子对象切分
    prefixLen = len(TypstMath(prefix))
    i_text.i_prefix = i_text[:prefixLen]
    i_terms = i_text.i_terms = Group(Group())
    i_coefs = i_text.i_coefs = Group()
    i_others = i_text.i_others = Group(Group())

    start = end = prefixLen

    def _addCoefs(i, j):
        nonlocal start, end
        l1, l2 = _psProdCoefChCount(i, j)
        end += l1 + l2
        i_coefsLine.add(i_text[start:end])
        i_text[start].color.set(GREEN)
        i_text[start + l1].color.set(BLUE)
        start = end

    i_coefsLine = Group()
    if n > 0:
        # 常数项系数
        _addCoefs(0, 0)
        i_coefs.add(i_coefsLine)
    for i in range(1, n + 1):
        i_coefsLine = Group()
        i_othersLine = Group()
        # 加号、左括号
        end += 2
        i_othersLine.add(i_text[start:end])
        start = end
        for j in range(i):
            # 系数
            _addCoefs(i, j)
            # 加号
            end += 1
            i_othersLine.add(i_text[start:end])
            start = end
        # 最后一个系数
        _addCoefs(i, i)
        # 右括号
        end += 1
        i_othersLine.add(i_text[start:end])
        start = end
        # 指数 x^i
        end += _psProdTermChCount(i)
        i_terms.add(i_text[start:end])
        start = end
        i_coefs.add(i_coefsLine)
        i_others.add(i_othersLine)
    # 最后一行的加号和省略号
    i_others.add(i_text[end:])
    i_text.i_noPrefix = i_text[prefixLen:]

    return i_text


def createPowerSeriesRec(
    n: int,
    coefSym: str = "a",
    termSym: str = "x",
    startDeg: int = 0,
    idxOffset: int = 0,
):
    # 生成公式对象
    src = ""
    if n > 0 and startDeg <= 0:
        src += f"&{coefSym}_{idxOffset} &&&arrow.l\\\n"
    if n > 1 and startDeg <= 1:
        src += f"&{coefSym}_{1 + idxOffset} &&{termSym} &arrow.l\\\n"
    for i in range(max(2, startDeg), n + 1):
        src += f"&{coefSym}_{i + idxOffset} &&{termSym}^{i} &arrow.l\\\n"
    i_text = TypstMath(src)

    # 子对象切分
    i_lines = i_text.i_lines = Group()
    i_coefs = i_text.i_coefs = Group()
    i_terms = i_text.i_terms = Group()
    i_arrows = i_text.i_arrows = Group()
    start = end = lineStart = 0
    if n > 0 and startDeg <= 0:
        # 系数
        end += len(str(idxOffset)) + 1
        i_text[start].color.set(ORANGE)
        i_coefs.add(i_text[start:end])
        start = end
        # x 的幂次 (常数项没有，故添加空组)
        i_terms.add(Group())
        # 箭头
        end += 1
        i_arrows.add(i_text[start:end])
        start = end
        # 整行
        i_lines.add(i_text[lineStart:end])
        lineStart = end

    if n > 1 and startDeg <= 1:
        # 系数
        end += len(str(idxOffset + 1)) + 1
        i_text[start].color.set(ORANGE)
        i_coefs.add(i_text[start:end])
        start = end
        # x 的幂次 (一次项没有指数)
        end += 1
        i_terms.add(i_text[start:end])
        start = end
        # 箭头
        end += 1
        i_arrows.add(i_text[start:end])
        start = end
        # 整行
        i_lines.add(i_text[lineStart:end])
        lineStart = end

    for i in range(max(2, startDeg), n + 1):
        # 系数
        end += len(str(idxOffset + i)) + 1
        i_text[start].color.set(ORANGE)
        i_coefs.add(i_text[start:end])
        start = end
        # x的幂次
        end += len(str(i)) + 1
        i_terms.add(i_text[start:end])
        start = end
        # 箭头
        end += 1
        i_arrows.add(i_text[start:end])
        start = end
        # 整行
        i_lines.add(i_text[lineStart:end])
        lineStart = end

    return i_text


class TL_Catalan(Timeline):
    CONFIG = Config(
        font=["FandolSong"],
    )

    def construct(self):
        catalanTable = generateCatalanTable()
        fw = Config.get.frame_width
        fh = Config.get.frame_height
        rectw = fw * 2
        recth = fh * 2
        n = 10
        u = 0.5
        coordOrigin = np.array((-4, -2, 0))
        signPos = np.array((4, -0.5, 0))

        def animatePath(
            vsteps: Collection[int], n=n, hcolor=GREEN, vcolor=BLUE, markColor=ORANGE
        ):
            i_path = Group()
            i_startingPath = Group()
            i_badSegs = Group()
            i_flippedPath = Group()

            nv = len(vsteps)
            nh = 2 * n - nv
            x0_scr, y0_scr, _ = i_grids.c2p(nh, 0)
            x = y = 0
            ag_transformHPath, ag_transformVPath = [], []
            ag_changeColor, ag_restoreColor = [], []
            fn_changeColor, fn_restoreColor = [], []
            ag_changeFlippedColor, ag_restoreFlippedColor = [], []
            fn_changeFlippedColor, fn_restoreFlippedColor = [], []
            ag_flip, ag_revert = [], []

            p0 = n

            def _helper(i):
                nonlocal x, y, p0
                # i_dot = Dot(i_grids.c2p(x, y), radius=0.04)
                # i_dot.depth.set(-1)
                # i_vg.add(i_dot)
                isVertical = i in vsteps
                if isVertical:  # 纵向移动
                    x1, y1, x2, y2 = x, y, x, y + 1
                    clr1, clr2 = vcolor, hcolor
                else:  # 横向移动
                    x1, y1, x2, y2 = x, y, x + 1, y
                    clr1, clr2 = hcolor, vcolor

                i_line = Line(i_grids.c2p(x1, y1), i_grids.c2p(x2, y2))
                i_line.stroke.set(color=clr1)
                i_line, i_line2 = i_line.copy(), i_line
                ag_restoreColor.append(lambda: i_line.anim.stroke.set(color=clr1))
                fn_restoreColor.append(lambda: i_line.stroke.set(color=clr1))

                if isVertical:
                    ag_transformVPath.append(lambda: Transform(i_line2, i_line))
                    i_line2.points.set_x(x0_scr)
                    y += 1
                    if y - 1 < x:
                        i_badSegs.add(i_line)
                else:
                    ag_transformHPath.append(lambda: Transform(i_line2, i_line))
                    i_line2.points.set_y(y0_scr)
                    x += 1
                    if y < x:
                        i_badSegs.add(i_line)
                    if y == x < p0:  # 确定第一个越界点的位置
                        p0 = y
                if i > 2 * p0:
                    ag_changeColor.append(
                        lambda: i_line.anim.stroke.set(color=markColor)
                    )
                    fn_changeColor.append(lambda: i_line.stroke.set(color=markColor))
                    i_flippedLine = Line(
                        i_grids.c2p(y1 + 1, x1 - 1), i_grids.c2p(y2 + 1, x2 - 1)
                    )
                    i_flippedLine.stroke.set(color=markColor)
                    ag_changeFlippedColor.append(
                        lambda: i_flippedLine.anim.stroke.set(color=markColor)
                    )
                    ag_restoreFlippedColor.append(
                        lambda: i_flippedLine.anim.stroke.set(color=clr2)
                    )
                    fn_changeFlippedColor.append(
                        lambda: i_flippedLine.stroke.set(color=markColor)
                    )
                    fn_restoreFlippedColor.append(
                        lambda: i_flippedLine.stroke.set(color=clr2)
                    )
                    ag_flip.append(lambda: Transform(i_line, i_flippedLine))
                    ag_revert.append(lambda: Transform(i_flippedLine, i_line))
                    i_flippedPath.add(i_flippedLine)
                else:
                    i_flippedPath.add(i_line)
                # i_line.add_tip(colorize=True, d_alpha=1)

                i_path.add(i_line)
                i_startingPath.add(i_line2)

            for i in range(2 * n):
                _helper(i)
            # i_dot = Dot(i_grids.c2p(x, y), radius=0.04)
            # i_vg.add(i_dot)

            class _retType:
                @property
                def ag_transformHPath(self):
                    return AnimGroup(*map(op.call, ag_transformHPath))

                @property
                def ag_transformVPath(self):
                    return AnimGroup(*map(op.call, ag_transformVPath))

                @property
                def ag_changeColor(self):
                    return AnimGroup(*map(op.call, ag_changeColor))

                @property
                def ag_restoreColor(self):
                    return AnimGroup(*map(op.call, ag_restoreColor))

                @property
                def ag_changeFlippedColor(self):
                    return AnimGroup(*map(op.call, ag_changeFlippedColor))

                @property
                def ag_restoreFlippedColor(self):
                    return AnimGroup(*map(op.call, ag_restoreFlippedColor))

                def changeColor(self):
                    for fn in fn_changeColor:
                        fn()

                def restoreColor(self):
                    for fn in fn_restoreColor:
                        fn()

                def changeFlipedColor(self):
                    for fn in fn_changeFlippedColor:
                        fn()

                def restoreFlippedColor(self):
                    for fn in fn_restoreFlippedColor:
                        fn()

                @property
                def ag_flip(self):
                    return AnimGroup(*map(op.call, ag_flip))

                @property
                def ag_revert(self):
                    return AnimGroup(*map(op.call, ag_revert))

            ret = _retType()
            ret.i_startingPath = i_startingPath
            ret.i_path = i_path
            ret.i_badSegs = i_badSegs
            ret.i_flippedPath = i_flippedPath
            ret.p0 = p0
            ret.i_pathToFlip = i_path[p0 * 2 + 1 :]

            return ret

        # 网格
        i_grids = NumberPlane(
            x_range=(-7, 24, 1),
            y_range=(-4, 14, 1),
            unit_size=u,
            faded_line_ratio=0,
            background_line_style={"stroke_alpha": 0.5},
        )
        i_grids.points.shift(coordOrigin)
        i_grids.depth.set(2)

        # 黑色背景
        i_blackRect = Rect(rectw, recth)
        i_blackRect.fill.set(color=BLACK, alpha=0.75)
        i_blackRect.stroke.set(alpha=0)
        i_blackRect.points.set_y(coordOrigin[1] - fh).rotate(
            PI / 4, about_point=coordOrigin
        )
        i_blackRect.depth.set(1)

        # 斜线 y = x, y = x - 1, y = x + 1
        i_dividingLine = Line(
            (-fw / 2, 0, 0),
            (fw / 2, 0, 0),
        )
        i_dividingLine.radius.set(0.015)
        i_dividingLine.stroke.set(alpha=0.5)

        i_dividingLine.points.set_y(coordOrigin[1]).rotate(
            PI / 4, about_point=coordOrigin
        )

        i_upperLine = i_dividingLine.copy()
        i_lowerLine = i_dividingLine.copy()
        for i_ in (i_upperLine, i_lowerLine):
            i_.stroke.set(color=ORANGE)
        i_upperLine.points.shift(UP * u)
        i_lowerLine.points.shift(DOWN * u)

        # 禁止进入标记
        i_sign = SVGItem(DIR / "assets/image/abandoned.svg")
        i_sign.points.scale(0.6).shift(signPos)

        # 起点、终点旗标
        i_startFlag = createFlag(color=RED, rotation=-PI / 12)
        i_stopFlag = createFlag(color=GREEN, rotation=PI / 12)
        i_startFlag.points.shift(i_grids.c2p(0, 0))
        i_stopFlag.points.shift(i_grids.c2p(n, n))

        i_startCoordText = TypstMath("(0, 0)")
        i_stopCoordText = TypstMath("(n, n)")
        i_flippedCoordText = TypstMath("(n + 1, n - 1)")
        i_startCoordText.points.next_to(i_grids.c2p(0, 0), DOWN)
        i_stopCoordText.points.next_to(i_grids.c2p(n, n), RIGHT)
        i_flippedCoordText.points.next_to(i_grids.c2p(n + 1, n - 1), RIGHT)

        self.play(Create(i_grids))
        self.play(FadeIn(i_blackRect))
        self.play(Create(i_dividingLine))
        self.play(FadeIn(i_sign, scale=0.8))
        self.play(
            FadeIn(i_startFlag, shift=DOWN * 0.2, scale=0.5, rate_func=rush_into),
            FadeIn(i_stopFlag, shift=DOWN * 0.2, scale=0.5, rate_func=rush_into),
            Write(i_startCoordText),
            Write(i_stopCoordText),
            duration=0.5,
        )
        self.forward(0.25)
        self.play(
            Flash(i_grids.c2p(0, 0)),
            Flash(i_grids.c2p(n, n)),
            duration=1,
        )
        self.forward(0.5)

        i_texts = Group()
        maxTextWidth = 0.4

        for d in range(2 * n + 1):
            i_diagonalGroup = Group()
            for i in range(d // 2 + 1) if d <= n else range(d - n, d // 2 + 1):
                j = d - i
                i_text = boolean_ops.Union.from_group(Text(f"{catalanTable[i,j]}"))
                i_text.depth.set(-2)
                i_text.points.shift(i_grids.c2p(i, j))
                i_text.set_stroke_background(True)
                i_text.color.set(YELLOW, alpha=1)
                i_text.stroke.set(color=BLACK, alpha=0.75)
                i_text.radius.set(0.05)
                w = i_text.points.box.width
                if w > maxTextWidth:
                    i_text.points.set_width(maxTextWidth)
                i_diagonalGroup.add(i_text)
            i_texts.add(i_diagonalGroup)

        ag = []
        i_lastGroup = None
        for i, i_diagonalGroup in enumerate(i_texts):
            dur = 0.6 - 0.5 * (rush_from(i / n / 2))
            if i_lastGroup is None:
                ag.append(FadeIn(i_diagonalGroup, duration=dur))
            else:
                ag1 = []
                if i <= n:
                    if i & 1 == 1:  # i odd
                        for j, i_text in enumerate(i_lastGroup[:-1]):
                            ag1.extend(
                                (
                                    Transform(
                                        i_text,
                                        i_diagonalGroup[j],
                                        hide_src=False,
                                    ),
                                    Transform(
                                        i_text,
                                        i_diagonalGroup[j + 1],
                                        hide_src=False,
                                    ),
                                )
                            )
                        ag1.append(
                            Transform(
                                i_lastGroup[-1],
                                i_diagonalGroup[-1],
                                hide_src=False,
                            )
                        )
                    else:  # i even
                        for j, i_text in enumerate(i_lastGroup):
                            ag1.extend(
                                (
                                    Transform(
                                        i_text,
                                        i_diagonalGroup[j],
                                        hide_src=False,
                                    ),
                                    Transform(
                                        i_text,
                                        i_diagonalGroup[j + 1],
                                        hide_src=False,
                                    ),
                                )
                            )
                else:
                    if i & 1 == 1:  # i odd
                        for j, i_text in enumerate(i_diagonalGroup):
                            ag1.extend(
                                (
                                    Transform(
                                        i_lastGroup[j],
                                        i_text,
                                        hide_src=False,
                                    ),
                                    Transform(
                                        i_lastGroup[j + 1],
                                        i_text,
                                        hide_src=False,
                                    ),
                                )
                            )
                    else:  # i even
                        for j, i_text in enumerate(i_diagonalGroup[:-1]):
                            ag1.extend(
                                (
                                    Transform(
                                        i_lastGroup[j],
                                        i_text,
                                        hide_src=False,
                                    ),
                                    Transform(
                                        i_lastGroup[j + 1],
                                        i_text,
                                        hide_src=False,
                                    ),
                                )
                            )
                        ag1.append(
                            Transform(
                                i_lastGroup[-1],
                                i_diagonalGroup[-1],
                                hide_src=False,
                            )
                        )

                ag1 = AnimGroup(*ag1, duration=dur)
                ag.append(ag1)
            i_lastGroup = i_diagonalGroup
        ag = Succession(*ag)

        self.play(FadeOut(Group(i_startCoordText, i_stopCoordText)), duration=0.5)
        self.play(ag)
        self.forward(0.5)
        self.play(FadeOut(i_texts))
        self.forward(0.5)
        self.play(FadeOut(i_dividingLine), FadeOut(i_blackRect), FadeOut(i_sign))

        samplePath1 = frozenset((0, 1, 2, 8, 9, 10, 15, 16, 17, 18))
        samplePath2 = frozenset((0, 1, 2, 7, 8, 9, 15, 16, 17, 18))
        pathAc1 = animatePath(samplePath1)
        pathAc2 = animatePath(samplePath2)
        self.play(FadeIn(pathAc1.i_startingPath))
        self.play(pathAc1.ag_transformHPath)
        self.play(pathAc1.ag_transformVPath)

        i_binom1 = TypstMath("binom(2n, n)")
        i_binom2 = TypstMath("binom(2n, n-1)")
        i_catalanFormula = TypstMath("a_(n) = sum_(k = 1)^(n) a_(k - 1) a_(n - k)")
        i_binom1.points.next_to(i_grids.c2p(n, n), RIGHT)
        i_binom2.points.next_to(i_grids.c2p(n + 1, n - 1), RIGHT)
        i_catalanFormula.points.shift((4, 1, 0)).scale(1.5)
        for i_ in (i_binom1, i_binom2):
            (
                i_(VItem)
                .set_stroke_background(True)
                .radius.set(0.05)
                .r.stroke.set(color=BLACK)
            )

        self.play(FadeIn(i_binom1))
        self.play(i_binom1.anim.points.move_to((4, 0.5, 0)).scale(1.5))
        self.play(Create(i_dividingLine), FadeIn(i_blackRect))

        def blink(i_s: Iterable[Item], dt=0.15, n=4):
            for _ in range(n):
                for i_ in i_s:
                    i_.stroke.set(alpha=0.5)
                self.forward(dt)
                for i_ in i_s:
                    i_.stroke.set(alpha=1)
                self.forward(dt)

        blink(pathAc1.i_badSegs)
        # self.play(*(i_.anim.stroke.set(alpha=0.5) for i_ in animPathRet.i_badSegs))
        self.play(
            AnimGroup(
                *(Indicate(i_, scale_factor=1, duration=0.25) for i_ in pathAc1.i_path),
                lag_ratio=0.25,
            )
        )
        i_focusBox = createFocusBox()
        i_focusBox.points.shift(i_grids.c2p(*it.repeat(pathAc1.p0, 2)))
        self.play(FadeIn(i_focusBox, scale=0.5), FadeOut(i_blackRect))
        self.play(
            i_focusBox.anim.points.move_to(i_grids.c2p(pathAc1.p0 + 1, pathAc1.p0))
        )
        self.play(
            Rotate(i_focusBox, PI / 2, duration=1),
            Succession(
                i_focusBox.anim.points.scale(2),
                i_focusBox.anim.points.scale(0.5),
                duration=1,
            ),
        )
        self.play(pathAc1.ag_changeColor, FadeIn(i_stopCoordText), duration=0.5)
        self.play(Create(i_lowerLine))
        vec = i_grids.c2p(1, -1) - coordOrigin
        turning_rf = ease_inout_sine
        self.play(
            pathAc1.ag_flip,
            i_stopFlag.anim.points.shift(vec),
            Transform(i_stopCoordText, i_flippedCoordText),
            rate_func=turning_rf,
            duration=4,
        )
        self.play(Flash(i_grids.c2p(n + 1, n - 1)))
        self.forward(1)
        self.play(
            pathAc1.ag_revert,
            i_stopFlag.anim.points.shift(-vec),
            Transform(i_flippedCoordText, i_stopCoordText),
            rate_func=turning_rf,
            duration=2,
        )
        self.forward(0.5)
        self.play(
            pathAc1.ag_flip,
            i_stopFlag.anim.points.shift(vec),
            Transform(i_stopCoordText, i_flippedCoordText),
            rate_func=turning_rf,
            duration=2,
        )
        self.play(FadeOut(pathAc1.i_flippedPath), FadeOut(i_focusBox), duration=0.5)

        pathAc2.restoreFlippedColor()
        i_focusBox.points.move_to(i_grids.c2p(pathAc2.p0 + 1, pathAc2.p0))

        self.play(Succession(*map(Create, pathAc2.i_flippedPath)), duration=1)
        self.play(FadeIn(i_focusBox, scale=0.5))
        self.play(
            Rotate(i_focusBox, PI / 2, duration=1),
            Succession(
                i_focusBox.anim.points.scale(2),
                i_focusBox.anim.points.scale(0.5),
                duration=1,
            ),
        )

        self.play(pathAc2.ag_changeFlippedColor, duration=0.5)
        pathAc2.changeColor()
        self.play(
            pathAc2.ag_revert,
            i_stopFlag.anim.points.shift(-vec),
            Transform(i_flippedCoordText, i_stopCoordText),
            rate_func=turning_rf,
            duration=2,
        )
        self.play(FadeIn(i_blackRect))
        blink(pathAc2.i_badSegs)
        self.play(
            pathAc2.ag_flip,
            i_stopFlag.anim.points.shift(vec),
            FadeOut(i_stopCoordText),
            FadeOut(i_blackRect),
            rate_func=turning_rf,
            duration=2,
        )
        self.play(FadeIn(i_binom2))
        self.play(
            pathAc2.ag_revert,
            i_stopFlag.anim.points.shift(-vec),
            FadeIn(i_blackRect),
            i_binom2.anim.points.next_to(i_grids.c2p(n, n), RIGHT),
            rate_func=turning_rf,
            duration=2,
        )

        i_binom3 = TypstMath("binom(2n, n) - binom(2n, n - 1) = 1/(n + 1) binom(2n, n)")
        i_binom3.points.move_to((4, 0, 0))
        self.play(
            Transform(i_binom1, i_binom3[:5]),
            FadeIn(i_binom3[5]),
            Transform(i_binom2, i_binom3[6:13]),
            FadeIn(i_binom3[13:]),
        )
        self.forward(2)
        self.play(FadeOut(Group(i_binom3, pathAc2.i_path, i_lowerLine, i_focusBox)))

        i_anTexts = Group()
        for i in range(n + 1):
            i_anText = TypstMath(f"a_{i}")
            i_anText.points.shift(i_grids.c2p(i, i))
            i_anText.depth.set(-2)
            i_anTexts.add(i_anText)

        self.play(AnimGroup(*map(Write, i_anTexts), lag_ratio=0.25), duration=3)

        showingPow = 4
        i_psText = createPowerSeries(showingPow, align=False)
        i_psText.points.move_to((3, -1, 0))

        ag = []
        for i_anText, i_coefText in zip(
            i_anTexts,
            it.chain(i_psText.i_coefs, it.repeat(i_psText.i_ellipsis)),
        ):
            ag.append(Transform(i_anText, i_coefText, hide_src=False))

        self.play(*ag, FadeIn(Group(i_psText.i_others, i_psText.i_terms)))
        self.play(ShowPassingFlashAround(i_psText, time_width=3))
        self.play(FadeOut(i_anTexts))
        self.forward(1)
        self.play(Write(i_catalanFormula), duration=1)
        self.play(ShowPassingFlashAround(i_catalanFormula, time_width=3))
        self.forward(2)
        self.play(FadeOut(i_catalanFormula))

        samplePaths = (
            frozenset((0, 1, 2, 6, 7, 10, 12, 13, 14, 18)),
            frozenset((0, 1, 3, 5, 6, 7, 8, 9, 10, 18)),
            frozenset(range(10)),
            frozenset((0, 1, 2, 5, 6, 7, 8, 14, 15, 18)),
        )
        samplePathAcs = tuple(map(animatePath, samplePaths))

        lastPathAc = None
        for pathAc in samplePathAcs:
            if lastPathAc is None:
                i_focusBox.points.move_to(i_grids.c2p(pathAc.p0, pathAc.p0))
                self.play(
                    FadeIn(pathAc.i_path, duration=0.25),
                    FadeIn(i_focusBox, scale=0.5, duration=0.5),
                )
            else:
                self.play(
                    Transform(lastPathAc.i_path, pathAc.i_path, duration=0.25),
                    i_focusBox.anim(duration=0.5).points.move_to(
                        i_grids.c2p(pathAc.p0, pathAc.p0)
                    ),
                )
            self.forward(0.75)
            lastPathAc = pathAc

        k = pathAc.p0
        i_p0Text = TypstMath("(k, k)")
        i_p0Text.points.next_to(i_grids.c2p(k, k), DR)
        self.play(FadeIn(Group(i_p0Text, i_stopCoordText)))

        i_upperTriangle = Polygon(
            *map(i_grids.c2p, *np.array(((k, k), (k, n), (n, n))).T)
        )
        i_lowerTriangle1 = Polygon(
            *map(i_grids.c2p, *np.array(((0, 0), (0, k), (k, k))).T)
        )
        i_lowerTriangle2 = Polygon(
            *map(i_grids.c2p, *np.array(((0, 1), (0, k), (k - 1, k))).T)
        )
        i_upperText = TypstMath("a_(n - k)")
        i_lowerText = TypstMath("a_(k - 1)")
        i_upperText.points.move_to(i_grids.c2p((n + 2 * k) / 3, (2 * n + k) / 3))
        i_lowerText.points.move_to(i_grids.c2p((k - 1) / 3, (1 + 2 * k) / 3))
        for i_ in (i_upperText, i_lowerText):
            (
                i_(VItem)
                .set_stroke_background(True)
                .stroke.set(color=BLACK, alpha=0.75)
                .r.radius.set(0.05)
                .r.depth.set(-3)
            )

        if n - k > 3:
            i_upperText.points.scale(1.5)
        if k - 1 > 3:
            i_lowerText.points.scale(1.5)

        i_crossTemplate = SVGItem(DIR / "assets/image/cross.svg", width=0.25)
        for i_ in (i_upperTriangle, i_lowerTriangle1, i_lowerTriangle2):
            i_.fill.set(color=YELLOW, alpha=0.25)
            i_.radius.set(0)
            i_.depth.set(3)

        i_crosses = Group()
        for i in range(1, k):
            i_cross = i_crossTemplate.copy()
            i_cross.points.move_to(i_grids.c2p(i, i))
            i_crosses.add(i_cross)

        self.play(FadeIn(i_upperTriangle))
        self.play(FadeIn(i_upperText))
        self.forward(1)
        self.play(FadeIn(i_lowerTriangle1))

        for _ in range(4):
            self.show(i_crosses)
            self.forward(0.25)
            self.hide(i_crosses)
            self.forward(0.25)

        self.forward(1)

        self.hide(i_dividingLine, i_blackRect)

        # 将分界线分成3段
        i_dividingLineUpper = Line(i_grids.c2p(k, k), i_grids.c2p(14, 14))
        i_dividingLineMiddle = Line(i_grids.c2p(0, 0), i_grids.c2p(k, k))
        i_dividingLineMiddleTr = Line(i_grids.c2p(0, 1), i_grids.c2p(k - 1, k))
        i_dividingLineLower = Line(i_grids.c2p(-8, -8), i_grids.c2p(0, 0))

        # 将黑色矩形改成多边形
        i_blackPolygon = Polygon(
            *map(
                i_grids.c2p,
                *np.array(
                    (
                        (-8, -8),
                        (0, 0),
                        (0, 0),
                        (k, k),
                        (k, k),
                        (14, 14),
                        (24, 14),
                        (24, -8),
                    )
                ).T,
            )
        )
        i_blackPolygonTr = Polygon(
            *map(
                i_grids.c2p,
                *np.array(
                    (
                        (-8, -8),
                        (0, 0),
                        (0, 1),
                        (k - 1, k),
                        (k, k),
                        (14, 14),
                        (24, 14),
                        (24, -8),
                    )
                ).T,
            )
        )
        for i_ in (i_blackPolygon, i_blackPolygonTr):
            i_.fill.set(color=BLACK, alpha=0.75)
            i_.radius.set(0)
            i_.depth.set(1)
        for i_ in (
            i_dividingLineUpper,
            i_dividingLineMiddle,
            i_dividingLineMiddleTr,
            i_dividingLineLower,
        ):
            i_.stroke.set(alpha=0.5)
            i_.radius.set(0.015)
        self.show(
            i_dividingLineUpper,
            i_dividingLineMiddle,
            i_dividingLineLower,
            i_blackPolygon,
        )
        vec = i_grids.c2p(-1, 0) - coordOrigin
        self.play(
            pathAc.i_path[0].anim(duration=0.5).color.set(YELLOW),
            Flash(pathAc.i_path[0], duration=1),
            pathAc.i_path[2 * k - 1].anim(duration=0.5).color.set(YELLOW),
            Flash(pathAc.i_path[2 * k - 1], duration=1),
        )
        self.play(
            Transform(i_lowerTriangle1, i_lowerTriangle2),
            Transform(i_dividingLineMiddle, i_dividingLineMiddleTr),
            Transform(i_blackPolygon, i_blackPolygonTr),
        )
        self.play(FadeIn(i_lowerText))

        i_upperTextCp = i_upperText.copy()
        i_lowerTextCp = i_lowerText.copy()
        for i_ in (i_upperTextCp, i_lowerTextCp):
            i_.radius.set(0)
            i_.depth.set(-4)

        self.play(
            Transform(i_upperTextCp, i_catalanFormula[-4:]),
            Transform(i_lowerTextCp, i_catalanFormula[-8:-4]),
        )
        self.play(FadeIn(i_catalanFormula[:8]))
        self.forward(2)

        i_catalanFormula2 = TypstMath("a_(n) = sum_(k = 1)^(n) a_(k - 1) a_(n - k)")
        i_catalanFormula2.points.move_to((4, 1, 0))

        showingPow2 = 6
        i_psText1 = createPowerSeries(showingPow2)
        i_psText2 = createPowerSeries(showingPow2)
        for i_ in i_psText1.i_coefs:
            i_[0].color.set(GREEN)
        for i_ in i_psText2.i_coefs:
            i_[0].color.set(BLUE)
        i_psText1.points.to_border(UP, 0.75)
        i_psText2.points.next_to(i_psText1, DOWN, buff=0.3)

        i_psProdText = createPowerSeriesProduct(
            showingPow2,
            prefix="&&f^2(x) &=",
            coef2Sym="a",
            useSqr=False,
        )

        self.play(
            TransformMatchingShapes(i_psText, i_psText1),
            Transform(i_catalanFormula, i_catalanFormula2),
            FadeOut(
                Group(
                    i_grids,
                    i_upperTriangle,
                    i_lowerTriangle2,
                    i_focusBox,
                    i_startFlag,
                    i_stopFlag,
                    i_dividingLineUpper,
                    i_dividingLineMiddleTr,
                    i_dividingLineLower,
                    i_blackPolygonTr,
                    i_upperText,
                    i_lowerText,
                    i_p0Text,
                    i_stopCoordText,
                    pathAc.i_path,
                )
            ),
        )
        # self.play(TransformMatchingShapes(i_psText, i_psText1), duration=0.5)
        self.play(Transform(i_psText1, i_psText2, hide_src=False), duration=0.5)

        i_psProdText.points.next_to(i_psText2, DOWN, buff=0.5)
        # self.show(i_psProdText)
        self.play(FadeIn(i_psProdText.i_prefix))

        i_boxes1 = createPsBoxes(i_psText1, color=GREEN, buff=0.05)
        i_boxes2 = createPsBoxes(i_psText2, color=BLUE, buff=0.05)

        i_showingBox1 = None
        i_showingBox2 = None
        for i in range(showingPow2 + 1):
            for j in range(i + 1):
                i_box1 = i_boxes1[j]
                i_box2 = i_boxes2[i - j]
                if i_showingBox1 is None:
                    self.play(FadeIn(Group(i_box1, i_box2)), duration=0.2)
                else:
                    self.play(
                        Transform(i_showingBox1, i_box1),
                        Transform(i_showingBox2, i_box2),
                        duration=0.2,
                        rate_func=rush_from,
                    )
                self.play(FadeIn(i_psProdText.i_coefs[i][j]), duration=0.2)
                i_showingBox1 = i_box1
                i_showingBox2 = i_box2
            self.play(
                FadeOut(Group(i_showingBox1, i_showingBox2)),
                FadeIn(i_psProdText.i_others[i]),
                FadeIn(i_psProdText.i_terms[i]),
                duration=0.2,
            )
            i_showingBox1 = i_showingBox2 = None
        self.play(FadeIn(i_psProdText.i_others[-1]), duration=0.25)
        self.play(
            i_psProdText.i_noPrefix.anim.points.shift(RIGHT * 0.5),
            FadeOut(i_psProdText.i_prefix),
            duration=0.5,
        )

        i_recursiveTerms = createPowerSeriesRec(showingPow2, idxOffset=1)
        i_recursiveTerms.points.move_to(i_psProdText, aligned_edge=UL).shift(
            (-0.5, -0.2, 0)
        )
        ag1, ag2 = [], []
        for i in range(showingPow2 + 1):
            i_coefsLine = i_psProdText.i_coefs[i]
            i_recLine = i_recursiveTerms.i_lines[i]
            ag1.append(ShowPassingFlashAround(i_coefsLine, time_width=3, duration=1))
            ag2.append(
                Transform(
                    i_coefsLine,
                    Group(i_recLine),
                    hide_src=False,
                    duration=0.75,
                )
            )

        self.play(ag1[-1])
        self.play(Indicate(i_catalanFormula2))
        self.play(ag2[-1])
        self.play(AnimGroup(*reversed(ag2[:-1]), lag_ratio=0.5))
        self.forward(2)

        i_dividingLine2 = Line(ORIGIN, (12, 0, 0))
        i_psText3 = createPowerSeries(showingPow2, prefix="f^2(x) =", idxOffset=1)
        i_psText3Tr = createPowerSeries(showingPow2 - 1, prefix="f^2(x) =", idxOffset=1)
        i_psText4 = createPowerSeries(showingPow2 - 1, prefix="x f^2(x) =", startPow=1)
        i_psText5 = createPowerSeries(showingPow2, prefix="x f^2(x) + 1 =")
        i_dividingLine2.points.next_to(i_psText2, DOWN, buff=0.3)
        i_dividingLine2.radius.set(0.015)
        for i_ in i_psText3, i_psText3Tr:
            i_.points.next_to(i_dividingLine2, DOWN, buff=0.3, coor_mask=UP)
        i_psText4.points.next_to(i_psText3, DOWN, buff=0.3, coor_mask=UP)
        for i_ in i_psText3Tr, i_psText4:
            i_.i_gridItems[1:].points.shift(RIGHT * i_.alignParams.totalStep)
        i_psText5.points.next_to(i_psText4, DOWN, buff=0.3, coor_mask=UP)
        i_boxes3 = createPsBoxes(i_psText3, color=ORANGE, buff=0.05)

        for i_ps in i_psText3, i_psText3Tr, i_psText4:
            for i_coef in i_ps.i_coefs:
                i_coef[0].color.set(ORANGE)

        for i_coef in i_psText5.i_coefs[1:]:
            i_coef[0].color.set(ORANGE)

        ag = []

        for i in range(showingPow2 + 1):
            ag.append(Transform(i_recursiveTerms.i_coefs[i], i_psText3.i_coefs[i]))
            ag.append(Transform(i_recursiveTerms.i_terms[i], i_psText3.i_terms[i]))

        self.play(
            *ag,
            FadeIn(i_psText3.i_others),
            FadeIn(i_psText3.i_ellipsis),
            FadeOut(i_psProdText.i_noPrefix),
            FadeOut(i_recursiveTerms.i_arrows),
            FadeOut(i_catalanFormula2),
            Create(i_dividingLine2),
        )
        self.forward(1)

        i_showingBox2 = i_showingBox3 = None
        for i in range(showingPow2):
            i_box2 = i_boxes2[i + 1]
            i_box3 = i_boxes3[i]
            if i_showingBox2 is None:
                self.play(FadeIn(Group(i_box2, i_box3)), duration=0.4)
            else:
                self.play(
                    Transform(i_showingBox2, i_box2),
                    Transform(i_showingBox3, i_box3),
                    duration=0.4,
                    rate_func=rush_from,
                )
            i_showingBox2 = i_box2
            i_showingBox3 = i_box3
            self.forward(0.2)
        self.play(FadeOut(Group(i_showingBox2, i_showingBox3)), duration=0.4)

        self.forward(0.5)
        self.play(
            TransformMatchingShapes(
                i_psText3,
                i_psText3Tr,
                mismatch=(FadeOut, FadeIn),
                collapse=False,
            ),
            duration=1,
        )

        def animatePsTransform(i_ps1, i_ps2, offset=0, matchOthers=False):
            l1, l2 = len(i_ps1.i_coefs), len(i_ps2.i_coefs)
            l = min(l1, l2)
            ag = []
            i_coefs1 = i_ps1.i_coefs
            i_coefs2 = i_ps2.i_coefs
            i_terms1 = i_ps1.i_terms
            i_terms2 = i_ps2.i_terms
            if offset > 0:
                ag.extend(
                    FadeIn(Group(i_1, i_2))
                    for i_1, i_2 in zip(
                        it.chain(
                            i_coefs1[l:],
                            i_coefs2[:offset],
                            i_coefs2[l + offset :],
                        ),
                        it.chain(
                            i_terms1[l:],
                            i_terms2[:offset],
                            i_terms2[l + offset :],
                        ),
                    )
                )
            for i in range(l - offset) if offset > 0 else range(-offset, l):
                i_coef1 = i_ps1.i_coefs[i]
                i_coef2 = i_ps2.i_coefs[i + offset]
                i_term1 = i_ps1.i_terms[i]
                i_term2 = i_ps2.i_terms[i + offset]
                ag.append(Transform(i_coef1, i_coef2, hide_src=False))
                if len(i_term1) == 0:
                    ag.append(FadeIn(i_term2))
                elif len(i_term2) == 0:
                    ag.append(FadeOut(i_term1))
                else:
                    ag.append(Transform(i_term1, i_term2, hide_src=False))
            self.play(
                (
                    TransformMatchingShapes(i_ps1.i_others.copy(), i_ps2.i_others)
                    if matchOthers
                    else Transform(i_ps1.i_others, i_ps2.i_others, hide_src=False)
                ),
                Transform(i_ps1.i_ellipsis, i_ps2.i_ellipsis, hide_src=False),
                *ag,
                duration=1,
            )

        self.forward(1)
        self.play(
            Transform(
                i_psText3Tr.i_gridItems[0],
                i_psText4.i_gridItems[0],
                hide_src=False,
            ),
            TransformMatchingShapes(
                i_psText3Tr.i_gridItems[1:].copy(),
                i_psText4.i_gridItems[1:],
                mismatch=(FadeOut, FadeIn),
                duration=1,
            ),
        )
        self.forward(1)
        self.play(
            TransformMatchingShapes(
                i_psText4.i_gridItems[0].copy(),
                i_psText5.i_gridItems[0],
                duration=1,
            ),
            FadeIn(i_psText5.i_gridItems[1:4]),
            TransformMatchingShapes(
                i_psText4.i_gridItems[1:].copy(),
                i_psText5.i_gridItems[4:],
                duration=1,
            ),
        )

        # 生成函数所满足的方程及其表达式
        i_gfEquation = TypstMath("f(x) = x f^2(x) + 1")
        i_gf = TypstMath("f(x) = (1 - sqrt(1 - 4x))/(2x)")
        i_gfProcess = TypstDoc((DIR / "assets/catalan-gf-process.typ").read_text())
        i_gfResult = TypstMath(
            "(1 - sqrt(1 - 4x))/(2x) = sum_(n = 0)^(infinity) 1/(n + 1) binom(2n, n) x^n"
        )
        i_gfEquation.points.move_to((-2.5, -1.75, 0))
        i_gf.points.move_to((2.5, -1.7, 0))
        i_gfProcess.points.shift(DOWN * 1.75)
        i_gfResult.points.next_to(i_gfProcess, DOWN, buff=0.75)

        self.play(Write(i_gfEquation))
        self.play(Transform(i_gfEquation, i_gf, hide_src=False))
        self.forward(2)
        self.play(
            FadeOut(
                Group(
                    i_psText1,
                    i_psText2,
                    i_dividingLine2,
                    i_psText3Tr,
                    i_psText4,
                    i_psText5,
                )
            ),
            Group(i_gfEquation, i_gf).anim.points.shift(UP * 4.5),
        )
        self.play(Write(i_gfProcess), duration=2)
        self.play(Transform(i_gfProcess, i_gfResult, hide_src=False))
        self.play(ShowPassingFlashAround(i_gfResult, time_width=3, duration=1))
        self.forward(2)
        pathAc.restoreColor()
        self.play(
            FadeOut(Group(i_gfEquation, i_gf, i_gfProcess, i_gfResult)),
            FadeIn(
                Group(
                    i_grids,
                    i_startFlag,
                    i_stopFlag,
                    i_blackRect,
                    i_dividingLine,
                    pathAc.i_path,
                )
            ),
        )

        i_catalanTermFormula = TypstMath("a_n = 1/(n + 1) binom(2n, n)")
        i_catalanTable = TypstDoc((DIR / "assets/catalan-table.typ").read_text())
        i_catalanTermFormula.points.scale(1.5).move_to((3.75, 1.5, 0))
        i_catalanTable.points.move_to((2.25, -1, 0))
        self.play(Write(i_catalanTermFormula))
        self.play(Write(i_catalanTable))

        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##########################################################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
##########################################################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
