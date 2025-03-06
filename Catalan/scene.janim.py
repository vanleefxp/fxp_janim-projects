from collections.abc import Collection
from pathlib import Path
import itertools as it
import operator as op

from janim.imports import *
import numpy as np

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


def createFlag(size=0.35, color=RED, rotation=0):
    mob_flagStem = Line((0, 0, 0), (0, size, 0))
    mob_flagFace = Polygon(
        (0, size, 0), (0, size * 0.6, 0), (size * 0.6, size * 0.8, 0)
    )
    mob_flagStem.radius.set(0.015)
    mob_flagStem.stroke.set(color=color)
    mob_flagFace.radius.set(0.015)
    mob_flagFace.stroke.set(color=color)
    mob_flagFace.fill.set(color=color, alpha=1)
    mob_flag = Group(mob_flagStem, mob_flagFace)
    mob_flag.points.rotate(rotation, about_point=ORIGIN)
    mob_flag.depth.set(-1)
    return mob_flag


def createFocusBox(radius=0.16, ratio=0.75):
    mob_focusBox = Group()
    mob_topRight = Polyline(
        (radius, radius * (1 - ratio), 0),
        (radius, radius, 0),
        (radius * (1 - ratio), radius, 0),
    )
    mob_topRight.radius.set(0.01)
    for i in range(4):
        mob_corner = mob_topRight.copy()
        mob_corner.points.rotate(i * PI / 2, about_point=ORIGIN)
        mob_focusBox.add(mob_corner)
    return mob_focusBox


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
    **kwargs,
) -> TypstMath:
    # 创建幂级数公式
    mob_coefGroup = Group()
    mob_termGroup = Group()
    mob_otherGroup = Group()
    src = f"{prefix} "
    if n >= 0 and startPow <= 0:
        src += f"{coefSym}_{idxOffset} + "
    if n >= 1 and startPow <= 1:
        src += f"{coefSym}_{1 + idxOffset} {termSym} + "
    for i in range(max(2, startPow), n + startPow + 1):
        src += f"{coefSym}_{i + idxOffset} {termSym}^{i} + "
    src += "..."
    mob_powerSeriesText = TypstMath(src, **kwargs)
    # 对公式文本进行分组处理
    start = 0
    end = len(TypstMath(prefix))
    # f(x) =
    mob_otherGroup.add(mob_powerSeriesText[:end])
    for i in range(startPow, n + startPow + 1):
        start = end
        l1, l2 = _powerSeriesTermCharCount(i, idxOffset)
        end += l1 + l2 + 1
        mid = start + l1
        # 幂级数的系数 a_n
        mob_coefGroup.add(mob_powerSeriesText[start:mid])
        # 幂级数的指数 x^n
        mob_termGroup.add(mob_powerSeriesText[mid : end - 1])
        # 加号
        mob_otherGroup.add(mob_powerSeriesText[end - 1])

    mob_powerSeriesText.mob_coefs = mob_coefGroup
    mob_powerSeriesText.mob_terms = mob_termGroup
    mob_powerSeriesText.mob_ellipsis = mob_powerSeriesText[end:]
    mob_powerSeriesText.mob_others = mob_otherGroup

    return mob_powerSeriesText


def _coefSrcGenerator(
    i: int, coef1Sym: str = "a", coef2Sym: str = "b", useSqr: bool = True
) -> Iterable[str]:
    if coef1Sym == coef2Sym:
        for j in range(i + 1):
            if i == j * 2 and useSqr:
                yield f"&{coef1Sym}_{j}^2"
            else:
                yield f"&{coef1Sym}_{j} {coef1Sym}_{i-j}"
    else:
        for j in range(i + 1):
            yield f"&{coef1Sym}_{j} {coef2Sym}_{i-j}"


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
    prefix: str = "f(x) g(x) &=",
    coef1Sym: str = "a",
    coef2Sym: str = "b",
    termSym: str = "x",
    useSqr: bool = True,
    **kwargs,
):
    # 创建两个幂级数的乘积公式
    src = f"{prefix}"
    if n > 0:
        if coef1Sym == coef2Sym and useSqr:
            src += f'&op(" ") &{coef1Sym}_0^2 \\\n&+ '
        else:
            src += f'&op(" ") &{coef1Sym}_0 {coef2Sym}_0 \\\n&+ '
    if n > 1:
        src += (
            f'&( op(" ") '
            f'&{coef1Sym}_0 {coef2Sym}_1 &+ &{coef1Sym}_1 {coef2Sym}_0 op(" ") &) op(" ") '
            f"&{termSym} \\\n&+ "
        )
    for i in range(2, n + 1):
        src += (
            f'&( op(" ")'
            + " &+ ".join(_coefSrcGenerator(i, coef1Sym, coef2Sym, useSqr))
            + ' op(" ") &) op(" ") '
            + f"&{termSym}^{i} \\\n&+ "
        )
    src += '&op(" ") &......'
    mob_text = TypstMath(src, **kwargs)

    prefixLen = len(TypstMath(prefix))
    mob_text.mob_prefix = mob_text[:prefixLen]
    mob_terms = mob_text.mob_terms = Group(Group())
    mob_coefs = mob_text.mob_coefs = Group()
    mob_others = mob_text.mob_others = Group(Group())

    start = end = prefixLen

    def _addCoefs(i, j):
        nonlocal start, end
        l1, l2 = _psProdCoefChCount(i, j)
        end += l1 + l2
        mob_coefsLine.add(mob_text[start:end])
        mob_text[start].color.set(GREEN)
        mob_text[start + l1].color.set(BLUE)
        start = end

    mob_coefsLine = Group()
    if n > 0:
        # 常数项系数
        _addCoefs(0, 0)
        mob_coefs.add(mob_coefsLine)
    for i in range(1, n + 1):
        mob_coefsLine = Group()
        mob_othersLine = Group()
        # 加号、左括号
        end += 2
        mob_othersLine.add(mob_text[start:end])
        start = end
        for j in range(i):
            # 系数
            _addCoefs(i, j)
            # 加号
            end += 1
            mob_othersLine.add(mob_text[start:end])
            start = end
        # 最后一个系数
        _addCoefs(i, i)
        # 右括号
        end += 1
        mob_othersLine.add(mob_text[start:end])
        start = end
        # 指数 x^i
        end += _psProdTermChCount(i)
        mob_terms.add(mob_text[start:end])
        start = end
        mob_coefs.add(mob_coefsLine)
        mob_others.add(mob_othersLine)
    # 最后一行的加号和省略号
    mob_others.add(mob_text[end:])
    mob_text.mob_noPrefix = mob_text[prefixLen:]

    return mob_text


def createPowerSeriesRec(
    n: int,
    coefSym: str = "a",
    termSym: str = "x",
    startDeg: int = 0,
    idxOffset: int = 0,
):
    src = ""
    if n > 0 and startDeg <= 0:
        src += f'&{coefSym}_{idxOffset} &op(" ") &arrow.l\\\n'
    if n > 1 and startDeg <= 1:
        src += f'&{coefSym}_{1 + idxOffset} &{termSym}^(op(" ")) &arrow.l\\\n'
    for i in range(max(2, startDeg), n + 1):
        src += f"&{coefSym}_{i + idxOffset} &{termSym}^{i} &arrow.l\\\n"
    mob_text = TypstMath(src)
    mob_lines = mob_text.mob_lines = Group()
    mob_coefs = mob_text.mob_coefs = Group()
    mob_terms = mob_text.mob_terms = Group()
    mob_arrows = mob_text.mob_arrows = Group()
    start = end = lineStart = 0
    if n > 0 and startDeg <= 0:
        # 系数
        end += len(str(idxOffset)) + 1
        mob_text[start].color.set(ORANGE)
        mob_coefs.add(mob_text[start:end])
        start = end
        # x 的幂次 (常数项没有，故添加空组)
        mob_terms.add(Group())
        # 箭头
        end += 1
        mob_arrows.add(mob_text[start:end])
        start = end
        # 整行
        mob_lines.add(mob_text[lineStart:end])
        lineStart = end

    if n > 1 and startDeg <= 1:
        # 系数
        end += len(str(idxOffset + 1)) + 1
        mob_text[start].color.set(ORANGE)
        mob_coefs.add(mob_text[start:end])
        start = end
        # x 的幂次 (一次项没有指数)
        end += 1
        mob_terms.add(mob_text[start:end])
        start = end
        # 箭头
        end += 1
        mob_arrows.add(mob_text[start:end])
        start = end
        # 整行
        mob_lines.add(mob_text[lineStart:end])
        lineStart = end

    for i in range(max(2, startDeg), n + 1):
        # 系数
        end += len(str(idxOffset + i)) + 1
        mob_text[start].color.set(ORANGE)
        mob_coefs.add(mob_text[start:end])
        start = end
        # x的幂次
        end += len(str(i)) + 1
        mob_terms.add(mob_text[start:end])
        start = end
        # 箭头
        end += 1
        mob_arrows.add(mob_text[start:end])
        start = end
        # 整行
        mob_lines.add(mob_text[lineStart:end])
        lineStart = end

    return mob_text


class CatalanScene(Timeline):
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
            mob_path = Group()
            mob_startingPath = Group()
            mob_badSegs = Group()
            mob_flippedPath = Group()

            nv = len(vsteps)
            nh = 2 * n - nv
            x0_scr, y0_scr, _ = mob_grids.c2p(nh, 0)
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
                # mob_dot = Dot(mob_grids.c2p(x, y), radius=0.04)
                # mob_dot.depth.set(-1)
                # mob_vg.add(mob_dot)
                isVertical = i in vsteps
                if isVertical:  # 纵向移动
                    x1, y1, x2, y2 = x, y, x, y + 1
                    clr1, clr2 = vcolor, hcolor
                else:  # 横向移动
                    x1, y1, x2, y2 = x, y, x + 1, y
                    clr1, clr2 = hcolor, vcolor

                mob_line = Line(mob_grids.c2p(x1, y1), mob_grids.c2p(x2, y2))
                mob_line.stroke.set(color=clr1)
                mob_line, mob_line2 = mob_line.copy(), mob_line
                ag_restoreColor.append(lambda: mob_line.anim.stroke.set(color=clr1))
                fn_restoreColor.append(lambda: mob_line.stroke.set(color=clr1))

                if isVertical:
                    ag_transformVPath.append(lambda: Transform(mob_line2, mob_line))
                    mob_line2.points.set_x(x0_scr)
                    y += 1
                    if y - 1 < x:
                        mob_badSegs.add(mob_line)
                else:
                    ag_transformHPath.append(lambda: Transform(mob_line2, mob_line))
                    mob_line2.points.set_y(y0_scr)
                    x += 1
                    if y < x:
                        mob_badSegs.add(mob_line)
                    if y == x < p0:  # 确定第一个越界点的位置
                        p0 = y
                if i > 2 * p0:
                    ag_changeColor.append(
                        lambda: mob_line.anim.stroke.set(color=markColor)
                    )
                    fn_changeColor.append(lambda: mob_line.stroke.set(color=markColor))
                    mob_flippedLine = Line(
                        mob_grids.c2p(y1 + 1, x1 - 1), mob_grids.c2p(y2 + 1, x2 - 1)
                    )
                    mob_flippedLine.stroke.set(color=markColor)
                    ag_changeFlippedColor.append(
                        lambda: mob_flippedLine.anim.stroke.set(color=markColor)
                    )
                    ag_restoreFlippedColor.append(
                        lambda: mob_flippedLine.anim.stroke.set(color=clr2)
                    )
                    fn_changeFlippedColor.append(
                        lambda: mob_flippedLine.stroke.set(color=markColor)
                    )
                    fn_restoreFlippedColor.append(
                        lambda: mob_flippedLine.stroke.set(color=clr2)
                    )
                    ag_flip.append(lambda: Transform(mob_line, mob_flippedLine))
                    ag_revert.append(lambda: Transform(mob_flippedLine, mob_line))
                    mob_flippedPath.add(mob_flippedLine)
                else:
                    mob_flippedPath.add(mob_line)
                # mob_line.add_tip(colorize=True, d_alpha=1)

                mob_path.add(mob_line)
                mob_startingPath.add(mob_line2)

            for i in range(2 * n):
                _helper(i)
            # mob_dot = Dot(mob_grids.c2p(x, y), radius=0.04)
            # mob_vg.add(mob_dot)

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
            ret.mob_startingPath = mob_startingPath
            ret.mob_path = mob_path
            ret.mob_badSegs = mob_badSegs
            ret.mob_flippedPath = mob_flippedPath
            ret.p0 = p0
            ret.mob_pathToFlip = mob_path[p0 * 2 + 1 :]

            return ret

        # 网格
        mob_grids = NumberPlane(
            x_range=(-7, 24, 1),
            y_range=(-4, 14, 1),
            unit_size=u,
            faded_line_ratio=0,
            background_line_style={"stroke_alpha": 0.5},
        )
        mob_grids.points.shift(coordOrigin)
        mob_grids.depth.set(2)

        # 黑色背景
        mob_blackRect = Rect(rectw, recth)
        mob_blackRect.fill.set(color=BLACK, alpha=0.75)
        mob_blackRect.stroke.set(alpha=0)
        mob_blackRect.points.set_y(coordOrigin[1] - fh).rotate(
            PI / 4, about_point=coordOrigin
        )
        mob_blackRect.depth.set(1)

        # 斜线 y = x, y = x - 1, y = x + 1
        mob_dividingLine = Line(
            (-fw / 2, 0, 0),
            (fw / 2, 0, 0),
        )
        mob_dividingLine.radius.set(0.015)
        mob_dividingLine.stroke.set(alpha=0.5)

        mob_dividingLine.points.set_y(coordOrigin[1]).rotate(
            PI / 4, about_point=coordOrigin
        )

        mob_upperLine = mob_dividingLine.copy()
        mob_lowerLine = mob_dividingLine.copy()
        for mob in (mob_upperLine, mob_lowerLine):
            mob.stroke.set(color=ORANGE)
        mob_upperLine.points.shift(UP * u)
        mob_lowerLine.points.shift(DOWN * u)

        # 禁止进入标记
        mob_sign = SVGItem(DIR / "assets/image/abandoned.svg")
        mob_sign.points.scale(0.6).shift(signPos)

        # 起点、终点旗标
        mob_startFlag = createFlag(color=RED, rotation=-PI / 12)
        mob_stopFlag = createFlag(color=GREEN, rotation=PI / 12)
        mob_startFlag.points.shift(mob_grids.c2p(0, 0))
        mob_stopFlag.points.shift(mob_grids.c2p(n, n))

        mob_startCoordText = TypstMath("(0, 0)")
        mob_stopCoordText = TypstMath("(n, n)")
        mob_flippedCoordText = TypstMath("(n + 1, n - 1)")
        mob_startCoordText.points.next_to(mob_grids.c2p(0, 0), DOWN)
        mob_stopCoordText.points.next_to(mob_grids.c2p(n, n), RIGHT)
        mob_flippedCoordText.points.next_to(mob_grids.c2p(n + 1, n - 1), RIGHT)

        self.play(Create(mob_grids))
        self.play(FadeIn(mob_blackRect))
        self.play(Create(mob_dividingLine))
        self.play(FadeIn(mob_sign, scale=0.8))
        self.play(
            FadeIn(mob_startFlag, shift=DOWN * 0.2, scale=0.5, rate_func=rush_into),
            FadeIn(mob_stopFlag, shift=DOWN * 0.2, scale=0.5, rate_func=rush_into),
            Write(mob_startCoordText),
            Write(mob_stopCoordText),
            duration=0.5,
        )
        self.forward(0.25)
        self.play(
            Flash(mob_grids.c2p(0, 0)),
            Flash(mob_grids.c2p(n, n)),
            duration=1,
        )
        self.forward(0.5)

        mob_texts = Group()
        maxTextWidth = 0.4

        for d in range(2 * n + 1):
            mob_diagonalGroup = Group()
            for i in range(d // 2 + 1) if d <= n else range(d - n, d // 2 + 1):
                j = d - i
                mob_text = boolean_ops.Union.from_group(Text(f"{catalanTable[i,j]}"))
                mob_text.depth.set(-2)
                mob_text.points.shift(mob_grids.c2p(i, j))
                mob_text.set_stroke_background(True)
                mob_text.color.set(YELLOW, alpha=1)
                mob_text.stroke.set(color=BLACK, alpha=0.75)
                mob_text.radius.set(0.05)
                w = mob_text.points.box.width
                if w > maxTextWidth:
                    mob_text.points.set_width(maxTextWidth)
                mob_diagonalGroup.add(mob_text)
            mob_texts.add(mob_diagonalGroup)

        ag = []
        mob_lastGroup = None
        for i, mob_diagonalGroup in enumerate(mob_texts):
            dur = 0.6 - 0.5 * (rush_from(i / n / 2))
            if mob_lastGroup is None:
                ag.append(FadeIn(mob_diagonalGroup, duration=dur))
            else:
                ag1 = []
                if i <= n:
                    if i & 1 == 1:  # i odd
                        for j, mob_text in enumerate(mob_lastGroup[:-1]):
                            ag1.extend(
                                (
                                    Transform(
                                        mob_text,
                                        mob_diagonalGroup[j],
                                        hide_src=False,
                                    ),
                                    Transform(
                                        mob_text,
                                        mob_diagonalGroup[j + 1],
                                        hide_src=False,
                                    ),
                                )
                            )
                        ag1.append(
                            Transform(
                                mob_lastGroup[-1],
                                mob_diagonalGroup[-1],
                                hide_src=False,
                            )
                        )
                    else:  # i even
                        for j, mob_text in enumerate(mob_lastGroup):
                            ag1.extend(
                                (
                                    Transform(
                                        mob_text,
                                        mob_diagonalGroup[j],
                                        hide_src=False,
                                    ),
                                    Transform(
                                        mob_text,
                                        mob_diagonalGroup[j + 1],
                                        hide_src=False,
                                    ),
                                )
                            )
                else:
                    if i & 1 == 1:  # i odd
                        for j, mob_text in enumerate(mob_diagonalGroup):
                            ag1.extend(
                                (
                                    Transform(
                                        mob_lastGroup[j],
                                        mob_text,
                                        hide_src=False,
                                    ),
                                    Transform(
                                        mob_lastGroup[j + 1],
                                        mob_text,
                                        hide_src=False,
                                    ),
                                )
                            )
                    else:  # i even
                        for j, mob_text in enumerate(mob_diagonalGroup[:-1]):
                            ag1.extend(
                                (
                                    Transform(
                                        mob_lastGroup[j],
                                        mob_text,
                                        hide_src=False,
                                    ),
                                    Transform(
                                        mob_lastGroup[j + 1],
                                        mob_text,
                                        hide_src=False,
                                    ),
                                )
                            )
                        ag1.append(
                            Transform(
                                mob_lastGroup[-1],
                                mob_diagonalGroup[-1],
                                hide_src=False,
                            )
                        )

                ag1 = AnimGroup(*ag1, duration=dur)
                ag.append(ag1)
            mob_lastGroup = mob_diagonalGroup
        ag = Succession(*ag)

        self.play(FadeOut(Group(mob_startCoordText, mob_stopCoordText)), duration=0.5)
        self.play(ag)
        self.forward(0.5)
        self.play(FadeOut(mob_texts))
        self.forward(0.5)
        self.play(FadeOut(mob_dividingLine), FadeOut(mob_blackRect), FadeOut(mob_sign))

        samplePath1 = frozenset((0, 1, 2, 8, 9, 10, 15, 16, 17, 18))
        samplePath2 = frozenset((0, 1, 2, 7, 8, 9, 15, 16, 17, 18))
        pathAc1 = animatePath(samplePath1)
        pathAc2 = animatePath(samplePath2)
        self.play(FadeIn(pathAc1.mob_startingPath))
        self.play(pathAc1.ag_transformHPath)
        self.play(pathAc1.ag_transformVPath)

        mob_binom1 = TypstMath("binom(2n, n)")
        mob_binom2 = TypstMath("binom(2n, n-1)")
        mob_catalanFormula = TypstMath("a_(n) = sum_(k = 1)^(n) a_(k - 1) a_(n - k)")
        mob_binom1.points.next_to(mob_grids.c2p(n, n), RIGHT)
        mob_binom2.points.next_to(mob_grids.c2p(n + 1, n - 1), RIGHT)
        mob_catalanFormula.points.shift((4, 1, 0)).scale(1.5)

        self.play(FadeIn(mob_binom1))
        self.play(mob_binom1.anim.points.move_to((4, 0.5, 0)).scale(1.5))
        self.play(Create(mob_dividingLine), FadeIn(mob_blackRect))

        def blink(mobs: Iterable[Item], dt=0.15, n=4):
            for _ in range(n):
                for mob in mobs:
                    mob.stroke.set(alpha=0.5)
                self.forward(dt)
                for mob in mobs:
                    mob.stroke.set(alpha=1)
                self.forward(dt)

        blink(pathAc1.mob_badSegs)
        # self.play(*(mob.anim.stroke.set(alpha=0.5) for mob in animPathRet.mob_badSegs))
        self.play(
            AnimGroup(
                *(
                    Indicate(mob, scale_factor=1, duration=0.25)
                    for mob in pathAc1.mob_path
                ),
                lag_ratio=0.25,
            )
        )
        mob_focusBox = createFocusBox()
        mob_focusBox.points.shift(mob_grids.c2p(*it.repeat(pathAc1.p0, 2)))
        self.play(FadeIn(mob_focusBox, scale=0.5), FadeOut(mob_blackRect))
        self.play(
            mob_focusBox.anim.points.move_to(mob_grids.c2p(pathAc1.p0 + 1, pathAc1.p0))
        )
        self.play(
            Rotate(mob_focusBox, PI / 2, duration=1),
            Succession(
                mob_focusBox.anim.points.scale(2),
                mob_focusBox.anim.points.scale(0.5),
                duration=1,
            ),
        )
        self.play(pathAc1.ag_changeColor, FadeIn(mob_stopCoordText), duration=0.5)
        self.play(Create(mob_lowerLine))
        vec = mob_grids.c2p(1, -1) - coordOrigin
        self.play(
            pathAc1.ag_flip,
            mob_stopFlag.anim.points.shift(vec),
            Transform(mob_stopCoordText, mob_flippedCoordText),
            duration=0.5,
        )
        self.play(Flash(mob_grids.c2p(n + 1, n - 1)))
        self.forward(1)
        self.play(
            pathAc1.ag_revert,
            mob_stopFlag.anim.points.shift(-vec),
            Transform(mob_flippedCoordText, mob_stopCoordText),
            duration=0.5,
        )
        self.forward(0.5)
        self.play(
            pathAc1.ag_flip,
            mob_stopFlag.anim.points.shift(vec),
            Transform(mob_stopCoordText, mob_flippedCoordText),
            duration=0.5,
        )
        self.play(FadeOut(pathAc1.mob_flippedPath), FadeOut(mob_focusBox), duration=0.5)

        pathAc2.restoreFlippedColor()
        mob_focusBox.points.move_to(mob_grids.c2p(pathAc2.p0 + 1, pathAc2.p0))

        self.play(Succession(*map(Create, pathAc2.mob_flippedPath)), duration=1)
        self.play(FadeIn(mob_focusBox, scale=0.5))
        self.play(
            Rotate(mob_focusBox, PI / 2, duration=1),
            Succession(
                mob_focusBox.anim.points.scale(2),
                mob_focusBox.anim.points.scale(0.5),
                duration=1,
            ),
        )

        self.play(pathAc2.ag_changeFlippedColor, duration=0.5)
        pathAc2.changeColor()
        self.play(
            pathAc2.ag_revert,
            mob_stopFlag.anim.points.shift(-vec),
            Transform(mob_flippedCoordText, mob_stopCoordText),
            duration=0.5,
        )
        self.play(FadeIn(mob_blackRect))
        blink(pathAc2.mob_badSegs)
        self.play(
            pathAc2.ag_flip,
            mob_stopFlag.anim.points.shift(vec),
            FadeOut(mob_stopCoordText),
            FadeOut(mob_blackRect),
            duration=0.5,
        )
        self.play(FadeIn(mob_binom2))
        self.play(
            pathAc2.ag_revert,
            mob_stopFlag.anim.points.shift(-vec),
            FadeIn(mob_blackRect),
            mob_binom2.anim.points.next_to(mob_grids.c2p(n, n), RIGHT),
            duration=0.5,
        )

        mob_binom3 = TypstMath(
            "binom(2n, n) - binom(2n, n - 1) = 1/(n + 1) binom(2n, n)"
        )
        mob_binom3.points.move_to((4, 0, 0))
        self.play(
            Transform(mob_binom1, mob_binom3[:5]),
            FadeIn(mob_binom3[5]),
            Transform(mob_binom2, mob_binom3[6:13]),
            FadeIn(mob_binom3[13:]),
        )
        self.forward(2)
        self.play(
            FadeOut(Group(mob_binom3, pathAc2.mob_path, mob_lowerLine, mob_focusBox))
        )

        mob_anTexts = Group()
        for i in range(n + 1):
            mob_anText = TypstMath(f"a_{i}")
            mob_anText.points.shift(mob_grids.c2p(i, i))
            mob_anText.depth.set(-2)
            mob_anTexts.add(mob_anText)

        self.play(AnimGroup(*map(Write, mob_anTexts), lag_ratio=0.25), duration=3)

        showingPow = 4
        mob_psText = createPowerSeries(showingPow)
        mob_psText.points.move_to((3, -1, 0))

        ag = []
        for mob_anText, mob_coefText in zip(
            mob_anTexts,
            it.chain(mob_psText.mob_coefs, it.repeat(mob_psText.mob_ellipsis)),
        ):
            ag.append(Transform(mob_anText, mob_coefText, hide_src=False))

        self.play(*ag, FadeIn(Group(mob_psText.mob_others, mob_psText.mob_terms)))
        self.play(ShowPassingFlashAround(mob_psText, time_width=3))
        self.play(FadeOut(mob_anTexts))
        self.forward(1)
        self.play(Write(mob_catalanFormula), duration=1)
        self.play(ShowPassingFlashAround(mob_catalanFormula, time_width=3))
        self.forward(2)
        self.play(FadeOut(mob_catalanFormula))

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
                mob_focusBox.points.move_to(mob_grids.c2p(pathAc.p0, pathAc.p0))
                self.play(
                    FadeIn(pathAc.mob_path, duration=0.25),
                    FadeIn(mob_focusBox, scale=0.5, duration=0.5),
                )
            else:
                self.play(
                    Transform(lastPathAc.mob_path, pathAc.mob_path, duration=0.25),
                    mob_focusBox.anim(duration=0.5).points.move_to(
                        mob_grids.c2p(pathAc.p0, pathAc.p0)
                    ),
                )
            self.forward(0.75)
            lastPathAc = pathAc

        k = pathAc.p0
        mob_p0Text = TypstMath("(k, k)")
        mob_p0Text.points.next_to(mob_grids.c2p(k, k), DR)
        self.play(FadeIn(Group(mob_p0Text, mob_stopCoordText)))

        mob_upperTriangle = Polygon(
            *map(mob_grids.c2p, *np.array(((k, k), (k, n), (n, n))).T)
        )
        mob_lowerTriangle1 = Polygon(
            *map(mob_grids.c2p, *np.array(((0, 0), (0, k), (k, k))).T)
        )
        mob_lowerTriangle2 = Polygon(
            *map(mob_grids.c2p, *np.array(((0, 1), (0, k), (k - 1, k))).T)
        )
        mob_upperText = TypstMath("a_(n - k)")
        mob_lowerText = TypstMath("a_(k - 1)")
        mob_upperText.points.move_to(mob_grids.c2p((n + 2 * k) / 3, (2 * n + k) / 3))
        mob_lowerText.points.move_to(mob_grids.c2p((k - 1) / 3, (1 + 2 * k) / 3))
        for mob in (mob_upperText, mob_lowerText):
            mob.set_stroke_background(True)
            mob.stroke.set(color=BLACK, alpha=0.75)
            mob.radius.set(0.05)
            mob.depth.set(-3)

        if n - k > 3:
            mob_upperText.points.scale(1.5)
        if k - 1 > 3:
            mob_lowerText.points.scale(1.5)

        mob_crossTemplate = SVGItem(DIR / "assets/image/cross.svg", width=0.25)
        for mob in (mob_upperTriangle, mob_lowerTriangle1, mob_lowerTriangle2):
            mob.fill.set(color=YELLOW, alpha=0.25)
            mob.radius.set(0)
            mob.depth.set(3)

        mob_crosses = Group()
        for i in range(1, k):
            mob_cross = mob_crossTemplate.copy()
            mob_cross.points.move_to(mob_grids.c2p(i, i))
            mob_crosses.add(mob_cross)

        self.play(FadeIn(mob_upperTriangle))
        self.play(FadeIn(mob_upperText))
        self.forward(1)
        self.play(FadeIn(mob_lowerTriangle1))

        for _ in range(4):
            self.show(mob_crosses)
            self.forward(0.25)
            self.hide(mob_crosses)
            self.forward(0.25)

        self.forward(1)

        self.hide(mob_dividingLine, mob_blackRect)

        # 将分界线分成3段
        mob_dividingLineUpper = Line(mob_grids.c2p(k, k), mob_grids.c2p(14, 14))
        mob_dividingLineMiddle = Line(mob_grids.c2p(0, 0), mob_grids.c2p(k, k))
        mob_dividingLineMiddleTr = Line(mob_grids.c2p(0, 1), mob_grids.c2p(k - 1, k))
        mob_dividingLineLower = Line(mob_grids.c2p(-8, -8), mob_grids.c2p(0, 0))

        # 将黑色矩形改成多边形
        mob_blackPolygon = Polygon(
            *map(
                mob_grids.c2p,
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
        mob_blackPolygonTr = Polygon(
            *map(
                mob_grids.c2p,
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
        for mob in (mob_blackPolygon, mob_blackPolygonTr):
            mob.fill.set(color=BLACK, alpha=0.75)
            mob.radius.set(0)
            mob.depth.set(1)
        for mob in (
            mob_dividingLineUpper,
            mob_dividingLineMiddle,
            mob_dividingLineMiddleTr,
            mob_dividingLineLower,
        ):
            mob.stroke.set(alpha=0.5)
            mob.radius.set(0.015)
        self.show(
            mob_dividingLineUpper,
            mob_dividingLineMiddle,
            mob_dividingLineLower,
            mob_blackPolygon,
        )
        vec = mob_grids.c2p(-1, 0) - coordOrigin
        self.play(
            pathAc.mob_path[0].anim(duration=0.5).color.set(YELLOW),
            Flash(pathAc.mob_path[0], duration=1),
            pathAc.mob_path[2 * k - 1].anim(duration=0.5).color.set(YELLOW),
            Flash(pathAc.mob_path[2 * k - 1], duration=1),
        )
        self.play(
            Transform(mob_lowerTriangle1, mob_lowerTriangle2),
            Transform(mob_dividingLineMiddle, mob_dividingLineMiddleTr),
            Transform(mob_blackPolygon, mob_blackPolygonTr),
        )
        self.play(FadeIn(mob_lowerText))

        mob_upperTextCp = mob_upperText.copy()
        mob_lowerTextCp = mob_lowerText.copy()
        for mob in (mob_upperTextCp, mob_lowerTextCp):
            mob.radius.set(0)
            mob.depth.set(-4)

        self.play(
            Transform(mob_upperTextCp, mob_catalanFormula[-4:]),
            Transform(mob_lowerTextCp, mob_catalanFormula[-8:-4]),
        )
        self.play(FadeIn(mob_catalanFormula[:8]))
        self.forward(2)

        mob_catalanFormula2 = TypstMath("a_(n) = sum_(k = 1)^(n) a_(k - 1) a_(n - k)")
        mob_catalanFormula2.points.move_to((4, 1, 0))

        self.play(
            mob_psText.anim.points.to_center().to_border(UP, 0.75),
            Transform(mob_catalanFormula, mob_catalanFormula2),
            FadeOut(
                Group(
                    mob_grids,
                    mob_upperTriangle,
                    mob_lowerTriangle2,
                    mob_focusBox,
                    mob_startFlag,
                    mob_stopFlag,
                    mob_dividingLineUpper,
                    mob_dividingLineMiddleTr,
                    mob_dividingLineLower,
                    mob_blackPolygonTr,
                    mob_upperText,
                    mob_lowerText,
                    mob_p0Text,
                    mob_stopCoordText,
                    pathAc.mob_path,
                )
            ),
        )

        def createPsBoxes(mob_ps: TypstMath, **kwargs):
            mob_boxes = Group()
            for mob_coef in mob_ps.mob_coefs:
                mob_boxes.add(SurroundingRect(mob_coef, **kwargs))
            return mob_boxes

        showingPow2 = 6
        mob_psText1 = createPowerSeries(showingPow2)
        mob_psText2 = createPowerSeries(showingPow2)
        for mob in mob_psText1.mob_coefs:
            mob[0].color.set(GREEN)
        for mob in mob_psText2.mob_coefs:
            mob[0].color.set(BLUE)
        mob_psText1.points.move_to(mob_psText)
        mob_psText2.points.next_to(mob_psText1, DOWN, buff=0.3)

        mob_psProdText = createPowerSeriesProduct(
            showingPow2,
            prefix="f^2(x) &=",
            coef2Sym="a",
            useSqr=False,
        )

        self.play(TransformMatchingShapes(mob_psText, mob_psText1), duration=0.5)
        self.play(Transform(mob_psText1, mob_psText2, hide_src=False), duration=0.5)

        mob_psProdText.points.next_to(mob_psText2, DOWN, buff=0.5)
        # self.show(mob_psProdText)
        self.play(FadeIn(mob_psProdText.mob_prefix))

        mob_boxes1 = createPsBoxes(mob_psText1, color=GREEN, buff=0.05)
        mob_boxes2 = createPsBoxes(mob_psText2, color=BLUE, buff=0.05)

        mob_showingBox1 = None
        mob_showingBox2 = None
        for i in range(showingPow2 + 1):
            for j in range(i + 1):
                mob_box1 = mob_boxes1[j]
                mob_box2 = mob_boxes2[i - j]
                if mob_showingBox1 is None:
                    self.play(FadeIn(Group(mob_box1, mob_box2)), duration=0.2)
                else:
                    self.play(
                        Transform(mob_showingBox1, mob_box1),
                        Transform(mob_showingBox2, mob_box2),
                        duration=0.2,
                        rate_func=rush_from,
                    )
                self.play(FadeIn(mob_psProdText.mob_coefs[i][j]), duration=0.2)
                mob_showingBox1 = mob_box1
                mob_showingBox2 = mob_box2
            self.play(
                FadeOut(Group(mob_showingBox1, mob_showingBox2)),
                FadeIn(mob_psProdText.mob_others[i]),
                FadeIn(mob_psProdText.mob_terms[i]),
                duration=0.2,
            )
            mob_showingBox1 = mob_showingBox2 = None
        self.play(FadeIn(mob_psProdText.mob_others[-1]), duration=0.25)
        self.play(
            mob_psProdText.mob_noPrefix.anim.points.shift(RIGHT * 0.5),
            FadeOut(mob_psProdText.mob_prefix),
            duration=0.5,
        )

        mob_recursiveTerms = createPowerSeriesRec(showingPow2, idxOffset=1)
        mob_recursiveTerms.points.move_to(mob_psProdText, aligned_edge=UL).shift(
            (-0.5, -0.2, 0)
        )
        ag1, ag2 = [], []
        for i in range(showingPow2 + 1):
            mob_coefsLine = mob_psProdText.mob_coefs[i]
            mob_recLine = mob_recursiveTerms.mob_lines[i]
            ag1.append(ShowPassingFlashAround(mob_coefsLine, time_width=3, duration=1))
            ag2.append(
                Transform(
                    mob_coefsLine,
                    Group(mob_recLine),
                    hide_src=False,
                    duration=0.75,
                )
            )

        self.play(ag1[-1])
        self.play(Indicate(mob_catalanFormula2))
        self.play(ag2[-1])
        self.play(AnimGroup(*reversed(ag2[:-1]), lag_ratio=0.5))
        self.forward(2)

        mob_dividingLine = Line(ORIGIN, (12, 0, 0))
        mob_psText3 = createPowerSeries(showingPow2, prefix="f^2(x) =", idxOffset=1)
        mob_psText4 = createPowerSeries(showingPow2, prefix="x f^2(x) =", startPow=1)
        mob_psText5 = createPowerSeries(showingPow2, prefix="x f^2(x) + 1 =")
        mob_dividingLine.points.next_to(mob_psText2, DOWN, buff=0.3)
        mob_dividingLine.radius.set(0.015)
        mob_psText3.points.next_to(mob_dividingLine, DOWN, buff=0.3)
        mob_psText4.points.next_to(mob_psText3, DOWN, buff=0.3)
        mob_psText5.points.next_to(mob_psText4, DOWN, buff=0.3)

        for mob_ps in mob_psText3, mob_psText4:
            for mob_coef in mob_ps.mob_coefs:
                mob_coef[0].color.set(ORANGE)

        for mob_coef in mob_psText5.mob_coefs[1:]:
            mob_coef[0].color.set(ORANGE)

        ag = []

        for i in range(showingPow2 + 1):
            ag.append(
                Transform(mob_recursiveTerms.mob_coefs[i], mob_psText3.mob_coefs[i])
            )
            ag.append(
                Transform(mob_recursiveTerms.mob_terms[i], mob_psText3.mob_terms[i])
            )

        self.play(
            *ag,
            FadeIn(mob_psText3.mob_others),
            FadeIn(mob_psText3.mob_ellipsis),
            FadeOut(mob_psProdText.mob_noPrefix),
            FadeOut(mob_recursiveTerms.mob_arrows),
            FadeOut(mob_catalanFormula2),
            Create(mob_dividingLine),
        )

        def animatePsTransform(mob_ps1, mob_ps2, offset=0, matchOthers=False):
            l1, l2 = len(mob_ps1.mob_coefs), len(mob_ps2.mob_coefs)
            l = min(l1, l2)
            ag = []
            mob_coefs1 = mob_ps1.mob_coefs
            mob_coefs2 = mob_ps2.mob_coefs
            mob_terms1 = mob_ps1.mob_terms
            mob_terms2 = mob_ps2.mob_terms
            if offset > 0:
                ag.extend(
                    FadeIn(Group(mob1, mob2))
                    for mob1, mob2 in zip(
                        it.chain(
                            mob_coefs1[l:],
                            mob_coefs2[:offset],
                            mob_coefs2[l + offset :],
                        ),
                        it.chain(
                            mob_terms1[l:],
                            mob_terms2[:offset],
                            mob_terms2[l + offset :],
                        ),
                    )
                )
            for i in range(l - offset) if offset > 0 else range(-offset, l):
                mob_coef1 = mob_ps1.mob_coefs[i]
                mob_coef2 = mob_ps2.mob_coefs[i + offset]
                mob_term1 = mob_ps1.mob_terms[i]
                mob_term2 = mob_ps2.mob_terms[i + offset]
                ag.append(Transform(mob_coef1, mob_coef2, hide_src=False))
                if len(mob_term1) == 0:
                    ag.append(FadeIn(mob_term2))
                elif len(mob_term2) == 0:
                    ag.append(FadeOut(mob_term1))
                else:
                    ag.append(Transform(mob_term1, mob_term2, hide_src=False))
            self.play(
                (
                    TransformMatchingShapes(
                        mob_ps1.mob_others.copy(), mob_ps2.mob_others
                    )
                    if matchOthers
                    else Transform(
                        mob_ps1.mob_others, mob_ps2.mob_others, hide_src=False
                    )
                ),
                Transform(mob_ps1.mob_ellipsis, mob_ps2.mob_ellipsis, hide_src=False),
                *ag,
                duration=1,
            )

        animatePsTransform(mob_psText3, mob_psText4)
        animatePsTransform(mob_psText4, mob_psText5, offset=1)

        # 生成函数所满足的方程及其表达式
        mob_gfEquation = TypstMath("f(x) = x f^2(x) + 1")
        mob_gf = TypstMath("f(x) = (1 - sqrt(1 - 4x))/(2x)")
        mob_gfProcess = TypstDoc((DIR / "assets/catalan-gf-process.typ").read_text())
        mob_gfResult = TypstMath(
            "(1 - sqrt(1 - 4x))/(2x) = sum_(n = 0)^(infinity) 1/(n + 1) binom(2n, n) x^n"
        )
        mob_gfEquation.points.move_to((-2.5, -1.75, 0))
        mob_gf.points.move_to((2.5, -1.7, 0))
        mob_gfProcess.points.shift(DOWN * 1.5)
        mob_gfResult.points.next_to(mob_gfProcess, DOWN, buff=1)

        self.play(Write(mob_gfEquation))
        self.play(Transform(mob_gfEquation, mob_gf, hide_src=False))
        self.play(
            FadeOut(
                Group(
                    mob_psText1,
                    mob_psText2,
                    mob_dividingLine,
                    mob_psText3,
                    mob_psText4,
                    mob_psText5,
                )
            ),
            Group(mob_gfEquation, mob_gf).anim.points.shift(UP * 5),
        )
        self.play(Write(mob_gfProcess), duration=2)
        self.play(Transform(mob_gfProcess, mob_gfResult, hide_src=False))
        self.play(ShowPassingFlashAround(mob_gfResult, time_width=3, duration=1))
        self.forward(2)


class CatalanGenFuncScene(Timeline):
    def construct(self): ...


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
##################################
