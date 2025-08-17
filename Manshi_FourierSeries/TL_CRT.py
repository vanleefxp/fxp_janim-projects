import math

from janim.imports import *
from egcd import egcd
from frozendict import frozendict

with reloads():
    from common import *


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

        gridConfig = frozendict(h_buff=0.3, v_buff=0.5, fill_rows_first=False)
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

        gridConfig = frozendict(fill_rows_first=False, h_buff=0.2, v_buff=0.4)
        rectConfig = frozendict(
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


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
