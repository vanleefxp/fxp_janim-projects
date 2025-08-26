from janim.imports import *

from common import *
from fourier_figure import FourierFigure

with reloads():
    from common import *
    from fourier_figure import FourierFigure


class TL_WindingGraph(Timeline):
    CONFIG = config

    def construct(self):
        coefs_file = DIR / "assets/data/fourier-coefs/cxk.npz"
        radius = 2
        axis_pad = 0.75
        coord_distance = 6.33
        max_n = 50
        fourier_anim_time = 10
        fourier_smooth_time = 0.5
        animEasing = smoothBoth(fourier_smooth_time * 2 / fourier_anim_time)

        fig: FourierFigure = FourierFigure(coefs_file)[:max_n]

        # 重新排序傅里叶系数
        coefs_alternating = fftalt(fig.coefs)
        coefsabs_alternating = np.abs(coefs_alternating)
        max_n = fig.max_n

        axis_range = (-radius - axis_pad, radius + axis_pad, radius)
        i_coord = Axes(
            axis_range, axis_range, x_axis_config=axis_cfg, y_axis_config=axis_cfg
        )
        i_coord2 = (
            Axes(axis_range, axis_range, x_axis_config=axis_cfg, y_axis_config=axis_cfg)
            .mark.set(RIGHT * coord_distance)
            .r
        )

        # 生成向量、圆和点
        i_vecs = Group[Vector](
            *(
                Line(
                    ORIGIN,
                    ORIGIN,
                    buff=0,
                    stroke_radius=interpolate(0.01, 0.0025, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i in alternatingSignedInts(max_n)
            )
        )
        i_circs = Group[Circle](
            *(
                Circle(
                    coefabs * radius,
                    stroke_radius=interpolate(0.01, 0.005, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i, coefabs in zip(alternatingSignedInts(), coefsabs_alternating)
            )
        )
        i_points = Group[Dot](
            *(
                Dot(radius=interpolate(0.04, 0.01, 1 - np.exp(-np.abs(i) / 5)))
                for i in alternatingSignedInts(max_n)
            )
        )
        i_circs[0].set(stroke_alpha=0)
        i_graphingItems = Group(i_vecs, i_circs, i_points)

        # 实时更新向量、圆和点的位置
        def updateVectors(t: float, items=i_graphingItems):
            i_vecs, i_circs, i_dots = items
            coord2Origin = i_coord2.mark.get()

            components = fig.components(t)
            components_cp = components.copy()
            components[0] = components_cp[0]
            components[1::2] = components_cp[1 : fig.max_n]
            components[2::2] = components_cp[-1 : -fig.max_n : -1]
            components_cumsum = np.cumsum(components)

            for (pc_prev, pc_next), i_vec, i_circ, i_dot in zip(
                it.pairwise(it.chain((0j,), components_cumsum)), i_vecs, i_circs, i_dots
            ):
                p_prev = complex2point(pc_prev * radius) + coord2Origin
                p_next = complex2point(pc_next * radius) + coord2Origin
                i_vec.points.set_start_and_end(p_prev, p_next)
                i_circ.points.move_to(p_prev)
                i_dot.points.move_to(p_next)

        updateVectors(0)

        # i_coefFormula = (
        #     TypstMath(
        #         r"""
        #         a_n
        #         &= angle.l up(e)^(2 pi up(i) n t), phi(t) angle.r \
        #         &= integral_0^1 up(e)^(- 2 pi up(i) n t) phi(t) dif t quad (n in ZZ)
        #         """,
        #         **textCfg,
        #         preamble=f"""
        #         #show sym.phi: set text(fill: rgb("{BILI_PINK}"), stroke: rgb("{BILI_PINK}"))
        #         """,
        #     )
        #     .points.shift((6, 1.8, 0))
        #     .r
        # )
        i_nLabel = (
            TypstMath("n = .", **textCfg)
            .points.to_border(UL, buff=0.375)
            .shift(RIGHT * (coord_distance / 2))
            .r
        )
        i_nLabel[-1].set(alpha=0)
        # i_values[2].set(alpha=0)
        # i_values[-1].set(alpha=0)
        # i_innerProductNote = (
        #     TypstDoc(
        #         """
        #         / 注: 自变量为实数的复值函数 $f(x)$, $g(x)$ 在区间 $[a, b]$ 上的内积定义为 $angle.l f(x), g(x) angle.r = integral_a^b macron(f)(x) g(x) dif x$, 其中 $macron(f)(x)$ 表示 $f(x)$ 的复共轭。
        #         """,
        #         **textCfg,
        #         additional_preamble="""
        #         #set text(size: 0.75em)
        #         #set page(width: 9cm, height: auto)
        #         """,
        #     )
        #     .points.shift((6, -5, 0))
        #     .r
        # )

        def getWindingFigure(n: float = 0) -> ParametricCurve:
            i_figure = ParametricCurve(
                lambda t: complex2point(fig(t) * np.exp(-2j * np.pi * n * t)) * radius,
                (0, 1, 0.01 / (abs(n) + 1)),
                color=BILI_PINK,
            )
            return i_figure

        i_figure = getWindingFigure()
        i_coefPoint = Dot(radius=0.05, depth=-1)
        i_lineToCenter = Line(stroke_radius=0.01, depth=-1)
        i_circToCenter = Circle(stroke_radius=0.01, depth=-1)

        def createNValueText(n: float) -> Text:
            i_nValue = Text(f"{round(n)}".replace("-", "\u2212"), **textCfg)
            i_nValue.points.shift(
                i_nLabel[-1].points.self_box.get(DL) - i_nValue[0].get_mark_orig()
            )
            return i_nValue

        # def createCoefValueText(n: float) -> Text:
        #     coef = fig.coefFn(n)
        #     coef_abs = np.abs(coef)
        #     coef_angle = np.angle(coef)
        #     i_coefValue = Text(
        #         f"{coef_abs:.4f} exp({coef_angle:.4f} i)".replace("-", "\u2212"),
        #         **textCfg,
        #     )
        #     i_coefValue.points.shift(
        #         i_values[-1].points.self_box.get(DL) - i_coefValue[0].get_mark_orig()
        #     )
        #     return i_coefValue

        i_nValue = createNValueText(0)
        # i_coefValue = createCoefValueText(0)

        i_windingGraphItems = Group(
            i_figure, i_coefPoint, i_lineToCenter, i_circToCenter
        )

        def updateWindingGraphItems(
            items: Group[VItem] = i_windingGraphItems, n: float = 0
        ) -> None:
            i_figure, i_coefPoint, i_lineToCenter, i_circToCenter = items
            i_figure.become(getWindingFigure(n))
            coef = fig.coefFn(n)
            coefPoint = complex2point(coef) * radius
            i_coefPoint.points.move_to(coefPoint)
            i_lineToCenter.points.set_start_and_end(ORIGIN, coefPoint)
            i_circToCenter.become(
                Circle(radius=np.abs(coef) * radius, stroke_radius=0.01)
            )

        def createNValueUpdaterFn(n_start, n_end):
            def nValueUpdaterFn(params: UpdaterParams):
                n = interpolate(n_start, n_end, params.alpha)
                return createNValueText(n)

            return nValueUpdaterFn

        # def createCoefValueUpdaterFn(n_start, n_end):
        #     def coefValueUpdaterFn(params: UpdaterParams):
        #         n = interpolate(n_start, n_end, params.alpha)
        #         return createCoefValueText(fig.coefFn(n))

        #     return coefValueUpdaterFn

        def createUpdaterFn(n_start, n_end):
            def updarterFn(items: Group[VItem], params: UpdaterParams):
                n = interpolate(n_start, n_end, params.alpha)
                updateWindingGraphItems(items, n)

            return updarterFn

        self.play(Create(i_coord))
        self.play(Create(i_figure, auto_close_path=False, duration=3))
        self.forward(0.5)
        updateWindingGraphItems()
        self.play(
            self.camera.anim.points.shift((coord_distance / 2, 0, 0)),
            Create(i_lineToCenter),
            FadeIn(i_coefPoint),
            Transform(i_coord, i_coord2, hide_src=False),
        )
        self.forward(0.5)
        self.play(Write(Group(*i_nLabel[:-1], i_nValue)), duration=0.5)
        self.forward(0.5)

        # self.play(Write(i_coefFormula, duration=0.75), FadeIn(i_innerProductNote))
        # self.play(
        # FadeIn(i_nValue),
        # FadeIn(i_coefValue),
        # FadeIn(i_values),
        # )

        def animateCoefChange(n_start, n_end):
            self.play(
                GroupUpdater(i_windingGraphItems, createUpdaterFn(n_start, n_end)),
                ItemUpdater(i_nValue, createNValueUpdaterFn(n_start, n_end)),
                # ItemUpdater(
                #     i_coefValue, createCoefValueUpdaterFn(n_start, n_end), duration=3
                # ),
                duration=5,
            )

        for i in range(1, 5):
            animateCoefChange(-i + 1, i)
            self.forward(0.5)
            self.play(
                Transform(i_lineToCenter, i_vecs[i * 2 - 1], hide_src=False),
                Transform(i_coefPoint, i_points[i * 2 - 1], hide_src=False),
                Transform(i_circToCenter, i_circs[i * 2 - 1], hide_src=False),
                duration=2,
            )
            self.play(FadeOut(i_circs[i * 2 - 1], duration=0.5))
            self.forward(0.5)
            animateCoefChange(i, -i)
            self.forward(0.5)
            self.play(
                Transform(i_lineToCenter, i_vecs[i * 2], hide_src=False),
                Transform(i_coefPoint, i_points[i * 2], hide_src=False),
                Transform(i_circToCenter, i_circs[i * 2], hide_src=False),
                duration=2,
            )
            self.play(FadeOut(i_circs[i * 2], duration=0.5))
            self.forward(0.5)
        animateCoefChange(-i, 0)
        self.forward(1)
        self.prepare(FadeOut(i_nLabel), FadeOut(i_nValue))
        self.prepare(
            *(
                FadeIn(item, duration=np.abs(coef))
                for item, coef in zip(
                    i_vecs[i * 2 + 1 :], coefs_alternating[i * 2 + 1 :]
                )
            ),
            duration=3,
            lag_ratio=0.25,
        )
        self.prepare(
            *(
                FadeIn(item, duration=np.abs(coef))
                for item, coef in zip(
                    i_points[i * 2 + 1 :], coefs_alternating[i * 2 + 1 :]
                )
            ),
            duration=3,
            lag_ratio=0.25,
        )
        self.prepare(
            *(
                GrowFromCenter(item, duration=np.sqrt(np.abs(coef)))
                for item, coef in zip(i_circs, coefs_alternating)
            ),
            duration=3,
            lag_ratio=0.125,
        )
        self.forward(3)
        self.forward(0.5)

        i_figure2 = i_figure.copy().points.shift(RIGHT * coord_distance).r
        self.play(
            GroupUpdater(
                i_graphingItems,
                lambda item, params: updateVectors(params.alpha, item),
            ),
            Create(i_figure2, auto_close_path=False),
            duration=fourier_anim_time,
            rate_func=animEasing,
        )
        self.forward(0.5)
        self.play(Transform(i_figure2, i_figure, hide_src=False), duration=2)
        self.play(i_figure.anim.glow.set(color=BILI_PINK, alpha=0.5, size=0.5))
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
