from janim.imports import *
from frozendict import frozendict

with reloads():
    from common import *


class TL_Spiral(Timeline):
    CONFIG = config

    def construct(self):
        def getSpiralFn(r=1, v=1):
            def spiral(t: float) -> Vect:
                return np.array(
                    (
                        np.cos(2 * np.pi * t) * r,
                        np.sin(2 * np.pi * t) * r,
                        v * r * t,
                    )
                )

            return spiral

        def getFigSpiralFn(fig: FourierFigure, r=1, v=1):
            def spiral(t: float) -> Vect:
                return complex2point(fig(t % 1)) + OUT * (v * r * t)

            return spiral

        def splitRealAndImag(
            fn: Callable[[float], Vect],
        ) -> tuple[Callable[[float], Vect], Callable[[float], Vect]]:
            def real(t: float) -> Vect:
                x, y, z = fn(t)
                return np.array((x, 0, z))

            def imag(t: float) -> Vect:
                x, y, z = fn(t)
                return np.array((0, y, z))

            return real, imag

        t_max = 5 + 1 / 6
        t_unit = 2.5
        smoothTime = 0.5
        animEasing = smoothBoth(smoothTime * 2 / t_max)
        radius = 1.25
        wavelength = 3
        projectionDist = 3
        velocity = 1.6
        axis_pad = 0.5
        axis_range = (-radius - axis_pad, radius + axis_pad, radius)
        axis_cfg = frozendict(include_tip=True, tip_config=arrowCfg, tick_size=0.05)
        i_coord = Axes(
            x_range=axis_range,
            y_range=axis_range,
            x_axis_config=axis_cfg,
            y_axis_config=axis_cfg,
        )
        i_coordImag = (
            Axes(
                x_range=(-axis_pad, axis_pad + wavelength, wavelength / 2),
                y_range=axis_range,
                x_axis_config=axis_cfg,
                y_axis_config=axis_cfg,
            )
            .points.shift(projectionDist * RIGHT)
            .r
        )
        i_coordReal = (
            Axes(
                x_range=axis_range,
                y_range=(-axis_pad, axis_pad + wavelength, wavelength / 2),
                x_axis_config=axis_cfg,
                y_axis_config=axis_cfg,
            )
            .points.shift(projectionDist * UP)
            .r
        )
        i_zAxis = NumberLine(
            x_range=(
                -axis_pad,
                t_max * velocity * radius + axis_pad,
                velocity * radius,
            ),
            **axis_cfg,
        )
        i_zAxis.points.shift(-i_zAxis.n2p(0)).rotate(
            -PI / 2, axis=UP, about_point=ORIGIN
        )
        for i_axis in i_coord.get_axes():
            i_axis.ticks.set(stroke_radius=0.015)
        i_circ = Circle(radius=radius, color=GREEN_SCREEN)
        i_spiralInit = ParametricCurve(
            getSpiralFn(radius, 0), (0, t_max, 0.05), color=GREEN_SCREEN
        )
        i_spiral = ParametricCurve(
            getSpiralFn(radius, velocity), (0, t_max, 0.05), color=GREEN_SCREEN
        )

        i_circEq: TypstMath = (
            toCircular(
                TypstMath("up(e)^(2 pi up(i) t)", **textCfg)
                .points.scale(0.95)
                .next_to(i_circ, UP, buff=0.1)
                .rotate(-PI / 2, about_point=ORIGIN)
                .r
            )
            .points.rotate(PI / 3 * 2, about_point=ORIGIN)
            .r
        )
        i_circEqLarge: TypstMath = (
            toCircular(
                TypstMath("up(e)^(2 pi up(i) t)", **textCfg)
                .points.scale(1.125)
                .next_to(i_circ, UP, buff=0.1)
                .rotate(-PI / 2, about_point=ORIGIN)
                .r
            )
            .points.rotate(PI / 3 * 2, about_point=ORIGIN)
            .r
        )
        i_sineEq = (
            TypstMath("sin(2 pi t)", **textCfg)
            .points.next_to(
                (projectionDist, radius + axis_pad, 0),
                UR,
                buff=0.1,
            )
            .r
        )
        i_cosineEq = (
            TypstMath("cos(2 pi t)", **textCfg)
            .points.rotate(-PI / 2)
            .next_to(
                (radius + axis_pad, projectionDist, 0),
                UR,
                buff=0.1,
            )
            .r
        )

        self.play(Create(i_coord))
        self.play(Create(i_circ, auto_close_path=False))
        self.play(Write(i_circEq))
        self.hide(i_circ)
        self.show(i_spiralInit)
        self.forward(1)

        # 存储 2D 原始视图
        i_cameraStored = self.camera.store()

        # 缩小视图，展示正弦和余弦
        self.play(
            self.camera.anim.points.shift((2, 2, 3)),
            *(FadeIn(item) for item in (i_coordImag, i_coordReal)),
            Transform(i_circEq, i_circEqLarge),
            duration=2,
        )
        self.forward(1)

        i_pointOnCirc = Dot(
            radius=0.06,
            glow_color=GREEN_SCREEN,
            glow_size=0.5,
            glow_alpha=0.5,
            fill_alpha=0.75,
            depth=-1,
        )
        i_pointOnSine = i_pointOnCirc.copy()
        i_pointOnCosine = i_pointOnCirc.copy()
        i_sineTem = (
            FunctionGraph(
                lambda x: np.sin(2 * np.pi * x / wavelength) * radius,
                (0, wavelength, wavelength / 200),
                color=GREEN_SCREEN,
            )
            .points.shift(i_coordImag.get_origin())
            .r
        )
        i_sine = i_sineTem.copy().set(alpha=0.5)
        i_sinePartial = i_sineTem.copy()
        i_cosineTem = (
            FunctionGraph(
                lambda x: -np.cos(2 * np.pi * x / wavelength) * radius,
                (0, wavelength, wavelength / 200),
                color=GREEN_SCREEN,
            )
            .points.rotate(PI / 2, about_point=ORIGIN)
            .shift(i_coordReal.get_origin())
            .r
        )
        i_cosinePartial = i_cosineTem.copy()
        i_cosine = i_cosineTem.copy().set(alpha=0.5)
        i_lineToSine = Line(stroke_radius=0.01)
        i_lineToCosine = i_lineToSine.copy()
        i_lineToCosineAxis = i_lineToSine.copy()
        i_lineToSineAxis = i_lineToSine.copy()
        i_lineToCenter = i_lineToSine.copy()
        i_graphingObjects = Group(
            i_pointOnCirc,
            i_pointOnSine,
            i_pointOnCosine,
            i_lineToCenter,
            i_lineToSine,
            i_lineToCosine,
            i_lineToSineAxis,
            i_lineToCosineAxis,
            i_sine,
            i_cosine,
            i_sinePartial,
            i_cosinePartial,
        )

        def updateGraphingItems(
            item: Group[VItem] = i_graphingObjects,
            t=0,
        ) -> None:
            (
                i_pointOnCirc,
                i_pointOnSine,
                i_pointOnCosine,
                i_lineToCenter,
                i_lineToSine,
                i_lineToCosine,
                i_lineToSineAxis,
                i_lineToCosineAxis,
                i_sine,
                i_cosine,
                i_sinePartial,
                i_cosinePartial,
            ) = item
            t_period = t % 1
            dist = projectionDist + wavelength * t_period
            x = np.cos(2 * np.pi * t)
            y = np.sin(2 * np.pi * t)

            pointOnCirc = np.array((x * radius, y * radius, 0))
            pointOnSine = np.array((dist, y * radius, 0))
            pointOnCosine = np.array((x * radius, dist, 0))
            pointOnSineAxis = RIGHT * dist
            pointOnCosineAxis = UP * dist

            i_pointOnCirc.points.move_to(pointOnCirc)
            i_pointOnSine.points.move_to(pointOnSine)
            i_pointOnCosine.points.move_to(pointOnCosine)

            i_lineToCenter.points.set_start_and_end(pointOnCirc, ORIGIN)
            i_lineToSine.points.set_start_and_end(pointOnCirc, pointOnSine)
            i_lineToCosine.points.set_start_and_end(pointOnCirc, pointOnCosine)
            i_lineToSineAxis.points.set_start_and_end(pointOnSine, pointOnSineAxis)
            i_lineToCosineAxis.points.set_start_and_end(
                pointOnCosine, pointOnCosineAxis
            )
            i_sinePartial.points.pointwise_become_partial_reduced(i_sine, 0, t_period)
            i_cosinePartial.points.pointwise_become_partial_reduced(
                i_cosine, 0, t_period
            )

        def updaterFn(item: Group[Dot | Line], params: UpdaterParams):
            t = params.alpha * t_max
            updateGraphingItems(item, t)

        updateGraphingItems()
        self.play(
            *(
                FadeIn(item)
                for item in (
                    i_pointOnSine,
                    i_pointOnCosine,
                    i_pointOnCirc,
                    i_sinePartial,
                    i_cosinePartial,
                )
            ),
            *(
                Create(item)
                for item in (
                    i_lineToCenter,
                    i_lineToSine,
                    i_lineToCosine,
                    i_lineToSineAxis,
                    i_lineToCosineAxis,
                )
            ),
        )
        self.prepare(
            *(FadeIn(item) for item in (i_sine, i_cosine, i_cosineEq, i_sineEq)),
            duration=1,
        )
        self.play(
            GroupUpdater(
                i_graphingObjects,
                updaterFn,
                duration=t_unit * t_max,
                rate_func=animEasing,
            )
        )
        self.forward(2)

        # 转换到 3D 视图
        cameraZShift = velocity * radius * t_max / 2
        self.play(
            *(
                FadeOut(item, duration=1)
                for item in (
                    i_coordImag,
                    i_coordReal,
                    i_graphingObjects,
                    i_sine,
                    i_cosine,
                    i_circEqLarge,
                    i_cosineEq,
                    i_sineEq,
                )
            ),
            self.camera.anim(duration=2)
            .restore(i_cameraStored)
            .points.shift((-4, 3, 2) + OUT * cameraZShift)
            .rotate(-DEGREES * 43, axis=RIGHT)
            .rotate(-DEGREES * 75, axis=UP),
        )

        # 将圆拉开成螺旋线
        self.prepare(
            *(FadeIn(item) for item in (i_zAxis.ticks, i_zAxis.tip)),
            at=3.5,
            duration=0.5,
        )
        self.play(
            Transform(i_spiralInit, i_spiral),
            Create(i_zAxis, root_only=True),
            duration=4,
        )
        self.forward(1)

        projectionDist3d = 5

        def toProjectionDown[I: VItem](item: I, shift=True) -> I:
            item.points.rotate(-PI / 2, about_point=ORIGIN, axis=RIGHT).rotate(
                -PI / 2, about_point=ORIGIN, axis=UP
            )
            if shift:
                item.points.shift(DOWN * projectionDist3d)
            return item

        def toProjectionRight[I: VItem](item: I, shift=True) -> I:
            item.points.rotate(-PI / 2, about_point=ORIGIN, axis=UP)
            if shift:
                item.points.shift(RIGHT * projectionDist3d)
            return item

        bgRectCfg = frozendict(color=WHITE, fill_alpha=0.125, stroke_radius=0.01)
        i_bgRect = Rect(
            (-axis_pad, -radius - axis_pad, 0),
            (t_max * velocity * radius + axis_pad, radius + axis_pad, 0),
            **bgRectCfg,
        )
        i_bgRectFnl = Rect(
            (-axis_pad, -radius - axis_pad, 0),
            (-axis_pad, radius + axis_pad, 0),
            **bgRectCfg,
        )
        i_coordTem = Axes(
            x_axis_config=axis_cfg,
            y_axis_config=axis_cfg,
            x_range=(
                -axis_pad,
                axis_pad + velocity * radius * t_max,
                velocity * radius,
            ),
            y_range=(-axis_pad - radius, axis_pad + radius, radius),
        )
        i_cosineTem = ParametricCurve(
            lambda t: np.array(
                (t * velocity * radius, np.cos(2 * np.pi * t) * radius, 0)
            ),
            (0, t_max, 0.05),
            color=GREEN_SCREEN,
        )
        i_sineTem = ParametricCurve(
            lambda t: np.array(
                (t * velocity * radius, np.sin(2 * np.pi * t) * radius, 0)
            ),
            (0, t_max, 0.05),
            color=GREEN_SCREEN,
        )

        i_cosineProj = toProjectionDown(i_cosineTem.copy(), False)
        i_coordRealProj = toProjectionDown(i_coordTem.copy(), False)
        i_cosine = toProjectionDown(i_cosineTem.copy())
        i_cosinePartial = i_cosine.copy()
        i_cosineFnl = toProjectionDown(
            ParametricCurve(
                lambda t: np.array((0, np.cos(2 * np.pi * t) * radius, 0)),
                (0, t_max, 0.05),
                color=GREEN_SCREEN,
                alpha=0,
            )
        )
        i_realBg = toProjectionDown(i_bgRect.copy())
        i_cosineEqProj = toProjectionDown(
            TypstMath("cos(2 pi t)", **textCfg)
            .points.next_to((0, radius + axis_pad, 0), UR, buff=0.1)
            .r,
            False,
        )
        i_cosineLabel: TypstMath = toProjectionDown(
            TypstMath("cos(2 pi t)", **textCfg)
            .points.scale(1.5)
            .next_to(i_bgRect.points.self_box.get(DL), DR, buff=0.1)
            .r
        )
        i_coordReal = toProjectionDown(i_coordTem.copy())
        # for i_axis in i_coordCosine.get_axes():
        #     i_axis.remove(i_axis.tip)
        i_realBgFnl = toProjectionDown(i_bgRectFnl.copy())

        i_sineProj = toProjectionRight(i_sineTem.copy(), False)
        i_coordImagProj = toProjectionRight(i_coordTem.copy(), False)
        i_sine = toProjectionRight(i_sineTem.copy())
        i_sinePartial = i_sine.copy()
        i_coordImag = toProjectionRight(i_coordTem.copy())
        # for i_axis in i_coordSine.get_axes():
        #     i_axis.remove(i_axis.tip)
        i_sineFnl = toProjectionRight(
            ParametricCurve(
                lambda t: np.array((0, np.sin(2 * np.pi * t) * radius, 0)),
                (0, t_max, 0.05),
                color=GREEN_SCREEN,
                alpha=0,
            )
        )
        i_imagBg = toProjectionRight(i_bgRect.copy())
        i_sineEqProj = toProjectionRight(
            TypstMath("sin(2 pi t)", **textCfg)
            .points.next_to((0, radius + axis_pad, 0), UR, buff=0.1)
            .r,
            False,
        )
        i_sineLabel: TypstMath = toProjectionRight(
            TypstMath("sin(2 pi t)", **textCfg)
            .points.scale(1.5)
            .next_to(i_bgRect.points.self_box.get(UL), UR, buff=0.1)
            .r
        )
        i_imagBgFnl = toProjectionRight(i_bgRectFnl.copy())
        i_spiralPartial = i_spiral.copy()

        # 存储原始 3D 视角
        i_camera3dStored = self.camera.store()

        # 转为侧视图
        self.prepare(FadeIn(i_sineEqProj), duration=2, at=2)
        self.play(
            self.camera.anim.restore(i_cameraStored)
            .points.shift((-2, 0, cameraZShift))
            .rotate(-PI / 2, axis=UP),
            FadeOut(i_coord),
            FadeOut(i_zAxis),
            FadeIn(i_coordImagProj),
            Transform(i_spiral, i_sineProj),
            duration=4,
        )
        i_cameraLeftStored = self.camera.store()
        self.forward(1)

        # 恢复 3D 视角
        self.play(
            self.camera.anim.restore(i_camera3dStored),
            Transform(i_sineProj, i_spiral),
            FadeOut(i_coordImagProj),
            FadeOut(i_sineEqProj),
            FadeIn(i_coord),
            FadeIn(i_zAxis),
            duration=2,
        )
        self.forward(1)

        # 转为顶视图
        self.prepare(FadeIn(i_cosineEqProj, duration=2), at=2)
        self.play(
            self.camera.anim.restore(i_cameraStored)
            .points.shift((0, 2, cameraZShift))
            .rotate(-PI / 2, axis=RIGHT)
            .rotate(-PI / 2, axis=UP),
            FadeOut(i_coord),
            FadeOut(i_zAxis),
            FadeIn(i_coordRealProj),
            Transform(i_spiral, i_cosineProj),
            duration=4,
        )
        i_cameraTopStored = self.camera.store()
        self.forward(1)

        # 恢复 3D 视角
        self.play(
            self.camera.anim.restore(i_camera3dStored),
            Transform(i_cosineProj, i_spiral),
            FadeOut(i_coordRealProj),
            FadeOut(i_cosineEqProj),
            FadeIn(i_coord),
            FadeIn(i_zAxis),
            duration=2,
        )
        self.forward(0.5)

        self.show(i_realBgFnl)
        self.play(Transform(i_realBgFnl, i_realBg))
        self.prepare(FadeIn(i_coordReal), FadeIn(i_cosineLabel))
        self.play(
            Transform(i_spiral, i_cosine, hide_src=False),
            duration=3,
        )
        self.forward(0.5)
        self.show(i_imagBgFnl)
        self.play(Transform(i_imagBgFnl, i_imagBg))
        self.prepare(FadeIn(i_coordImag), FadeIn(i_sineLabel))
        self.play(Transform(i_spiral, i_sine, hide_src=False), duration=3)

        i_pointOnSine.points.rotate(PI / 2, axis=UP)
        i_pointOnCosine.points.rotate(PI / 2, axis=RIGHT)
        i_graphingItems3d = Group(
            i_pointOnCirc,
            i_pointOnSine,
            i_pointOnCosine,
            i_lineToCenter,
            i_lineToSine,
            i_lineToCosine,
            i_lineToSineAxis,
            i_lineToCosineAxis,
            i_sinePartial,
            i_cosinePartial,
            i_spiralPartial,
        )

        def updateGraphingItems3d(
            items: Group[VItem] = i_graphingItems3d,
            t: float = 0,
        ):
            (
                i_pointOnCirc,
                i_pointOnSine,
                i_pointOnCosine,
                i_lineToCenter,
                i_lineToSine,
                i_lineToCosine,
                i_lineToSineAxis,
                i_lineToCosineAxis,
                i_sinePartial,
                i_cosinePartial,
                i_spiralPartial,
            ) = items
            x = np.cos(2 * np.pi * t)
            y = np.sin(2 * np.pi * t)
            z = t * velocity

            pointOnCirc = np.array((x, y, z)) * radius
            pointOnCosine = np.array((x * radius, -projectionDist3d, z * radius))
            pointOnSine = np.array((projectionDist3d, y * radius, z * radius))
            pointOnAxis = np.array((0, 0, z * radius))
            pointOnCosineAxis = np.array((0, -projectionDist3d, z * radius))
            pointOnSineAxis = np.array((projectionDist3d, 0, z * radius))

            i_pointOnCirc.points.move_to(pointOnCirc)
            i_pointOnCosine.points.move_to(pointOnCosine)
            i_pointOnSine.points.move_to(pointOnSine)

            i_lineToCenter.points.set_start_and_end(pointOnCirc, pointOnAxis)
            i_lineToCosine.points.set_start_and_end(pointOnCirc, pointOnCosine)
            i_lineToSine.points.set_start_and_end(pointOnCirc, pointOnSine)
            i_lineToCosineAxis.points.set_start_and_end(
                pointOnCosine, pointOnCosineAxis
            )
            i_lineToSineAxis.points.set_start_and_end(pointOnSine, pointOnSineAxis)

            alpha = t / t_max
            i_sinePartial.points.pointwise_become_partial_reduced(i_sine, 0, alpha)
            i_cosinePartial.points.pointwise_become_partial_reduced(i_cosine, 0, alpha)
            i_spiralPartial.points.pointwise_become_partial_reduced(i_spiral, 0, alpha)

        def updaterFn3d(items: Group[VItem], params: UpdaterParams):
            t = params.alpha * t_max
            updateGraphingItems3d(items, t)

        updateGraphingItems3d()
        self.show(i_sinePartial, i_cosinePartial, i_spiralPartial)
        self.play(
            *(FadeIn(item) for item in (i_pointOnCirc, i_pointOnSine, i_pointOnCosine)),
            *(
                Create(item)
                for item in (
                    i_lineToCenter,
                    i_lineToSine,
                    i_lineToCosine,
                    i_lineToSineAxis,
                    i_lineToCosineAxis,
                )
            ),
            *(item.anim.set(alpha=0.5) for item in (i_sine, i_cosine, i_spiral)),
        )
        self.forward(0.5)
        self.play(
            GroupUpdater(
                i_graphingItems3d,
                updaterFn3d,
                duration=t_unit * t_max,
                rate_func=animEasing,
            )
        )

        for item in (i_sine, i_cosine, i_spiral):
            item.set(alpha=1)
        self.play(FadeOut(i_graphingItems3d))
        self.forward(2)

        # 转回平面视角
        self.play(
            Transform(i_spiral, i_spiralInit, duration=2),
            Transform(i_cosine, i_cosineFnl, duration=2),
            Transform(i_sine, i_sineFnl, duration=2),
            Uncreate(i_zAxis, root_only=True, duration=2),
            *(FadeOut(item, duration=0.5) for item in (i_zAxis.ticks, i_zAxis.tip)),
            *(
                FadeOut(item, duration=1)
                for item in (
                    i_realBg,
                    i_imagBg,
                    i_coordReal,
                    i_coordImag,
                    i_cosineLabel,
                    i_sineLabel,
                )
            ),
            FadeIn(i_circEq, duration=2),
            self.camera.anim(duration=2).restore(i_cameraStored),
        )
        self.hide(i_cosineFnl, i_sineFnl, i_spiralInit)
        self.show(i_circ)
        self.forward(2)

        # 加载傅里叶系数
        coefs = np.load(DIR / "assets/data/fourier-coefs/like.npy")
        max_n = 20
        fig: FourierFigure = FourierFigure(coefs)[:max_n]
        max_n = fig.max_n
        coefs = fig.coefs

        def animateFigureSpiral(fig: FourierFigure = fig):
            i_figure = ParametricCurve(
                lambda t: complex2point(fig(t) * radius), (0, 1, 0.01), color=BILI_PINK
            )
            spiralTRange = (0, t_max, 0.01)
            spiralFn = getFigSpiralFn(fig, radius, velocity)

            def spiralFnInit(t: float) -> Vect:
                return complex2point(fig(t) * radius)

            spiralFnReal, spiralFnImag = splitRealAndImag(spiralFn)
            spiralFnInitReal, spiralFnInitImag = splitRealAndImag(spiralFnInit)

            i_figSpiralInit = ParametricCurve(
                lambda t: complex2point(fig(t) * radius), spiralTRange, color=BILI_PINK
            )
            i_figSpiral = ParametricCurve(spiralFn, spiralTRange, color=BILI_PINK)
            i_figSpiralReal = ParametricCurve(
                spiralFnReal, spiralTRange, color=BILI_PINK
            )
            i_figSpiralImag = ParametricCurve(
                spiralFnImag, spiralTRange, color=BILI_PINK
            )
            i_figSpiralRealInit = (
                ParametricCurve(
                    spiralFnInitReal, spiralTRange, color=BILI_PINK, alpha=0
                )
                .points.shift(DOWN * projectionDist3d)
                .r
            )
            i_figSpiralImagInit = (
                ParametricCurve(
                    spiralFnInitImag, spiralTRange, color=BILI_PINK, alpha=0
                )
                .points.shift(RIGHT * projectionDist3d)
                .r
            )

            i_realEqProj = toProjectionDown(
                Text("Re", **textCfg)
                .points.next_to((0, radius + axis_pad, 0), UR, buff=0.1)
                .r,
                False,
            )
            i_imagEqProj = toProjectionRight(
                Text("Im", **textCfg)
                .points.next_to((0, radius + axis_pad, 0), UR, buff=0.1)
                .r,
                False,
            )
            i_realLabel: Text = toProjectionDown(
                Text("Re", **textCfg)
                .points.scale(1.5)
                .next_to(i_bgRect.points.self_box.get(DL), DR, buff=(0.1, 0.2, 0))
                .r
            )
            i_imagLabel: Text = toProjectionRight(
                Text("Im", **textCfg)
                .points.scale(1.5)
                .next_to(i_bgRect.points.self_box.get(UL), UR, buff=(0.1, 0.2, 0))
                .r
            )

            self.play(Transform(i_circ, i_figure, duration=2), FadeOut(i_circEq))
            self.forward(1)
            self.hide(i_figure)
            self.show(i_figSpiralInit)
            self.play(self.camera.anim.restore(i_camera3dStored))
            self.forward(0.5)

            # 将图形拉开成螺旋线
            self.prepare(
                *(FadeIn(item) for item in (i_zAxis.ticks, i_zAxis.tip)),
                at=3.5,
                duration=0.5,
            )
            self.play(
                Transform(i_figSpiralInit, i_figSpiral),
                Create(i_zAxis, root_only=True),
                duration=4,
            )
            self.forward(0.5)

            # 转为侧视图
            self.play(
                self.camera.anim.restore(i_cameraLeftStored),
                Transform(i_figSpiral, i_figSpiralImag),
                FadeOut(i_coord),
                FadeOut(i_zAxis),
                FadeIn(i_coordImagProj),
                FadeIn(i_imagEqProj),
            )
            self.forward(0.5)

            # 恢复 3D 视角
            self.play(
                self.camera.anim.restore(i_camera3dStored),
                Transform(i_figSpiralImag, i_figSpiral),
                FadeIn(i_coord),
                FadeIn(i_zAxis),
                FadeOut(i_coordImagProj),
                FadeOut(i_imagEqProj),
            )
            self.forward(0.5)

            # 转为顶视图
            self.play(
                self.camera.anim.restore(i_cameraTopStored),
                Transform(i_figSpiral, i_figSpiralReal),
                FadeOut(i_coord),
                FadeOut(i_zAxis),
                FadeIn(i_coordRealProj),
                FadeIn(i_realEqProj),
            )
            self.forward(0.5)

            # 恢复 3D 视角
            self.play(
                self.camera.anim.restore(i_camera3dStored),
                Transform(i_figSpiralReal, i_figSpiral),
                FadeIn(i_coord),
                FadeIn(i_zAxis),
                FadeOut(i_coordRealProj),
                FadeOut(i_realEqProj),
            )
            self.forward(0.5)

            # 投影到下方和右侧的两个平面上
            self.show(i_realBgFnl, i_imagBgFnl)
            self.play(
                Transform(i_realBgFnl, i_realBg), Transform(i_imagBgFnl, i_imagBg)
            )

            i_figSpiralReal.points.shift(DOWN * projectionDist3d)
            i_figSpiralImag.points.shift(RIGHT * projectionDist3d)

            self.prepare(
                *(
                    FadeIn(item)
                    for item in (i_realLabel, i_imagLabel, i_coordReal, i_coordImag)
                )
            )
            self.play(
                Transform(i_figSpiral, i_figSpiralReal, hide_src=False),
                Transform(i_figSpiral, i_figSpiralImag, hide_src=False),
                duration=3,
            )
            self.forward(2)

            # 转回平面视角
            self.play(
                Transform(i_figSpiral, i_figSpiralInit, duration=2),
                Transform(i_figSpiralReal, i_figSpiralRealInit, duration=2),
                Transform(i_figSpiralImag, i_figSpiralImagInit, duration=2),
                Uncreate(i_zAxis, root_only=True, duration=2),
                *(FadeOut(item, duration=0.5) for item in (i_zAxis.ticks, i_zAxis.tip)),
                *(
                    FadeOut(item, duration=1)
                    for item in (
                        i_realBg,
                        i_imagBg,
                        i_coordReal,
                        i_coordImag,
                        i_realLabel,
                        i_imagLabel,
                    )
                ),
                self.camera.anim(duration=2).restore(i_cameraStored),
            )
            self.hide(i_figSpiralRealInit, i_figSpiralImagInit, i_figSpiralInit)
            self.show(i_figure)

        animateFigureSpiral(fig)

        # i_lineToCenter.points.set_start_and_end(RIGHT * radius, ORIGIN)
        # i_pointOnCirc.points.move_to(RIGHT * radius)
        # self.play(Create(i_lineToCenter), FadeIn(i_pointOnCirc))
        # self.play(
        #     *(
        #         Rotate(
        #             item,
        #             TAU,
        #             about_point=ORIGIN,
        #             duration=10,
        #         )
        #         for item in (i_pointOnCirc, i_lineToCenter)
        #     )
        # )

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
