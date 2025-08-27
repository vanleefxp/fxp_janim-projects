from janim.imports import *
from common import *
from fourier_figure import FourierFigure

with reloads():
    from common import *
    from fourier_figure import FourierFigure


class TL_RotatingCircles(Timeline):
    def construct(self):
        max_n = 50
        n_displayed = 5

        figs = tuple(
            FourierFigure(DIR / f"assets/data/fourier-coefs/{name}.npz")[:max_n]
            for name in ("like", "coin", "fav", "cxk")
        )
        coef_lists = tuple(fftalt(fig.coefs) for fig in figs)

        circ_x_distance = 2
        circ_y_distance = 1.5
        rotating_circ_radius = 1.25
        radius = 2
        coord_distance = 6
        axis_pad = 0.75
        self.camera.points.shift(LEFT * (coord_distance / 2))

        axis_range = (-radius - axis_pad, radius + axis_pad, radius)
        i_coord = Axes(
            axis_range, axis_range, x_axis_config=axis_cfg, y_axis_config=axis_cfg
        )

        # 建立系数模长和辐角的 ValueTracker
        i_coefsAbs = Group(*(ValueTracker(1.0) for _ in range(2 * max_n - 1)))
        i_coefsAngle = Group(*(ValueTracker(0.0) for _ in range(2 * max_n - 1)))

        # 建立旋转圆相关对象
        n_circs = (n_displayed - 1) * 2
        i_rotatingCircs = Group(*(Circle(stroke_radius=0.01) for _ in range(n_circs)))
        i_pointsOnCirc = Group(*(Dot(radius=0.05) for _ in range(n_circs)))
        i_linesToCenter = Group(*(Line(stroke_radius=0.01) for _ in range(n_circs)))
        i_linesToCenterInit = Group(
            *(Line(stroke_radius=0.01, alpha=0.5) for _ in range(n_circs))
        )

        # 建立傅里叶级数动画相关对象
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
                    1,
                    stroke_radius=interpolate(0.01, 0.005, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i in alternatingSignedInts(max_n)
            )
        )
        i_points = Group[Dot](
            *(
                Dot(radius=interpolate(0.04, 0.01, 1 - np.exp(-np.abs(i) / 5)))
                for i in alternatingSignedInts(max_n)
            )
        )
        i_figure = ParametricCurve(
            lambda t: complex2point(np.exp(2j * np.pi * t)) * radius,
            (0, 1, 0.01),
            color=BILI_PINK,
        )
        i_circs[0].set(stroke_alpha=0)

        i_graphingItems = Group(
            i_coefsAbs,
            i_coefsAngle,
            i_rotatingCircs,
            i_pointsOnCirc,
            i_linesToCenter,
            i_linesToCenterInit,
            i_vecs,
            i_circs,
            i_points,
            i_figure,
        )

        def updateGraphingItems(t: float, item: Group[VItem] = i_graphingItems) -> None:
            (
                i_coefsAbs,
                i_coefsAngle,
                i_rotatingCircs,
                i_pointsOnCirc,
                i_linesToCenter,
                i_linesToCenterInit,
                i_vecs,
                i_circs,
                i_points,
                i_figure,
            ) = item

            # 由于 ValueTracker 可能正在更新中，傅里叶系数求和需要手动计算
            sl = slice(1, n_displayed * 2 - 1)
            coefs_abs = np.array([item.data.get() for item in i_coefsAbs])
            coefs_angle = np.array([item.data.get() for item in i_coefsAngle])
            coefs = coefs_abs * np.exp(1j * coefs_angle)
            n_values = fftalt(np.fft.ifftshift(np.arange(-max_n + 1, max_n)))
            components = coefs * np.exp(2j * np.pi * n_values * t)
            components_cumsum = np.cumsum(components)
            i_figure.become(
                ParametricCurve(
                    lambda t: complex2point(
                        np.sum(coefs * np.exp(2j * np.pi * n_values * t))
                    )
                    * radius,
                    (0, 1, 0.01),
                    color=BILI_PINK,
                )
            )
            if t < 1:
                i_figure.points.pointwise_become_partial_reduced(i_figure, 0, t)

            # 更新傅里叶级数图像
            for (pc_prev, pc_next), coef_abs, i_vec, i_circ, i_dot in zip(
                it.pairwise(it.chain((0j,), components_cumsum)),
                coefs_abs,
                i_vecs,
                i_circs,
                i_points,
            ):
                p_prev = complex2point(pc_prev * radius)
                p_next = complex2point(pc_next * radius)
                i_vec.points.set_start_and_end(p_prev, p_next)
                i_circ.become(
                    Circle(coef_abs * radius, stroke_radius=i_circ.radius.get()[0])
                    .points.shift(p_prev)
                    .r
                )
                i_dot.points.move_to(p_next)

            # 更新旋转圆
            for (
                n,
                coef_abs,
                coef,
                i_circ,
                i_pointOnRotatingCirc,
                i_lineToCenter,
                i_lineToCenterInit,
            ) in zip(
                it.islice(alternatingSignedInts(), 1, None),
                coefs_abs[sl],
                coefs[sl],
                i_rotatingCircs,
                i_pointsOnCirc,
                i_linesToCenter,
                i_linesToCenterInit,
            ):
                component = coef * np.exp(2j * np.pi * n * t)
                pointOnRotatingCirc = complex2point(component * rotating_circ_radius)
                pointOnRotatingCircInit = complex2point(coef * rotating_circ_radius)
                rotatingCircCenter = np.array(
                    (0, -circ_y_distance * (n - 1), 0)
                    if n > 0
                    else (circ_x_distance, -circ_y_distance * (abs(n) - 1), 0)
                ) + (
                    -circ_x_distance / 2 - coord_distance,
                    (n_displayed / 2 - 1) * circ_y_distance,
                    0,
                )

                i_circ.become(
                    Circle(
                        radius=coef_abs * rotating_circ_radius,
                        stroke_radius=i_circ.radius.get()[0],
                    )
                    .points.shift(rotatingCircCenter)
                    .r
                )
                i_pointOnRotatingCirc.points.move_to(
                    pointOnRotatingCirc + rotatingCircCenter
                )
                i_lineToCenter.points.set_start_and_end(
                    rotatingCircCenter, pointOnRotatingCirc + rotatingCircCenter
                )
                i_lineToCenterInit.points.set_start_and_end(
                    rotatingCircCenter, pointOnRotatingCircInit + rotatingCircCenter
                )

        def setCoefs(new_coefs):
            new_coefs_abs = np.abs(new_coefs)
            new_coefs_angle = np.angle(new_coefs)

            for item, coef_abs in zip(i_coefsAbs, new_coefs_abs):
                item.data.set(coef_abs)
            for item, coef_angle in zip(i_coefsAngle, new_coefs_angle):
                item.data.set(coef_angle)

        def getCoefChangeAnim(new_coefs):
            new_coefs_abs = np.abs(new_coefs)
            new_coefs_angle = np.angle(new_coefs)

            return AnimGroup(
                *(
                    item.anim.data.set(coef_abs)
                    for item, coef_abs in zip(i_coefsAbs, new_coefs_abs)
                ),
                *(
                    item.anim.data.set(coef_angle)
                    for item, coef_angle in zip(i_coefsAngle, new_coefs_angle)
                ),
            )

        setCoefs(coef_lists[0])
        updateGraphingItems(0)
        self.forward(0.25)
        ag1 = []
        ag2 = []
        ag3 = []
        ag4 = []
        for i_circ, i_line, i_lineInit, i_point, coef in zip(
            i_rotatingCircs,
            i_linesToCenter,
            i_linesToCenterInit,
            i_pointsOnCirc,
            coef_lists[0][1:],
        ):
            duration = np.sqrt(np.abs(coef))
            ag1.append(GrowFromCenter(i_circ, duration=duration))
            ag2.append(Create(i_line, duration=duration))
            ag3.append(Create(i_lineInit, duration=duration))
            ag4.append(
                GrowFromPoint(i_point, i_circ.points.self_box.center, duration=duration)
            )
        lag_ratio = 0.125
        ag1 = AnimGroup(*ag1, lag_ratio=lag_ratio)
        ag2 = AnimGroup(*ag2, lag_ratio=lag_ratio)
        ag3 = AnimGroup(*ag3, lag_ratio=lag_ratio)
        ag4 = AnimGroup(*ag4, lag_ratio=lag_ratio)
        self.play(ag1, ag2, ag3, ag4, duration=2)
        self.forward(0.5)
        self.play(FadeIn(i_coord))
        self.forward(0.5)

        lag_ratio = 0.5
        ag1 = AnimGroup(
            *(
                Transform(item1, item2, hide_src=False)
                for item1, item2 in zip(
                    i_rotatingCircs, i_circs[1 : 1 + len(i_rotatingCircs)]
                )
            ),
            lag_ratio=lag_ratio,
        )
        ag2 = AnimGroup(
            *(
                Transform(item1, item2, hide_src=False)
                for item1, item2 in zip(
                    i_linesToCenter, i_vecs[1 : 1 + len(i_linesToCenter)]
                )
            ),
            lag_ratio=lag_ratio,
        )
        ag3 = AnimGroup(
            *(
                Transform(item1, item2, hide_src=False)
                for item1, item2 in zip(
                    i_pointsOnCirc, i_points[1 : 1 + len(i_pointsOnCirc)]
                )
            ),
            lag_ratio=lag_ratio,
        )
        self.play(
            FadeIn(i_points[0]),
            FadeIn(i_circs[0]),
            FadeIn(i_vecs[0]),
            ag1,
            ag2,
            ag3,
            duration=8,
        )
        self.forward(1)

        lag_ratio = 1 / 8
        sl = slice(2 * n_displayed - 1, None)
        ag1 = []
        ag2 = []
        ag3 = []
        for i_circ, i_line, i_point, coef in zip(
            i_circs[sl],
            i_vecs[sl],
            i_points[sl],
            coef_lists[0][sl],
        ):
            duration = np.sqrt(np.abs(coef))
            ag1.append(GrowFromCenter(i_circ, duration=duration))
            ag2.append(Create(i_line, duration=duration))
            ag3.append(
                GrowFromPoint(i_point, i_circ.points.self_box.center, duration=duration)
            )
        ag1 = AnimGroup(*ag1, lag_ratio=lag_ratio)
        ag2 = AnimGroup(*ag2, lag_ratio=lag_ratio)
        ag3 = AnimGroup(*ag3, lag_ratio=lag_ratio)

        self.play(ag1, ag2, ag3, duration=2)
        self.forward(0.5)

        t0 = self.current_time
        period = 6
        n_periods = 2
        for i, coefs in enumerate(coef_lists[1:], 1):
            self.prepare(
                getCoefChangeAnim(coefs), at=period * n_periods * i, duration=3
            )
        self.play(
            GroupUpdater(
                i_graphingItems,
                lambda item, params: updateGraphingItems(
                    (params.global_t - t0) / period, item
                ),
                rate_func=linear,
            ),
            duration=len(figs) * period * n_periods,
        )
        self.forward(2)


class TL_Test(Timeline):
    def construct(self):
        i_value = ValueTracker(1 + 0j)
        i_circ = Circle(stroke_radius=0.01)
        i_pointOnCirc = Dot(radius=0.04)
        i_lineToCenter = Line(stroke_radius=0.01)
        i_graphingItems = Group(i_value, i_circ, i_pointOnCirc, i_lineToCenter)
        t0 = self.current_time

        def updateGraphingItems(t, item=i_graphingItems):
            i_value, i_circ, i_pointOnCirc, i_lineToCenter = item
            coef = i_value.data.get()
            pointOnCirc = complex2point(coef * np.exp(2j * np.pi * t))
            i_circ.become(
                Circle(radius=np.abs(coef), stroke_radius=i_circ.radius.get()[0])
            )
            i_pointOnCirc.points.move_to(pointOnCirc)
            i_lineToCenter.points.set_start_and_end(ORIGIN, pointOnCirc)

        def updaterFn(
            item: Group[ValueTracker | Circle], params: UpdaterParams
        ) -> None:
            updateGraphingItems(params.global_t - t0, item)

        self.prepare(i_value.anim.data.set(1 + 1j), at=1)
        self.prepare(i_value.anim.data.set(0.5 - 0.5j), at=3)
        self.play(
            GroupUpdater(i_graphingItems, updaterFn, rate_func=linear), duration=10
        )

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
