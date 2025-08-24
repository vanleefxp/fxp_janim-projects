from pathlib import Path
from functools import lru_cache

from janim.imports import *
from frozendict import frozendict

with reloads():
    from common import *

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


class TL_AnalyticGeom(Timeline):
    CONFIG = config

    def construct(self):
        ratio = 1 / 3

        i_coord = (
            Axes(
                x_range=(-1.35, 1.35, 1),
                y_range=(-1.35, 1.35, 1),
                x_axis_config=dict(
                    unit_size=2.25,
                    include_tip=True,
                    tip_config=arrowConfig,
                ),
                y_axis_config=dict(
                    unit_size=2.25, include_tip=True, tip_config=arrowConfig
                ),
            )
            .points.shift((-3.25, 0.25, 0))
            .r
        )
        i_circ = (
            Circle(i_coord.x_axis.unit_size, stroke_radius=0.015)
            .points.shift(i_coord.get_origin())
            .r
        )
        i_pM = Dot(i_coord.c2p(ratio, 0), radius=0.06, depth=-1)
        i_pP = Dot(i_coord.c2p(1, 0), radius=0.06, depth=-1)

        geomEqnConfig = frozendict(
            preamble="#set text(size: 9pt)",
            stroke_radius=0.005,
            stroke_color=WHITE,
            stroke_alpha=1,
            depth=-1,
        )
        i_circleEqn = (
            toCircular(
                TypstMath("x^2 + y^2 = 1", **geomEqnConfig)
                .points.next_to(UP * i_coord.x_axis.unit_size, UP, buff=0.1)
                .rotate(-PI / 2, about_point=ORIGIN)
                .r
            )
            .points.rotate(PI * 3 / 4, about_point=ORIGIN)
            .shift(i_coord.get_origin())
            .r
        )
        i_pMText = (
            TypstMath(r"M (lambda, 0)", **geomEqnConfig)
            .points.next_to(i_pM.points.self_box.center, DR, buff=0.075)
            .r
        )
        i_pPText = (
            TypstMath(r"P (1, 0)", **geomEqnConfig)
            .points.next_to(i_pP.points.self_box.center, UR, buff=0.075)
            .r
        )
        # i_circleEqn.points.apply_point_fn(
        #     partial(wrap_circular, about_point=i_coord.get_origin())
        # )
        i_problem = (
            TypstText(
                r"/ 问题: 已知 $P A$, $P B$ 斜率分别为 $k_1$, $k_2$, \ 求证 $k_1 k_2$ 为定值。",
                **geomEqnConfig,
            )
            .points.move_to((3.25, 2, 0))
            .r
        )
        i_math1 = (
            TypstMath(
                r"""
                (m y + lambda)^2 + y^2 - 1 &= 0 \
                (m^2 + 1) y^2 + 2 m lambda dot y + (lambda^2 - 1) &= 0 \
                """,
                **geomEqnConfig,
            )
            .points.move_to((3.25, 0.25, 0))
            .r
        )
        i_math2 = (
            TypstMath(
                """
                y_1 + y_2 = -(2 m lambda)/(m^2 + 1) #h(2em)
                y_1 y_2 = (lambda^2 - 1) / (m^2 + 1)
                """,
                **geomEqnConfig,
            )
            .points.next_to(i_math1, DOWN, buff=0.75)
            .r
        )
        i_math3 = (
            TypstMath(
                r"""
                k_1 dot k_2 &= (y_1 y_2) / ((x_1 - 1) (x_2 - 1)) = (y_1 y_2) / ((m y_1 + lambda - 1) (n y_2 + lambda - 1)) \
                &= (y_1 y_2) / (m^2 dot y_1 y_2 + m (lambda - 1) dot (y_1 + y_2) + (lambda - 1)^2) \
                &= (lambda^2 - 1) / (m^2 (lambda^2 - 1) - 2 m^2 lambda (lambda - 1) + (lambda - 1)^2 (m^2 + 1)) \
                &= (lambda + 1) / (m^2 (lambda + 1) - 2 m^2 lambda + (lambda - 1) (m^2 + 1))
                = - (1 + lambda) / (1 - lambda)
                """,
                **geomEqnConfig,
            )
            .points.scale(0.75)
            .move_to((3.25, -1, 0))
            .r
        )

        @lru_cache
        def getIntersections(angle, ratio=ratio):
            m = np.tan(angle)
            a = m * m + 1
            b = 2 * m * ratio
            c = ratio * ratio - 1
            y1 = (-b + np.sqrt(b * b - 4 * a * c)) / a / 2
            y2 = (-b - np.sqrt(b * b - 4 * a * c)) / a / 2
            x1 = m * y1 + ratio
            x2 = m * y2 + ratio

            return np.array(((x1, y1), (x2, y2)))

        startAngle = 15 * DEGREES
        endAngle = -15 * DEGREES
        p1, p2 = getIntersections(startAngle)
        i_lAB = Line(i_coord.c2p(*p1), i_coord.c2p(*p2), buff=-0.5, stroke_color=GREEN)
        i_lPA = Line(i_coord.c2p(1, 0), i_coord.c2p(*p1), buff=-0.5, stroke_color=RED)
        i_lPB = Line(i_coord.c2p(1, 0), i_coord.c2p(*p2), buff=-0.5, stroke_color=RED)
        i_pA = Dot(i_coord.c2p(*p1), radius=0.06, depth=-1)
        i_pB = Dot(i_coord.c2p(*p2), radius=0.06, depth=-1)
        i_pAText = (
            TypstMath("A (x_1, y_1)", **geomEqnConfig)
            .points.next_to(i_pA.points.self_box.center, UR, buff=0.075)
            .r
        )
        i_pBText = (
            TypstMath("B (x_2, y_2)", **geomEqnConfig)
            .points.next_to(i_pB.points.self_box.center, DR, buff=0.075)
            .r
        )

        def getLineRotationAnimation(startAngle, endAngle, **kwargs):
            def lABUpdaterFn(i_line: Line, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                p1, p2 = getIntersections(angle)
                i_line.points.set_start_and_end(i_coord.c2p(*p1), i_coord.c2p(*p2))

            def lPAUpdaterFn(i_line: Line, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                p1, _ = getIntersections(angle)
                i_line.points.set_start_and_end(i_coord.c2p(1, 0), i_coord.c2p(*p1))

            def lPBUpdaterFn(i_line: Line, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                _, p2 = getIntersections(angle)
                i_line.points.set_start_and_end(i_coord.c2p(1, 0), i_coord.c2p(*p2))

            def p1UpdaterFn(i_p1: Dot, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                p1, _ = getIntersections(angle)
                i_p1.points.move_to(i_coord.c2p(*p1))

            def p1TextUpdaterFn(i_p1Text: TypstMath, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                p1, _ = getIntersections(angle)
                i_p1Text.points.next_to(i_coord.c2p(*p1), UR, buff=0.075)

            def p2UpdaterFn(i_p2: Dot, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                _, p2 = getIntersections(angle)
                i_p2.points.move_to(i_coord.c2p(*p2))

            def p2TextUpdaterFn(i_p2Text: TypstMath, params: UpdaterParams):
                angle = interpolate(startAngle, endAngle, params.alpha)
                _, p2 = getIntersections(angle)
                i_p2Text.points.next_to(i_coord.c2p(*p2), DR, buff=0.075)

            return AnimGroup(
                GroupUpdater(i_lAB, lABUpdaterFn),
                GroupUpdater(i_lPA, lPAUpdaterFn),
                GroupUpdater(i_lPB, lPBUpdaterFn),
                GroupUpdater(i_pA, p1UpdaterFn),
                GroupUpdater(i_pB, p2UpdaterFn),
                GroupUpdater(i_pAText, p1TextUpdaterFn),
                GroupUpdater(i_pBText, p2TextUpdaterFn),
                **kwargs,
            )

        self.play(
            Create(i_coord),
            Create(i_circ, auto_close_path=False),
            lag_ratio=0.25,
            duration=1,
        )
        self.play(*(FadeIn(i_) for i_ in (i_pM, i_pP)), duration=0.5)
        self.play(
            AnimGroup(Create(i_lAB), FadeIn(i_pA), FadeIn(i_pB)),
            AnimGroup(Create(i_lPA), Create(i_lPB)),
            lag_ratio=0.5,
            duration=1,
        )

        self.forward(0.5)
        self.play(Write(i_circleEqn), duration=0.5)
        self.play(
            Write(i_pMText),
            Write(i_pPText),
            Write(i_pAText),
            Write(i_pBText),
            duration=1.5,
            lag_ratio=0.25,
        )
        self.forward(0.5)

        self.play(Write(i_problem))

        self.play(getLineRotationAnimation(startAngle, endAngle), duration=1)
        self.play(getLineRotationAnimation(endAngle, startAngle), duration=1)
        self.forward(0.5)

        i_lineEqn = (
            TypstMath("x = m y + lambda", **geomEqnConfig)
            .points.next_to(ORIGIN, UP, buff=0.15)
            .rotate(-startAngle + PI / 2, about_point=ORIGIN)
            .shift(i_coord.c2p(*((p1 + p2) / 2)))
            .r
        )

        self.play(Write(i_lineEqn))
        self.play(Write(i_math1), Write(i_math2), lag_ratio=0.75)
        self.forward(0.5)
        self.play(
            AnimGroup(
                FadeOut(i_math1, duration=0.5),
                i_problem.anim.points.shift(UP * 0.375),
                i_math2.anim.points.shift(UP * 2.375),
            ),
            Write(i_math3, duration=3),
            lag_ratio=0.75,
        )
        i_result = i_math3[149:]
        self.play(
            ShowPassingFlashAround(i_result, time_width=3, duration=2),
            i_result.anim.set(glow_alpha=0.25, glow_size=0.25, glow_color=YELLOW),
            lag_ratio=0.75,
        )
        self.forward(2)


class TL_Test(Timeline):
    def construct(self):
        i_point = Dot(ORIGIN)
        i_line = Line((1, -1, 0), (1, 1, 0))
        i_line2 = i_line.copy()
        self.show(i_line, i_line2, i_point)
        self.play(i_line2.anim.points.insert_n_curves(10).apply_point_fn(wrap_circular))
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
