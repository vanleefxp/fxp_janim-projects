import random
from pathlib import Path

from janim.imports import *
import numpy as np
from fantazia.synth.waveform import analyzeSoundFile

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


class TL_PeriodicFunctions(Timeline):
    def construct(self):
        random.seed(42)
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius
        CORNER_TL = (-rx, ry, 0)
        CORNER_TR = (rx, ry, 0)
        CORNER_BL = (-rx, -ry, 0)
        CORNER_BR = (rx, -ry, 0)
        corners = (CORNER_TL, CORNER_TR, CORNER_BL, CORNER_BR)

        instrumentWaveforms = [
            analyzeSoundFile(file, sampleTime=0.5)[1]
            for file in (DIR / "../assets/sound/instrument").glob("*.wav")
        ]

        i_sine = FunctionGraph(np.sin, (0, TAU, 0.1))
        i_square = Polyline(
            (0, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, -1, 0),
            (2, -1, 0),
            (2, 0, 0),
        )
        i_triangle = Polyline(
            (0, 0, 0),
            (0.5, 1, 0),
            (1.5, -1, 0),
            (2, 0, 0),
        )
        i_sawtooth = Polyline(
            (0, 0, 0),
            (0, 1, 0),
            (2, -1, 0),
            (2, 0, 0),
        )
        i_circular = VItem(
            *PathBuilder()
            .move_to(ORIGIN)
            .arc_to((2, 0, 0), -PI)
            .arc_to((4, 0, 0), PI)
            .get()
        )

        i_waveformTems = Group(
            i_sine,
            i_square,
            i_triangle,
            i_sawtooth,
            i_circular,
            *(
                FunctionGraph(waveform, x_range=(0, 1, 0.01))
                for waveform in instrumentWaveforms
            ),
        )
        i_waveformTems.set(
            stroke_color=GREEN_SCREEN,
            stroke_alpha=0.75,
            glow_color=GREEN_SCREEN,
            glow_size=0.25,
        )
        for item in i_waveformTems:
            item.points.set_size(1.5, 0.75)

        n_rows, n_cols = 6, 7
        i_waveforms = (
            Group(
                *(random.choice(i_waveformTems).copy() for _ in range(n_rows * n_cols))
            )
            .points.arrange_in_grid(n_rows, n_cols, h_buff=0.4, v_buff=0.3)
            .to_center()
            .shift(UP * 0.4)
            .r
        )

        # self.show(i_waveforms)
        ag = [GrowFromPoint(item, random.choice(corners)) for item in i_waveforms]
        random.shuffle(ag)
        self.play(AnimGroup(*ag, lag_ratio=0.01, duration=3))
        self.forward(1)
        ag = [
            item.anim(duration=np.random.uniform(0.5, 2)).set(
                glow_alpha=0.5, stroke_alpha=1
            )
            for item in i_waveforms
        ]
        random.shuffle(ag)
        self.play(AnimGroup(*ag, lag_ratio=0.25, duration=6))
        self.forward(2)
        ag = [FadeOutToPoint(item, ORIGIN) for item in i_waveforms]
        random.shuffle(ag)
        self.play(AnimGroup(*ag, lag_ratio=0.025, duration=2))
        self.forward(2)


class TL_FourierDotProductBasis(Timeline):
    def construct(self):
        def createDiagram(n=1, cos=False):
            i_rect = Rect((0, -1, 0), (TAU, 1, 0), stroke_radius=0, stroke_alpha=0)
            i_graph = FunctionGraph(
                (lambda x: np.cos(n * x)) if cos else (lambda x: np.sin(n * x)),
                x_range=(0, TAU, 0.1),
                stroke_color=GREEN_SCREEN,
                stroke_alpha=0.75,
                glow_color=GREEN_SCREEN,
                glow_size=0.25,
            )
            i_rect.add(i_graph)
            i_rect.points.set_size(1.5, 0.75)
            return i_rect

        max_n = 7
        i_cosDiagrams = Group(*(createDiagram(n, cos=True) for n in range(max_n)))
        i_sinDiagrams = Group(*(createDiagram(n, cos=False) for n in range(max_n)))
        i_sinDiagrams[0][0].set(stroke_alpha=0.375)
        i_diagrams = (
            Group(*i_cosDiagrams, *i_sinDiagrams)
            .points.arrange_in_grid(n_rows=2, n_cols=max_n, h_buff=0.4, v_buff=0.3)
            .to_border(UP, buff=0.5)
            .r
        )
        i_coord = (
            Axes(
                x_range=(0, TAU, PI / 2),
                y_range=(-1, 1, 1),
                x_axis_config=dict(unit_size=2),
                y_axis_config=dict(unit_size=1.25),
                num_sampled_graph_points_per_tick=20,
            )
            .points.to_center()
            .shift(DOWN * 0.25)
            .r
        )

        self.show(i_diagrams, i_coord)

        i_box1 = i_box2 = None
        rectConfig = dict(color=WHITE, stroke_radius=0.015)

        def animateDotProd(n1, n2, n1_is_cos=True, n2_is_cos=True):
            nonlocal i_box1, i_box2

            i_box1Prev, i_box2Prev = i_box1, i_box2
            i_diagram1, i_diagram2 = (
                (i_cosDiagrams if n1_is_cos else i_sinDiagrams)[n1],
                (i_cosDiagrams if n2_is_cos else i_sinDiagrams)[n2],
            )

            i_box1 = SurroundingRect(i_diagram1, buff=0.1, **rectConfig)
            i_box2 = SurroundingRect(i_diagram2, buff=0.1, **rectConfig)

            fn1, fn2 = (
                (np.cos if n1_is_cos else np.sin),
                (np.cos if n2_is_cos else np.sin),
            )
            i_formula = (
                TypstMath(f"""
                angle.l {fn1.__name__}({n1} x), {fn2.__name__}({n2} x) angle.r
                = integral_0^(2 pi) {fn1.__name__}({n1} x) {fn2.__name__}({n2} x) dif x
                = 0
            """)
                .points.to_border(DOWN, buff=1)
                .r
            )
            i_graph = i_coord.get_graph(
                lambda x: fn1(n1 * x) * fn2(n2 * x),
                stroke_color=GREEN_SCREEN,
                glow_color=GREEN_SCREEN,
                glow_size=0.25,
                glow_alpha=0.5,
            )
            i_area = i_coord.get_area(i_graph, fill_color=GREEN_SCREEN)
            self.play(
                *(
                    (FadeIn(i_box1), FadeIn(i_box2))
                    if i_box1Prev is None
                    else (
                        Transform(i_box1Prev, i_box1),
                        Transform(i_box2Prev, i_box2),
                    )
                ),
                duration=0.5,
            )
            self.play(
                Succession(
                    *(
                        item[0].anim.set(stroke_alpha=1, glow_alpha=0.5)
                        for item in (i_diagram1, i_diagram2)
                    )
                ),
                duration=0.75,
            )
            self.play(Create(i_graph), duration=0.5)
            self.play(FadeIn(i_area), FadeIn(i_formula), duration=0.25)
            self.forward(1)
            self.play(
                *(
                    item[0].anim.set(stroke_alpha=0.75, glow_alpha=0)
                    for item in (i_diagram1, i_diagram2)
                ),
                FadeOut(i_graph),
                FadeOut(i_area),
                FadeOut(i_formula),
                duration=0.5,
            )
            self.forward(0.5)
            i_box1Prev, i_box2Prev = i_box1, i_box2

        animageArgs = (
            (0, 2, True, True),
            (1, 4, True, True),
            (3, 6, True, True),
            (1, 2, False, False),
            (1, 4, False, False),
            (3, 6, False, False),
            (1, 2, True, False),
            (3, 5, True, False),
            (2, 3, True, False),
        )
        for args in animageArgs:
            animateDotProd(*args)
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
