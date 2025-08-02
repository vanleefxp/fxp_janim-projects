from pathlib import Path

import numpy as np
from janim.imports import *
from fantazia.synth.waveform import analyzeSoundFile

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


class TL_FourierDotProduct(Timeline):
    def construct(self):
        waveform = analyzeSoundFile(
            DIR / "../assets/sound/instrument/piano.wav", sampleTime=0.5
        )[1]

        def func(x):
            return np.sin(2 * x * TAU)

        coordXUnit = 8
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius

        def createCoord():
            i_coord = (
                Axes(
                    x_range=(*(np.array((-rx, rx)) / coordXUnit + 0.5), 0.5),
                    num_sampled_graph_points_per_tick=100,
                    x_axis_config=dict(unit_size=coordXUnit, numbers_to_exclude=()),
                    y_axis_config=dict(unit_size=1),
                )
                .points.to_center()
                .r
            )
            i_yAxis = i_coord.get_axes()[1]
            i_yAxis.set(stroke_alpha=0, stroke_radius=0)
            i_yAxis.ticks.set(stroke_alpha=0, stroke_radius=0)
            # i_coord.remove(i_coord.get_axes()[1])
            return i_coord

        i_coord = createCoord().points.shift(UP * 1.8).r
        i_coord2 = createCoord().points.shift(DOWN * 1.8).r

        i_waveformGraph = i_coord.get_graph(
            waveform,
            stroke_color=GREEN_SCREEN,
            glow_size=0.5,
            glow_alpha=0.5,
            glow_color=GREEN_SCREEN,
            bind=False,
        )
        i_funcGraph = i_coord2.get_graph(func, stroke_color=RED, bind=False)

        self.play(Create(i_waveformGraph), FadeIn(i_coord))
        self.forward(1)
        self.play(Create(i_funcGraph), FadeIn(i_coord2))
        self.forward(1)

        def createDivisionLines(n_divisions, i_coord=i_coord, fn=waveform):
            i_divLines = Group()

            for x in np.linspace(0, 1, n_divisions + 1):
                pointOnGraph = i_coord.c2p(x, fn(x))
                pointOnX = i_coord.c2p(x, 0)
                i_dot = Dot(pointOnGraph, radius=0.04)
                i_line = Line(
                    pointOnGraph, pointOnX, stroke_alpha=0.75, stroke_radius=0.015
                )
                i_divLines.add(Group(i_dot, i_line))

            return i_divLines

        i_waveToAxisArea = i_coord.get_area(
            i_waveformGraph, (0, 1), fill_color=GREEN_SCREEN
        )
        i_funcToAxisArea = i_coord2.get_area(i_funcGraph, (0, 1), fill_color=RED)

        i_periodRectInit = Rect(
            (i_coord.c2p(0, -1)[0], -ry, 0),
            (i_coord.c2p(0.01, 1)[0], ry, 0),
            fill_color=WHITE,
            fill_alpha=0.25,
            stroke_alpha=0,
            stroke_radius=0,
        )
        i_periodRect = Rect(
            (i_coord.c2p(0, -1)[0], -ry, 0),
            (i_coord.c2p(1, 1)[0], ry, 0),
            fill_color=WHITE,
            fill_alpha=0.2,
            stroke_alpha=0,
            stroke_radius=0,
            depth=1,
        )

        self.play(Transform(i_periodRectInit, i_periodRect))
        i_divLines = createDivisionLines(10)
        i_divLines2 = createDivisionLines(10, i_coord2, func)
        self.play(
            AnimGroup(*(Create(i_) for i_ in i_divLines), lag_ratio=0.25),
            AnimGroup(*(Create(i_) for i_ in i_divLines2), lag_ratio=0.25),
            duration=2,
        )
        self.forward(1)

        divs = range(11, 40)

        totalTime = 4
        incFactor = 0.9
        firstTime = totalTime / (incFactor ** len(divs) - 1) * (incFactor - 1)

        for i, n_divs in enumerate(divs):
            self.hide(i_divLines, i_divLines2)
            i_divLines = createDivisionLines(n_divs)
            i_divLines2 = createDivisionLines(n_divs, i_coord2, func)
            self.show(i_divLines, i_divLines2)
            self.forward(firstTime * incFactor**i)

        self.play(
            *(FadeOut(i_) for i_ in (i_divLines, i_divLines2)),
            *(FadeIn(i_) for i_ in (i_waveToAxisArea, i_funcToAxisArea)),
            duration=0.5,
        )
        i_dotProductFormula = TypstMath(
            'angle.l f, g angle.r = integral_0^T f(x) g(x) "d"x'
        )
        for idx in (1, 9):
            i_dotProductFormula[idx].set(fill_color=GREEN)
        for idx in (3, 13):
            i_dotProductFormula[idx].set(fill_color=RED)
        self.play(Write(i_dotProductFormula))
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
