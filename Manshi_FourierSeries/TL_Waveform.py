from janim.imports import *
import fantazia.synth.waveform as w

with reloads():
    from common import *


class TL_Waveform(Timeline):
    def __init__(self, waveform: w.Waveform = w.square) -> None:
        super().__init__()
        self._waveform = waveform

    def construct(self):
        waveform = self._waveform
        n_harmonics = 15
        smallAmpThresh = 1e-4

        i_coord = (
            Axes(
                x_range=(0, 1, 0.5),
                y_range=(-1, 1, 1),
                axis_config=dict(tick_size=0.05),
                x_axis_config=dict(unit_size=self.config_getter.frame_x_radius * 2),
                y_axis_config=dict(unit_size=2),
                num_sampled_graph_points_per_tick=100,
            )
            .points.move_to(ORIGIN)
            .r
        )
        for i_axis in i_coord.get_axes():
            i_axis.ticks.set(stroke_radius=0.015)
        i_coord.remove(i_coord.get_axes()[1])
        i_waveformGraph = i_coord.get_graph(
            waveform,
            stroke_color="#00ff00",
            glow_color="#00ff00",
            glow_alpha=0.5,
            glow_size=0.5,
            x_range=(-0.01, 1.01),
        )

        self.play(Create(i_coord))
        self.play(Create(i_waveformGraph))
        self.forward(1)
        self.play(FadeOut(i_waveformGraph))
        self.forward(1)

        i_harmonicGraphs = Group(
            *(
                i_coord.get_graph(partial(waveform.h, k), stroke_color="#00ff00")
                for k in range(1, n_harmonics + 1)
            )
        )
        i_lastCumulativeGraph = None
        for k, (coef, i_harmonicGraph) in enumerate(
            zip(it.islice(waveform, 1, None), i_harmonicGraphs), 1
        ):
            if abs(coef) < smallAmpThresh:
                continue
            i_cumulativeGraph = i_coord.get_graph(waveform[:k], stroke_color="#00ff00")
            self.play(Create(i_harmonicGraph), duration=1)
            if i_lastCumulativeGraph is None:
                self.play(Transform(i_harmonicGraph, i_cumulativeGraph), duration=1)
            else:
                self.play(
                    Transform(i_harmonicGraph, i_cumulativeGraph),
                    Transform(i_lastCumulativeGraph, i_cumulativeGraph),
                    duration=1,
                )
            # self.forward(1)
            i_lastCumulativeGraph = i_cumulativeGraph

        self.play(Transform(i_lastCumulativeGraph, i_waveformGraph))
        self.forward(1)

        for i_ in i_harmonicGraphs:
            i_.stroke.set(alpha=0)
        self.show(i_harmonicGraphs)

        distance = 1
        a_flip = Audio(DIR / "assets/sound/flip.mp3")
        a_penClick = Audio(DIR / "assets/sound/pen-click.wav")
        i_cameraInitState = self.camera.store()
        self.play_audio(a_flip)
        self.play(
            *(
                i_graph.update.points.shift(IN * distance * (i + 1))
                .r.stroke.set(alpha=0.5)
                .r
                for i, i_graph in enumerate(i_harmonicGraphs)
            ),
            self.camera.anim.points.rotate(-PI / 6, axis=RIGHT, about_point=ORIGIN)
            .rotate(PI / 6, axis=UP, about_point=ORIGIN)
            .shift((4, 1.72, 1.5))
            .r,
        )

        totalTime = 3
        incFactor = 1.1
        firstTime = totalTime / (incFactor**n_harmonics - 1) * (incFactor - 1)

        self.forward(0.5)
        self.play_audio(a_penClick)
        self.play(i_harmonicGraphs[0].update.stroke.set(alpha=1), duration=0.1)
        self.forward(firstTime)
        for i, (i_lastGraph, i_graph) in enumerate(it.pairwise(i_harmonicGraphs), 1):
            self.play_audio(a_penClick)
            i_lastGraph.stroke.set(alpha=0.5)
            i_graph.stroke.set(alpha=1)
            self.forward(firstTime * incFactor**i)
        self.play(i_graph.update.stroke.set(alpha=0.5), duration=0.1)
        self.forward(2)

        self.play_audio(a_flip)
        self.play(
            self.camera.anim.restore(i_cameraInitState),
            *(
                i_graph.update.points.shift(OUT * distance * (i + 1))
                .r.stroke.set(alpha=0)
                .r
                for i, i_graph in enumerate(i_harmonicGraphs)
            ),
        )
        self.hide(i_harmonicGraphs)

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
