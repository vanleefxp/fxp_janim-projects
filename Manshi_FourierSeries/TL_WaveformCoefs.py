from janim.imports import *
from fantazia.synth.waveform import (
    analyzeSoundFile,
    Waveform,
    SineWave,
    SquareWave,
    TriangleWave,
    SawtoothWave,
)

with reloads():
    from common import *


def getWaveCoefs(waveform: Waveform, n: int) -> np.ndarray:
    coefs: np.ndarray[complex] = waveform.coefs[:n]
    amps = np.abs(coefs)
    phases = np.nan_to_num(np.angle(coefs), copy=False, nan=0)
    coefs_real = np.empty((2, n), float)
    coefs_real[0] = amps * np.sin(phases)
    coefs_real[1] = amps * np.cos(phases)
    return coefs_real


class SineDiagram(Rect):
    def __init__(self, freq=1, phi=0, **kwargs):
        super().__init__(
            (0, -1, 0), (1, 1, 0), stroke_radius=0, stroke_alpha=0, **kwargs
        )
        self.add(
            FunctionGraph(
                lambda x: np.sin(TAU * freq * x + phi),
                x_range=(0, 1, 0.01),
                stroke_color=GREEN_SCREEN,
                stroke_alpha=0.75,
                glow_color=GREEN_SCREEN,
                glow_size=0.25,
            )
        )
        self.points.set_size(1.5, 0.625)

    @property
    def i_graph(self) -> FunctionGraph:
        return self[0]


class TL_WaveformCoefs(Timeline):
    CONFIG = config

    def construct(self):
        n = 7
        waveforms = (
            *(
                analyzeSoundFile(
                    DIR / f"../assets/sound/instrument/{name}.wav", sampleTime=0.5
                )[1]
                for name in ("piano", "violin", "trombone", "oboe", "sax")
            ),
            SquareWave(),
            TriangleWave(),
            SawtoothWave(),
            SineWave(),
        )
        waveform = waveforms[0]

        i_cosines = Group(*(SineDiagram(freq=i, phi=PI / 2) for i in range(n)))
        i_sines = Group(*(SineDiagram(freq=i) for i in range(n)))
        i_diagrams = Group(i_cosines, i_sines)
        i_sines[0].i_graph.set(stroke_alpha=0.375)
        (
            Group(*i_cosines, *i_sines)
            .points.arrange_in_grid(n_rows=2, h_buff=0.375, v_buff=0.75)
            .to_border(DOWN, buff=1.5)
        )

        def createWaveCoef(value, col, row) -> Text:
            return (
                Text(
                    f"{value:.4f}".replace("-", "−"),
                    stroke_alpha=1,
                    stroke_radius=0.005,
                )
                .points.scale(0.95)
                .next_to(i_diagrams[row][col], DOWN, buff=0.125)
                .r
            )

        def createWaveCoefs(coefs):
            coefs_cos, coefs_sin = coefs
            i_cosCoefs = Group()
            i_sinCoefs = Group()
            for i_diagram, coef in zip(i_cosines, coefs_cos):
                i_text = (
                    Text(
                        f"{coef:.4f}".replace("-", "−"),
                        stroke_alpha=1,
                        stroke_radius=0.005,
                    )
                    .points.scale(0.95)
                    .next_to(i_diagram, DOWN, buff=0.125)
                    .r
                )
                i_cosCoefs.add(i_text)
            for i_diagram, coef in zip(i_sines, coefs_sin):
                i_text = (
                    Text(
                        f"{coef:.4f}".replace("-", "−"),
                        stroke_alpha=1,
                        stroke_radius=0.005,
                    )
                    .points.scale(0.95)
                    .next_to(i_diagram, DOWN, buff=0.125)
                    .r
                )
                i_sinCoefs.add(i_text)
            return Group(i_cosCoefs, i_sinCoefs)

        def createWaveform(waveform: Waveform) -> WaveformDiagram:
            return WaveformDiagram(waveform, glow_size=0.5).mark.set((0, 1.875, 0)).r

        def animateWaveformCreation(waveform: Waveform):
            nonlocal i_waveform, coefs
            new_coefs = getWaveCoefs(waveform, n)
            i_waveform = createWaveform(waveform)
            waveforms = (waveform[:i] for i in range(n))
            i_waveforms = Group(*(createWaveform(w) for w in waveforms))
            i_lastWaveform = None
            for i, (i_currentWaveform, (cos_coef, sin_coef)) in enumerate(
                zip(i_waveforms, new_coefs.T)
            ):
                coefUpdateAnims = (
                    getCoefUpdateAnimation(cos_coef, i, 0),
                    getCoefUpdateAnimation(sin_coef, i, 1),
                )
                if i_lastWaveform is None:
                    self.play(Create(i_currentWaveform), *coefUpdateAnims)
                else:
                    self.play(
                        Transform(i_lastWaveform, i_currentWaveform), *coefUpdateAnims
                    )
                i_lastWaveform = i_currentWaveform
                self.forward(0.5)
            self.play(Transform(i_lastWaveform, i_waveform))
            coefs = new_coefs

        def animateWaveformTransform(new_waveform: Waveform):
            nonlocal i_waveform, coefs
            old_coefs = coefs.copy()
            new_coefs = getWaveCoefs(new_waveform, n)
            i_newWaveform = createWaveform(new_waveform)

            def coefsUpdaterFn(params: UpdaterParams):
                current_coefs = interpolate(old_coefs, new_coefs, params.alpha)
                return createWaveCoefs(current_coefs)

            self.play(
                ItemUpdater(i_coefs, coefsUpdaterFn),
                Transform(i_waveform, i_newWaveform),
                duration=3,
            )
            i_waveform = i_newWaveform
            coefs = new_coefs

        def getCoefUpdateAnimation(value, col, row):
            start = coefs[row][col]

            def updaterFn(params: UpdaterParams) -> Text:
                return createWaveCoef(interpolate(start, value, params.alpha), col, row)

            return ItemUpdater(i_coefs[row][col], updaterFn)

        i_waveform = None
        coefs = np.zeros((2, n), dtype=float)
        i_coefs = createWaveCoefs(coefs)

        self.play(
            AnimGroup(
                *(Create(item, rate_func=linear) for item in i_cosines), lag_ratio=1
            ),
            AnimGroup(
                *(Create(item, rate_func=linear) for item in i_sines), lag_ratio=1
            ),
            lag_ratio=0.25,
            duration=2,
        )
        self.play(FadeIn(i_coefs))
        animateWaveformCreation(waveform)
        self.forward(1)
        print(coefs)
        for waveform in waveforms[1:]:
            animateWaveformTransform(waveform)
            self.forward(1)
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
