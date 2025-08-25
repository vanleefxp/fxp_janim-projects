from janim.imports import *
from fantazia.synth.waveform import analyzeSoundFile, HarmonicSeries
from common import *

with reloads():
    from fantazia.synth.waveform import analyzeSoundFile, HarmonicSeries
    from common import *


class TL_WaveformFreqDomain(Timeline):
    CONFIG = config

    def construct(self):
        waveform = analyzeSoundFile(
            DIR / "../assets/sound/instrument/violin.wav",
            sampleTime=0.5,
            harmonicThreshold=0.05,
        )[1]
        # waveform = SawtoothWave()

        n_harmonics = 15
        hasMoreHarmonics = True
        amp_mul = 1.8
        smallAmp = 1e-5
        waveformPos = ORIGIN
        z_distance_multiple = 1

        if isinstance(waveform, HarmonicSeries):
            if len(waveform) < n_harmonics:
                n_harmonics = len(waveform)
                hasMoreHarmonics = False
        amps = np.array([np.abs(waveform[i]) for i in range(n_harmonics + 1)])
        phases = np.array([np.angle(waveform[i]) for i in range(n_harmonics + 1)])
        phases[phases < 0] += np.pi * 2
        max_amp = np.max(amps)

        i_waveform = WaveformDiagram(waveform, depth=-1).mark.set(waveformPos).r

        wave_width, wave_height = (
            i_waveform.points.self_box.width,
            i_waveform.points.self_box.height,
        )
        z_distance = wave_width * z_distance_multiple
        z_unit = z_distance / n_harmonics
        spectrumBarWidth = 0.8
        ampBarHeight = 6
        phaseBarHeight = 2

        i_waveBg = (
            Rect(
                wave_width,
                wave_height * amp_mul,
                color=WHITE,
                fill_alpha=0.1,
                stroke_radius=0.01,
                depth=0.001,
            )
            .points.move_to(i_waveform.mark.get())
            .r
        )
        i_waveBgInit = (
            i_waveBg.copy()
            .points.set_width(0, stretch=True)
            .move_to(i_waveBg.points.self_box.left)
            .r
        )
        i_waveAxis = NumberLine(
            x_range=(0, 1, 0.25), unit_size=wave_width, tick_size=0.05
        )
        i_waveAxis.ticks.set(stroke_radius=0.01)
        i_waveAxis.points.shift(i_waveform.mark.get())
        i_zAxis = NumberLine(
            x_range=(0, n_harmonics, 1),
            unit_size=z_unit,
            tick_size=0.05,
        )
        i_zAxis.points.shift(-i_zAxis.n2p(0)).rotate(
            PI / 2, axis=UP, about_point=ORIGIN
        )
        i_zAxis.ticks.set(stroke_radius=0.01)
        i_timeDomainLabel = (
            Text("时域", **textCfg)
            .points.next_to(i_waveBg.points.self_box.get(DR), DL, buff=(0.1, 0.2, 0))
            .r
        )
        i_timeDomainLabelLarge = (
            Text("时域", **textCfg)
            .points.scale(1.25)
            .next_to(i_waveBg.points.self_box.get(DR), DL, buff=(0.1, 0.2, 0))
            .r
        )
        i_freqDomainLabelLarge = (
            Text("频域", **textCfg)
            .points.scale(1.25)
            .next_to(ORIGIN, UR, buff=0.2)
            .rotate(PI / 2, axis=UP, about_point=ORIGIN)
            .rotate(PI / 6, axis=OUT, about_point=ORIGIN)
            .r
        )

        i_harmonics = Group(
            *(
                WaveformDiagram(lambda t: waveform.h(k, t), alpha=0.5)
                .mark.set(waveformPos)
                .r
                for k in range(1, n_harmonics + 1)
            )
        )
        i_waveformCum = Group(
            *(WaveformDiagram(waveform[:k]) for k in range(n_harmonics + 1))
        )

        # 预定义摄像机视角
        # 保存初始视角备用
        i_cameraInit = self.camera.store()
        # 3D 视角
        (
            self.camera.points.rotate(-PI / 6, axis=RIGHT, about_point=ORIGIN)
            .rotate(PI / 6, axis=UP, about_point=ORIGIN)
            .shift((4, 1.72, 1.5))
        )
        i_camera3d = self.camera.store()
        self.camera.restore(i_cameraInit)
        # 右侧视图
        self.camera.points.rotate(PI / 2, axis=UP).shift((1, 0, -z_distance / 2))
        i_cameraRight = self.camera.store()
        self.camera.restore(i_cameraInit)
        # 右侧视图，频谱柱状图
        self.camera.points.rotate(PI / 2, axis=UP).shift(
            (5, ampBarHeight * max_amp / 2 - 1, -n_harmonics / 2)
        )
        i_cameraRight2 = self.camera.store()
        self.camera.restore(i_cameraInit)

        self.forward(0.25)
        self.show(i_waveBgInit)
        self.play(Transform(i_waveBgInit, i_waveBg))
        self.play(FadeIn(i_waveAxis))
        self.forward(0.5)
        self.play(Create(i_waveform))
        self.forward(0.5)

        i_lastWaveform = i_curWaveform = i_waveformCum[0]
        self.play(Transform(i_waveform, i_waveformCum[0]))
        self.forward(0.5)
        for coef, i_curWaveform, i_harmonic in zip(
            # waveform,
            it.islice(waveform, 1, None),
            i_waveformCum[1:],
            i_harmonics,
        ):
            if np.abs(coef) > smallAmp:
                self.play(Create(i_harmonic), duration=0.5)
                self.play(
                    Transform(i_lastWaveform, i_curWaveform),
                    Transform(i_harmonic, i_curWaveform),
                    duration=0.75,
                )
                self.forward(0.25)
                i_lastWaveform = i_curWaveform

        if hasMoreHarmonics:
            self.play(
                Transform(i_lastWaveform, i_waveform),
            )
        else:
            self.hide(i_lastWaveform)
            self.show(i_waveform)
        self.forward(2)

        for item in i_harmonics:
            item.i_graph.set(alpha=0)
        self.show(i_harmonics)

        # 将全部泛音成分的正弦波压缩成线段
        i_waveformCompressed = i_waveform.copy().points.set_width(0, stretch=True).r
        i_waveformCompressed.i_graph.set(alpha=0)
        i_waveBgCompressed = (
            i_waveBg.copy().set(alpha=0).points.set_width(0, stretch=True).r
        )
        i_waveAxisCompressed = (
            i_waveAxis.copy().set(alpha=0).points.set_width(0, stretch=True).r
        )
        i_harmonicsCompressed = i_harmonics.copy().points.set_width(0, stretch=True).r
        for n, i_harmonic in enumerate(i_harmonicsCompressed, 1):
            i_harmonic.i_graph.set(alpha=1)
            i_harmonic.mark.set((0, 0, -n / n_harmonics * z_distance))

        # 音效
        a_flip = Audio(DIR / "assets/sound/flip.mp3")
        a_penClick = Audio(DIR / "assets/sound/pen-click.wav")

        # 转到 3D 视角
        self.play(Write(i_timeDomainLabel), duration=0.5)
        self.forward(0.5)
        self.play_audio(a_flip)
        self.prepare(FadeIn(i_zAxis.ticks, duration=0.5), at=3.5)
        self.play(
            self.camera.anim.restore(i_camera3d),
            *(
                item.anim.mark.set((0, 0, -n * z_unit))
                for n, item in enumerate(i_harmonics, 1)
            ),
            *(item.i_graph.anim.set(alpha=0.5) for item in i_harmonics),
            Create(i_zAxis, root_only=True),
            Transform(i_timeDomainLabel, i_timeDomainLabelLarge),
            duration=4,
        )
        self.forward(0.5)
        self.play(Write(i_freqDomainLabelLarge), duration=0.5)
        self.forward(2)

        # 点数一遍所有泛音
        totalTime = 3
        incFactor = 1.1
        firstTime = totalTime / (incFactor**n_harmonics - 1) * (incFactor - 1)

        self.play_audio(a_penClick)
        self.play(i_harmonics[0].i_graph.anim.set(alpha=1), duration=0.1)
        self.forward(firstTime)
        for i, (i_lastHarmonic, i_curHarmonic) in enumerate(
            it.pairwise(i_harmonics), 1
        ):
            self.play_audio(a_penClick)
            i_lastHarmonic.i_graph.set(alpha=0.5)
            i_curHarmonic.i_graph.set(alpha=1)
            self.forward(firstTime * incFactor**i)
        self.play(i_curHarmonic.i_graph.anim.set(alpha=0.5), duration=0.1)
        self.forward(2)

        # 转到右侧视图
        self.play_audio(a_flip)
        self.play(
            self.camera.anim.restore(i_cameraRight),
            *(FadeOut(item) for item in (i_waveform, i_waveAxis, i_waveBg)),
            Transform(i_waveform, i_waveformCompressed),
            Transform(i_harmonics, i_harmonicsCompressed),
            Transform(i_waveBg, i_waveBgCompressed),
            Transform(i_waveAxis, i_waveAxisCompressed),
            FadeOut(i_freqDomainLabelLarge),
            FadeOut(i_timeDomainLabelLarge),
            duration=2,
        )
        self.forward(0.5)

        # 右视图转化为频谱柱状图
        i_harmonicRects = Group()
        i_ampBars = Group()
        i_phaseBars = Group()
        i_phaseBarsCompressed = Group()
        for n, (amp, phase) in enumerate(zip(amps, phases)):
            i_harmonicRects.add(
                Rect(
                    DOWN * amp * wave_height / 2,
                    UP * amp * wave_height / 2,
                    color=GREEN_SCREEN,
                    fill_alpha=0.5,
                )
                .points.rotate(PI / 2, axis=UP, about_point=ORIGIN)
                .shift(IN * (n * z_unit))
                .r
            )
            i_ampBars.add(
                Rect(
                    (-spectrumBarWidth / 2, 0, 0),
                    (
                        spectrumBarWidth / 2,
                        ampBarHeight * amp / max_amp,
                        0,
                    ),
                    color=GREEN_SCREEN,
                    stroke_radius=0.01,
                    fill_alpha=0.5,
                )
                .points.rotate(PI / 2, axis=UP, about_point=ORIGIN)
                .shift(IN * n)
                .r
            )
            i_phaseBarsCompressed.add(
                Rect(
                    (-spectrumBarWidth / 2, 0, 0),
                    (spectrumBarWidth / 2, 0, 0),
                    color=GREY_C,
                    stroke_radius=0.01,
                    fill_alpha=0.5,
                )
                .points.rotate(PI / 2, axis=UP, about_point=ORIGIN)
                .shift(IN * n)
                .r
            )
            i_phaseBars.add(
                Rect(
                    (-spectrumBarWidth / 2, 0, 0),
                    (
                        spectrumBarWidth / 2,
                        -phaseBarHeight * phase / TAU,
                        0,
                    ),
                    color=GREY_C,
                    stroke_radius=0.01,
                    fill_alpha=0.5,
                )
                .points.rotate(PI / 2, axis=UP, about_point=ORIGIN)
                .shift(IN * n)
                .r
            )

        i_newZAxis = (
            NumberLine(
                x_range=(0, n_harmonics, 1),
                unit_size=1,
                tick_size=0.05,
                depth=1,
            )
            .mark.set(ORIGIN)
            .r.points.rotate(PI / 2, axis=UP, about_point=ORIGIN)
            .r.depth.arrange()
            .r
        )
        # i_axisNumbers = Group()
        # for i in range(n_harmonics + 1):
        #     i_axisNumbers.add(
        #         Text(str(i), **textCfg)
        #         .points.scale(1.125)
        #         .next_to(ORIGIN, DOWN, buff=0.2)
        #         .rotate(PI / 2, axis=UP, about_point=ORIGIN)
        #         .shift(i * IN)
        #         .r
        #     )

        self.hide(
            i_harmonicsCompressed,
            i_waveformCompressed,
            i_waveBgCompressed,
            i_waveAxisCompressed,
        )
        self.show(i_harmonicRects)
        self.play(
            Transform(i_harmonicRects, i_ampBars),
            Transform(i_zAxis, i_newZAxis),
            # FadeIn(i_axisNumbers),
            self.camera.anim.restore(i_cameraRight2),
        )
        i_ampLabel = (
            Text("振幅", **textCfg)
            .points.scale(1.125)
            .next_to(ORIGIN, UL)
            .rotate(PI / 2, axis=UP, about_point=ORIGIN)
            .r
        )
        i_phaseLabel = (
            Text("相位", **textCfg)
            .points.scale(1.125)
            .next_to(ORIGIN, DL)
            .rotate(PI / 2, axis=UP, about_point=ORIGIN)
            .r
        )
        self.show(i_phaseBarsCompressed)
        self.play(
            Transform(i_phaseBarsCompressed, i_phaseBars),
            *(FadeIn(item) for item in (i_ampLabel, i_phaseLabel)),
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
##################################################################
