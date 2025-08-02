import os
from pathlib import Path
import math
import time
import subprocess
import shutil
from functools import lru_cache

from janim.imports import *
import numpy as np
from scipy import fft

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()

waveRectConfig = dict(
    stroke_radius=0,
    stroke_alpha=0,
    fill_alpha=0.75,
    fill_color=GREEN_SCREEN,
    depth=-2,
    glow_size=0,
    glow_alpha=0,
    glow_color=GREEN_SCREEN,
)
selectedWaveRectConfig = waveRectConfig | dict(
    fill_alpha=1,
    glow_size=0.25,
    glow_alpha=0.1,
)
spectrumRectConfig = waveRectConfig | dict(fill_color=PINK, glow_color=PINK)
selectedSpectrumRectConfig = spectrumRectConfig | dict(
    fill_alpha=1,
    glow_size=0.25,
    glow_alpha=0.1,
)
boxConfig = dict(
    fill_color="#282c34",
    fill_alpha=1,
    stroke_radius=0.01,
    stroke_alpha=0.5,
    depth=1,
)
playHeaderConfig = dict(
    stroke_radius=0.01,
    stroke_alpha=0.75,
    stroke_color=WHITE,
)
shadeConfig = dict(
    stroke_radius=0,
    stroke_alpha=0,
    fill_alpha=0.1,
    fill_color=WHITE,
    depth=-1,
)


class MarkedGroup(Group, MarkedItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AudioWaveform(Rect, MarkedItem):
    def __init__(
        self,
        audio_file: os.PathLike,
        width: float = 6,
        height: float = 2,
        begin: float = -1,
        end: float = -1,
        n_samples: int = 100,
        gap_ratio=0.1,
        height_buff=0.25,
    ):
        super().__init__(width, height, **boxConfig)
        self.points.shift(RIGHT * (width / 2))

        self.audio = Audio(audio_file, begin=begin, end=end)
        self.audio_file = Path(audio_file)
        self.n_samples = n_samples
        self._selected_range = (0, 0)
        data = self.audio._samples.data

        maxData = np.max(data)
        data_n_samples = len(data)

        if len(data.shape) > 1:
            data_ch1, data_ch2 = data[:, 0], data[:, 1]
        else:
            data_ch1 = data_ch2 = data

        # self.i_waveRects = i_waveRects = MarkedGroup()
        i_waveRects = MarkedGroup()
        self.i_waveRectsInit = i_waveRectsInit = MarkedGroup()

        for item in (i_waveRects, i_waveRectsInit):
            item.mark.set_points((ORIGIN,))

        wave_rect_width = width / n_samples * (1 - gap_ratio)
        for i in range(n_samples):
            j1 = math.floor(data_n_samples * i / n_samples)
            j2 = math.ceil(data_n_samples * (i + 1) / n_samples)
            up = np.max(data_ch1[j1:j2]) / maxData
            down = np.max(data_ch2[j1:j2]) / maxData
            x1 = width * i / n_samples
            y1 = -(height - height_buff) * down / 2
            y2 = (height - height_buff) * up / 2
            i_waveRectsInit.add(
                Rect(wave_rect_width, 0, **waveRectConfig)
                .points.shift(RIGHT * (wave_rect_width / 2 + x1))
                .r
            )
            i_waveRects.add(
                Rect(wave_rect_width, max(y2 - y1, 0), **waveRectConfig)
                .points.shift((wave_rect_width / 2 + x1, (y1 + y2) / 2, 0))
                .r
            )

        self.i_boxInit = Rect(0, height, **boxConfig)
        # self.i_axis = i_axis = Line(
        i_axis = Line(
            ORIGIN,
            RIGHT * width,
            stroke_radius=0.01,
            stroke_color=WHITE,
            stroke_alpha=0.5,
        )
        # self.i_shadeRect = i_shadeRect = Rect(0, 0, **shadeConfig)
        i_shadeRect = Rect(0, 0, **shadeConfig)
        i_startHeader = Line(ORIGIN, ORIGIN, **playHeaderConfig)
        i_endHeader = Line(ORIGIN, ORIGIN, **playHeaderConfig)
        self.add(i_axis, i_waveRects, i_shadeRect, Group(i_startHeader, i_endHeader))
        self.mark.set_points((ORIGIN,))

    @property
    def i_axis(self) -> Line:
        return self[0]

    @property
    def i_waveRects(self) -> MarkedGroup:
        return self[1]

    @property
    def i_shadeRect(self) -> Rect:
        return self[2]

    @property
    def i_headers(self) -> Group:
        return self[3]

    def getCreateAnim(self):
        self.i_boxInit.points.move_to(self.mark.get())
        self.i_waveRectsInit.mark.set(self.mark.get())
        return Succession(
            AnimGroup(
                AnimGroup(
                    Transform(self.i_boxInit, self, root_only=True),
                    Create(self.i_axis),
                    duration=1,
                ),
                AnimGroup(
                    *(
                        Transform(item1, item2)
                        for item1, item2 in zip(self.i_waveRectsInit, self.i_waveRects)
                    ),
                    lag_ratio=0.01,
                    duration=1,
                ),
                lag_ratio=0.25,
            ),
            FadeIn(self, duration=0),
        )

    def getAudioPlayAnim(self, begin=0, end=-1):
        audioDuration = self.audio.duration()
        begin = 0 if begin < 0 else min(begin, audioDuration)
        end = audioDuration if end < 0 else min(end, audioDuration)

        def updaterFn(item: Self, params: UpdaterParams):
            t = params.alpha
            item.selected_range = (begin, interpolate(begin, end, t))

        return GroupUpdater(self, updaterFn, duration=audioDuration)

    def getSelectedRangeAnim(self, begin=0, end=-1, **kwargs):
        audioDuration = self.audio.duration()
        orig_begin, orig_end = self._selected_range
        begin = 0 if begin < 0 else max(begin, 0)
        end = audioDuration if end < 0 else min(end, audioDuration)

        def updaterFn(item: Self, params: UpdaterParams):
            t = params.alpha
            item.selected_range = (
                interpolate(orig_begin, begin, t),
                interpolate(orig_end, end, t),
            )

        return GroupUpdater(self, updaterFn, **kwargs)

    @property
    def selected_range(self):
        return self._selected_range

    @selected_range.setter
    def selected_range(self, new_range):
        begin, end = new_range
        width = self.points.self_box.width
        height = self.points.self_box.height
        x0, y0, *_ = pos = self.mark.get()

        audioDuration = self.audio.duration()
        begin = 0 if begin < 0 else min(begin, audioDuration)
        end = audioDuration if end < 0 else min(end, audioDuration)
        self._selected_range = (begin, end)

        t_begin = begin / audioDuration
        t_end = end / audioDuration

        x_begin = x0 + width * t_begin
        x_end = x0 + width * t_end
        y_down = y0 - height / 2
        y_up = y0 + height / 2

        self.i_shadeRect.become(
            Rect((x_begin, y_down, 0), (x_end, y_up, 0), **shadeConfig)
        )

        i_startHeader, i_endHeader = self.i_headers
        if begin > 0:
            i_startHeader.points.put_start_and_end_on(
                (x_begin, y_down, 0), (x_begin, y_up, 0)
            )
        else:
            i_startHeader.points.put_start_and_end_on(pos, pos)
        if end < audioDuration and end != begin:
            i_endHeader.points.put_start_and_end_on(
                (x_end, y_up, 0), (x_end, y_down, 0)
            )
        else:
            i_endHeader.points.put_start_and_end_on(pos, pos)

        j1 = math.floor(self.n_samples * t_begin)
        j2 = math.ceil(self.n_samples * t_end)

        self.i_waveRects[:j1].set(**waveRectConfig)
        self.i_waveRects[j1:j2].set(**selectedWaveRectConfig)
        self.i_waveRects[j2:].set(**waveRectConfig)

        return self


class AudioSpectrum(Rect, MarkedItem):
    def __init__(
        self,
        audio_file: os.PathLike,
        width: float = 6,
        height: float = 4,
        begin: float = -1,
        end: float = -1,
        freq_range: tuple[float, float] = (20, 20000),
        min_db=-3,
        n_samples: int = 100,
        gap_ratio: float = 0.1,
        height_buff: float = 0.25,
    ):
        super().__init__(ORIGIN, (width, height, 0), **boxConfig)
        self.audio = Audio(audio_file, begin=begin, end=end)
        self.n_samples = n_samples
        data = self.audio._samples.data
        audioDuration = self.audio.duration()

        if len(data.shape) > 1:
            data_fft = fft.fft(data, axis=0)
            # data_fft = data_fft[: len(data_fft) // 2]
            fft_ch1 = data_fft[:, 0]
            fft_ch2 = data_fft[:, 1]
        else:
            data_fft = fft.fft(data)
            # data_fft = data_fft[: len(data_fft) // 2]
            fft_ch1 = fft_ch2 = data_fft

        freq_range_log = np.log10(freq_range)
        self.freq_range = freq_range
        self._selected_range = (freq_range[0], freq_range[0])

        i_dataRects = MarkedGroup()
        self.i_dataRectsInit = i_dataRectsInit = MarkedGroup()
        for item in (i_dataRects, i_dataRectsInit):
            item.mark.set_points((ORIGIN,))
        data_rect_width = width / n_samples * (1 - gap_ratio)
        max_amp = np.max(np.abs(data_fft))

        for i in range(n_samples):
            f1 = 10 ** interpolate(*freq_range_log, i / n_samples)
            f2 = 10 ** interpolate(*freq_range_log, (i + 1) / n_samples)
            j1 = math.floor(f1 * audioDuration)
            j2 = math.ceil(f2 * audioDuration)

            amp_up = np.max(np.abs(fft_ch1[j1:j2])) / max_amp
            amp_down = np.max(np.abs(fft_ch2[j1:j2])) / max_amp

            t_up = max(0, (np.log10(amp_up) - min_db) / (-min_db))
            t_down = max(0, (np.log10(amp_down) - min_db) / (-min_db))

            x = interpolate(0, width, i / n_samples)
            x2 = x + data_rect_width
            y = (height - height_buff) * (t_up + t_down) / 2

            i_dataRects.add(Rect((x, 0, 0), (x2, y, 0), **spectrumRectConfig))
            i_dataRectsInit.add(Rect((x, 0, 0), (x2, 0, 0), **spectrumRectConfig))
        # print(data_fft)
        i_shadeRect = Rect(0, 0, **shadeConfig)
        i_startHeader = Line(ORIGIN, ORIGIN, **playHeaderConfig)
        i_endHeader = Line(ORIGIN, ORIGIN, **playHeaderConfig)
        self.add(i_dataRects, i_shadeRect, Group(i_startHeader, i_endHeader))
        self.mark.set_points((ORIGIN,))

    @property
    def i_dataRects(self) -> MarkedGroup:
        return self[0]

    @property
    def i_shadeRect(self) -> Rect:
        return self[1]

    @property
    def i_headers(self) -> Group:
        return self[2]

    def getCreateAnim(self):
        i_rectInit = (
            Rect(self.points.self_box.width, 0, **boxConfig)
            .points.move_to(self.points.self_box.bottom)
            .r
        )
        self.i_dataRectsInit.mark.set(self.mark.get())
        return Succession(
            AnimGroup(
                Transform(i_rectInit, self, root_only=True),
                AnimGroup(
                    *(
                        Transform(item1, item2)
                        for item1, item2 in zip(self.i_dataRectsInit, self.i_dataRects)
                    ),
                    lag_ratio=0.01,
                    duration=1,
                ),
                lag_ratio=0.25,
            ),
            FadeIn(self, duration=0),
        )

    def getSelectedRangeAnim(self, f_begin=-1, f_end=-1, **kwargs):
        f_min, f_max = self.freq_range
        orig_f_begin, orig_f_end = self._selected_range
        f_begin = f_min if f_begin <= 0 else max(f_begin, f_min)
        f_end = f_max if f_end < 0 else min(f_end, f_max)

        def updaterFn(item: Self, params: UpdaterParams):
            t = params.alpha
            item.selected_range = (
                interpolate(orig_f_begin, f_begin, t),
                interpolate(orig_f_end, f_end, t),
            )

        return GroupUpdater(self, updaterFn, **kwargs)

    @property
    def selected_range(self):
        return self._selected_range

    @selected_range.setter
    def selected_range(self, new_range):
        f_min, f_max = self.freq_range
        f_begin, f_end = new_range

        f_begin = f_begin if f_begin <= 0 else max(f_begin, f_min)
        f_end = f_end if f_end <= 0 else min(f_end, f_max)
        self._selected_range = (f_begin, f_end)

        f_min_log, f_max_log = np.log10(self.freq_range)
        f_begin_log, f_end_log = np.log10(f_begin), np.log10(f_end)

        t_begin = (f_begin_log - f_min_log) / (f_max_log - f_min_log)
        t_end = (f_end_log - f_min_log) / (f_max_log - f_min_log)

        x0, y0, *_ = pos = self.mark.get()
        width, height = self.points.self_box.width, self.points.self_box.height
        x_begin = x0 + width * t_begin
        x_end = x0 + width * t_end
        y_up = y0 + height

        i_startHeader, i_endHeader = self.i_headers
        if f_begin > f_min:
            i_startHeader.points.put_start_and_end_on(
                (x_begin, y0, 0), (x_begin, y_up, 0)
            )
        else:
            i_startHeader.points.put_start_and_end_on(pos, pos)
        if f_end < f_max and f_end != f_begin:
            i_endHeader.points.put_start_and_end_on((x_end, y0, 0), (x_end, y_up, 0))
        else:
            i_endHeader.points.put_start_and_end_on(pos, pos)

        self.i_shadeRect.become(
            Rect((x_begin, y0, 0), (x_end, y0 + height, 0), **shadeConfig)
        )

        j1 = math.floor(t_begin * self.n_samples)
        j2 = math.ceil(t_end * self.n_samples)

        self.i_dataRects[:j1].set(**spectrumRectConfig)
        self.i_dataRects[j1:j2].set(**selectedSpectrumRectConfig)
        self.i_dataRects[j2:].set(**spectrumRectConfig)

        return self


@lru_cache
def getFilteredAudio(audio_file: os.PathLike, f_begin=20, f_end=20000):
    audio_file = Path(audio_file)
    timestamp = time.time_ns()
    temp_folder = DIR / f"__temp_{timestamp}"
    temp_folder.mkdir(exist_ok=True, parents=True)
    out_audio_file = temp_folder / f"{audio_file.stem}_filtered.wav"
    subprocess.run(
        (
            "sox",
            str(audio_file),
            str(out_audio_file),
            "bandpass",
            str(f_begin),
            str(f_end - f_begin),
        )
    )
    audio = Audio(out_audio_file)
    shutil.rmtree(temp_folder)
    return audio


class TL_Test(Timeline):
    def construct(self):
        i_spectrum = (
            AudioSpectrum(DIR / "assets/sound/watermelon/watermelon_1.wav")
            .points.to_center()
            .r
        )
        self.show(i_spectrum)
        self.forward(2)


class TL_Watermelon(Timeline):
    def construct(self):
        i_watermelon_template = (
            SVGItem(DIR / "assets/image/watermelon.svg").points.set_width(2).r
        )
        i_waveforms = Group(
            *(
                AudioWaveform(
                    DIR / f"assets/sound/watermelon/watermelon_{i}.wav", width=4
                )
                for i in range(1, 4)
            )
        )
        n = len(i_waveforms)
        i_spectrums = (
            Group(
                *(
                    AudioSpectrum(
                        DIR / f"assets/sound/watermelon/watermelon_{i}.wav",
                        width=4,
                        height=3,
                    )
                    for i in range(1, 4)
                )
            )
            .points.arrange_in_grid(n_rows=1, h_buff=0.5)
            .r
        )
        i_watermelons = i_watermelon_template * n
        Group(*i_waveforms, *i_watermelons).points.arrange_in_grid(
            n_cols=n, h_buff=0.5, v_buff=0.75
        ).shift(UP * 0.5)
        i_spectrums.points.next_to(i_waveforms, DOWN, buff=0.5)

        self.forward(0.25)
        self.play(
            AnimGroup(
                *(item.getCreateAnim() for item in i_waveforms),
                lag_ratio=0.5,
                duration=2,
            ),
            AnimGroup(
                *(GrowFromCenter(item) for item in i_watermelons),
                lag_ratio=0.5,
                duration=2,
            ),
        )

        self.forward(0.5)
        for i_waveform, i_watermelon in zip(i_waveforms, i_watermelons):
            for _ in range(3):
                self.play_audio(i_waveform.audio)
                self.prepare(
                    i_waveform.getAudioPlayAnim(),
                    WiggleOutThenIn(i_watermelon, duration=0.25, n_wiggles=3),
                )
                self.forward(0.25)
            self.play(i_waveform.getSelectedRangeAnim(begin=0, end=0), duration=0.5)
            self.forward(1)

        self.forward(1)
        self.play(FadeOut(i_watermelons))
        self.play(
            AnimGroup(*(item.getCreateAnim() for item in i_spectrums), lag_ratio=0.5)
        )
        self.forward(1)

        def playSounds(repeat=3):
            for i_waveform in i_waveforms:
                for _ in range(repeat):
                    self.play_audio(i_waveform.audio)
                    self.prepare(i_waveform.getAudioPlayAnim())
                    self.forward(0.25)
                self.play(i_waveform.getSelectedRangeAnim(begin=0, end=0), duration=0.5)
                self.forward(0.25)

        def playFilteredSounds(f_begin=20, f_end=20000, repeat=3):
            self.play(
                *(
                    item.getSelectedRangeAnim(
                        f_begin, f_end, rate_func=ease_inout_quart
                    )
                    for item in i_spectrums
                )
            )
            self.forward(0.5)
            for i_waveform in i_waveforms:
                for _ in range(repeat):
                    self.play_audio(
                        getFilteredAudio(i_waveform.audio_file, f_begin, f_end)
                    )
                    self.prepare(i_waveform.getAudioPlayAnim())
                    self.forward(0.25)
                self.play(i_waveform.getSelectedRangeAnim(begin=0, end=0), duration=0.5)
                self.forward(0.25)

        low_freq = (20, 80)
        low_mid_freq = (80, 250)
        high_freq = (1000, 20000)

        playSounds()
        for freq_lims in (low_freq, low_mid_freq, high_freq):
            playFilteredSounds(*freq_lims)
        self.play(
            *(
                item.getSelectedRangeAnim(20, 20, rate_func=ease_inout_quart)
                for item in i_spectrums
            )
        )
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
