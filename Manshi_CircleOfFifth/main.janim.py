import sys
from pathlib import Path

from janim.imports import *
import fantazia as fz
import pyrsistent as pyr

_musicFont = ("Chaconne Ex", "Leland", "Bravura")
_textFont = ("FandolSong",)


class Text(Text):
    def __init__(
        text: str,
        font: str | Iterable[str] = _textFont,
        weight: int | Weight | WeightName = "regular",
        *args,
        **kwargs,
    ):
        super().__init__(text, font=font, weight=weight, *args, **kwargs)


DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
sys.path.append(str((DIR / "..").resolve()))
from public.items.music import *  # noqa: E402
from public.items.coord import *  # noqa: E402
from public.utils.number_theory_utils import modShift  # noqa: E402
from public.sound import midi2Audio  # noqa: E402


_staffConfig = pyr.m(
    staffLineThickness=0.05,
    staffHeight=0.6,
    musicFont=_musicFont,
)


class MajorName(Group[NoteName | Text], PositionedVItem):
    def __init__(
        self,
        tonic: fz.Pitch,
        font: str | Iterable[str] = (),
        musicFont: str | Iterable[str] = _musicFont,
        fs=DEFAULT_FONT_SIZE,
        buff=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ch = Text("0", font_size=fs, font=font).points.box.width
        i_noteName = self._i_noteName = NoteName(
            tonic, font=font, musicFont=musicFont, fs=fs
        )
        i_majorText = self._i_majorText = Text(
            "大调", font=font, font_size=fs, **kwargs
        )
        i_noteName.points.next_to(i_majorText, LEFT, buff=buff * ch, coor_mask=RIGHT)
        self.add(i_noteName, i_majorText)

    @property
    def i_noteName(self) -> NoteName:
        return self._i_noteName

    @property
    def i_majorText(self) -> Text:
        return self._i_majorText

    def getPosition(self):
        return self.i_noteName.getPosition()


def createScaleNoteheadType():
    return it.chain(("whole",), it.repeat("black"))


class TL_StaffTest(Timeline):
    def construct(self):
        i_staff = Staff(**_staffConfig, staffLength=60)
        i_clef = Clef(i_staff)
        i_keySigSharps = KeySig(i_staff, 7)
        i_barline = Barline(i_staff, ":|.|:")
        i_keySigFlats = KeySig(i_staff, -7)
        i_chord = Chord(i_staff, (-9, -7, -6, -4, -2, 0), accis=(-1, None, 1, 1, 0, 2))
        i_scale = Scale(i_staff, np.arange(-6, 1))

        i_clef.setHpos(0.5)
        i_keySigSharps.after(i_clef, 1)
        i_barline.after(i_keySigSharps, 2)
        i_keySigFlats.after(i_barline, 1)
        i_chord.after(i_keySigFlats, 4)
        i_scale.after(i_chord, 4)
        i_staff.points.shift(-i_staff.i_staffLines.points.box.center)
        self.play(Create(i_staff.i_staffLines))
        self.play(
            Succession(
                Write(i_clef),
                Write(i_keySigSharps),
                Write(i_barline),
                Write(i_keySigFlats),
                Write(i_chord),
                Write(i_scale),
            )
        )
        self.forward(2)


class TL_PianoKeyboardTest(Timeline):
    def construct(self):
        i_keyboard = PianoKeyboard(keyRange=(-12, 36))
        i_keyboard.points.move_to(ORIGIN)
        self.play(Create(i_keyboard))
        self.play(i_keyboard.keys[0, 4, 7].anim.mark())
        self.play(i_keyboard.keys[4, 7].anim.unmark())
        self.play(i_keyboard.anim.unmark())
        self.forward(2)


class TL_Co5Intro(Timeline):
    CONFIG = Config(font=_textFont)

    def construct(self):
        NumberLine
        i_circ = NumberCircle(valueRange=(0, 12)).points.shift(UP * 0.25).r
        i_keyboard = (
            PianoKeyboard(keyRange=(0, 12)).points.to_center().shift(UP * 0.25).r
        )
        i_keyboard2 = (
            PianoKeyboard(keyRange=(0, 18)).points.to_center().shift(UP * 0.25).r
        )
        i_keyboard3 = (
            PianoKeyboard(keyRange=(0, 24)).points.to_center().shift((3.25, 1.75, 0)).r
        )

        # def createToneLabel(tone: int):
        #     isBlack = tone not in fz.MAJOR_SCALE_TONES
        #     i_label = Circle(radius=0.2).radius.set(0.015).r.fill.set(alpha=1).r
        #     if isBlack:
        #         i_label.stroke.set(color=WHITE).r.fill.set(color=BLACK)
        #     else:
        #         i_label.fill.set(color=WHITE).r.stroke.set(color=WHITE)
        #     return i_label

        i_ticks = i_circ.createTicks()
        i_labels = i_circ.addLabels()
        i_sectors = (
            Group(
                *(
                    i_circ.createSector(tone - 0.5, 1, add=False)
                    .fill.set(alpha=0.15)
                    .r.stroke.set(alpha=0)
                    .r
                    for tone in fz.MAJOR_SCALE_TONES
                )
            )
            .depth.set(3)
            .r
        )
        i_circ.add(i_sectors)
        i_rays = (
            Group(
                *(
                    i_circ.createRay(value, add=False)
                    .radius.set(0.01)
                    .r.stroke.set(alpha=0.25)
                    .r
                    for value in (np.arange(12) - 0.5)
                )
            )
            .depth.set(3)
            .r
        )
        i_circ.add(i_rays)

        self.play(
            Create(i_circ, auto_close_path=False, root_only=True, duration=1),
            AnimGroup(
                *(
                    FadeIn(Group(i_tick, i_label, i_ray))
                    for i_tick, i_label, i_ray in zip(i_ticks, i_labels, i_rays)
                ),
                lag_ratio=0.5,
                duration=1,
            ),
        )
        self.play(FadeIn(Group(i_keyboard, i_sectors)))
        # self.play(
        #     ShowPassingFlash(i_circ.createArc(0, 7).color.set(YELLOW).r, time_width=5),
        #     duration=2,
        # )
        ag = []
        for i in range(12):
            i_dot = (
                Dot(i_circ.n2p(i), radius=0.06)
                .glow.set(size=1, color=YELLOW, alpha=0.5)
                .r
            )
            ag.extend(
                (
                    i_keyboard.anim.unmark(),
                    i_keyboard.keys[i].anim.mark(),
                    FadeIn(i_dot),
                )
            )
            self.play_audio(midi2Audio(i))
            self.play(*ag, duration=0.25)
            self.forward(0.25)
            ag.clear()
            ag.append(FadeOut(i_dot))
        self.play(
            Transform(i_keyboard.i_octaves[0], i_keyboard2.i_octaves[0]),
            FadeIn(i_keyboard2.i_octaves[-1]),
        )
        i_dot = (
            Dot(i_circ.n2p(0), radius=0.06).glow.set(size=1, color=YELLOW, alpha=0.5).r
        )
        self.play(
            i_keyboard2.keys[0, 12].anim.mark(),
            FadeIn(i_dot),
            *ag,
            duration=0.25,
        )
        self.forward(1)
        self.play(FadeOut(i_dot), duration=0.2)

        self.forward(2)

        i_staff = (
            Staff(**_staffConfig, staffLength=32)
            .points.to_center()
            .shift((3.25, -0.5, 0))
            .r
        )
        i_clef = Clef(i_staff).setHpos(0.5)
        i_scale = Scale.fromNotation(
            i_staff,
            fz.Scale("C", fz.Modes.MAJOR),
            buff=2,
            acciSpaceRatio=0,
            noteheadType=createScaleNoteheadType(),
        ).shiftHpos(8)

        self.play(
            i_circ.anim.points.shift(LEFT * 3),
            FadeOut(i_keyboard2),
            FadeIn(i_keyboard3),
            FadeIn(Group(i_staff.i_staffLines, i_clef)),
        )
        keyNamePos = (0.8, -2, 0)
        scaleTextPos = (2.4, -2, 0)
        scale = fz.Scale("C", fz.Modes.MAJOR)
        i_keyName = MajorName(fz.OPitch("C")).setPosition(keyNamePos)
        i_scaleText = ScaleText(scale, color=colorByAcci()).setPosition(scaleTextPos)
        i_segs = Group[Line]()
        i_degs = Group[TypstMath]()
        i_dot = (
            Dot(i_circ.n2p(0), radius=0.06).glow.set(size=1, color=YELLOW, alpha=0.5).r
        )
        self.play(Write(i_keyName), duration=0.5)
        for i, (tone1, tone2) in enumerate(
            it.pairwise(it.chain(fz.MAJOR_SCALE_TONES, (12,)))
        ):
            diff = tone2 - tone1
            point1 = i_circ.n2p(tone1)
            i_note = i_scale[i]
            i_noteName = i_scaleText[i]
            i_deg = (
                TypstMath(f"hat({i + 1})")(VItem)
                .set_stroke_background(True)
                .points.scale(1.25)
                .r.fill.set(color=WHITE, alpha=1)
                .r.stroke.set(color=BLACK, alpha=0.75)
                .r.radius.set(0.05)
                .r.depth.set(-3)
                .r
            )
            i_seg = Line(point1, i_circ.n2p(tone2)).stroke.set(PINK).r.depth.set(-2).r
            i_arc = i_circ.createArc(tone1, diff)
            i_circ.addLabel(tone1, i_deg, buff=0.25, side=INSIDE, add=False)
            i_segs.add(i_seg)
            i_degs.add(i_deg)

            self.play_audio(midi2Audio(tone1))
            self.play(
                FadeIn(Group(i_deg, i_dot) if i == 0 else i_deg),
                FadeIn(Group(i_note, i_noteName)),
                Flash(point1),
                i_keyboard3.keys[tone1].anim.mark(),
                duration=0.5,
            )
            self.play(
                Create(i_seg, duration=0.5),
                ShowPassingFlash(i_arc, time_width=3, duration=1),
            )

        i_degsCo5 = Group[TypstMath](*(i_degs[int(deg)] for deg in fz.STEPS_CO5))
        self.forward(2)

        currentTone = 0

        def animateKeyChange(co5Order: int = 0, playScale: bool = True):
            nonlocal currentTone, scale, i_keyName, i_scaleText, i_scale
            newTonic = fz.OPitch.co5(co5Order)
            newScale = fz.Scale(newTonic, fz.Modes.MAJOR)
            if co5Order > 0:
                alteredDegs = np.arange(6 - co5Order, 6) * 4 % 7
                alterColor = RED
            else:
                alteredDegs = np.arange(-2 - co5Order, -2, -1) * 4 % 7
                alterColor = GREEN

            i_newKeyName = MajorName(newTonic).setPosition(keyNamePos)
            i_newScaleText = ScaleText(newScale, color=colorByAcci()).setPosition(
                scaleTextPos
            )
            i_keySig = KeySig(i_staff, co5Order).after(i_clef, 2)

            markColors = (
                MARK_BLUE if p.acci == 0 else MARK_RED if p.acci > 0 else MARK_GREEN
                for p in newScale
            )

            tone = newTonic.tone
            diff = modShift(tone - currentTone, 12, 6)
            rotation = i_circ.n2a(diff, diff=True)
            vpos = np.arange(7) + (newTonic.step - 6)
            i_newScale = Scale.fromNotation(
                i_staff,
                newScale,
                buff=2,
                acciSpaceRatio=0,
                color=colorByAcci(),
                noteheadType=createScaleNoteheadType(),
            ).shiftHpos(8)
            i_newScale2 = Scale(
                i_staff,
                vpos,
                buff=2,
                acciSpaceRatio=0,
                noteheadType=createScaleNoteheadType(),
            ).shiftHpos(8)
            i_newScale3 = Scale(
                i_staff, vpos, buff=1.25, noteheadType=createScaleNoteheadType()
            ).after(i_keySig, 2)

            for deg in alteredDegs:
                for i_ in i_newScale2, i_newScale3:
                    i_[int(deg)].i_notehead.color.set(alterColor)

            def createDegUpdater(i: int):
                i_deg = i_degs[i]
                pos = currentTone

                def updaterFn(data: TypstMath, p: UpdaterParams):
                    tone = p.alpha * diff + pos + fz.MAJOR_SCALE_TONES[i]
                    i_circ.addLabel(tone, data, buff=0.25, side=INSIDE, add=False)

                updater = GroupUpdater(i_deg, updaterFn)

                return updater

            if currentTone != 0:
                self.prepare(
                    *(i_(VItem).update.fill.set(color=WHITE) for i_ in i_degs),
                    duration=0.5,
                )

            # 多边形旋转到对应的调
            self.play(
                Rotate(
                    Group(i_segs, i_dot),
                    angle=rotation,
                    about_point=i_circ.getPosition(),
                ),
                *(createDegUpdater(i) for i in range(len(fz.MAJOR_SCALE_TONES))),
                Transform(i_scale, i_newScale),
                Transform(i_keyName, i_newKeyName),
                Transform(i_scaleText, i_newScaleText),
                i_keyboard3.anim.setMarks(
                    fz.MAJOR_SCALE_TONES + tone, color=markColors
                ),
                duration=2,
            )

            if playScale:
                self.forward(0.5)
                for i in fz.MAJOR_SCALE_TONES:
                    self.play_audio(midi2Audio(i + tone))
                    self.prepare(Flash(i_circ.n2p(i + tone)), duration=0.5)
                    self.forward(0.25)
                self.forward(1.25)

            if co5Order != 0:
                # 闪动强调所有的临时变音记号
                i_accis = Group(*(i_newScale[int(deg)].i_acci for deg in alteredDegs))
                self.play(
                    AnimGroup(
                        *(
                            AnimGroup(
                                Indicate(i_accis[i], scale_factor=1.5),
                                Flash(i_circ.n2p(newScale[deg].otone)),
                                i_degs[int(deg)](VItem).anim.fill.set(alterColor),
                                duration=0.5,
                            )
                            for i, deg in enumerate(alteredDegs)
                        ),
                        lag_ratio=0.75,
                    ),
                )
                self.forward(0.5)

                # 临时变音记号变成调号
                self.hide(i_newScale)
                self.show(i_newScale2)
                self.play(
                    *(Transform(i_s, i_t) for i_s, i_t in zip(i_accis, i_keySig)),
                    Transform(i_newScale2, i_newScale3),
                )
                self.play(ShowPassingFlashAround(i_keySig, time_width=3), duration=1.5)
                self.forward(1)

                # 还原回临时变音记号，方便下一步演示
                self.play(
                    *(Transform(i_s, i_t) for i_s, i_t in zip(i_keySig, i_accis)),
                    Transform(i_newScale3, i_newScale2),
                    duration=0.5,
                )
                self.hide(i_newScale2)
                self.show(i_newScale)
            else:
                self.forward(1)

            i_scaleText = i_newScaleText
            i_scale = i_newScale
            i_keyName = i_newKeyName
            currentTone = tone
            scale = newScale

        animateKeyChange(2)
        animateKeyChange(0, False)
        self.forward(2)
        animateKeyChange(-3)
        animateKeyChange(4)
        animateKeyChange(-1)
        animateKeyChange(0, False)
        self.play(
            FadeOut(Group(i_segs, i_scale, i_scaleText)),
            i_keyboard3.anim.unmark(),
        )

        # 按照五度圈顺序生成大调音阶
        i_co5 = Group[Line]()
        tone2 = fz.MAJOR_SCALE_TONES_CO5[0]
        deg = (-1) * 4 % 7
        point2 = i_circ.n2p(tone2)
        self.forward(1)
        self.play_audio(midi2Audio(tone2))
        self.play(
            Flash(point2),
            i_keyboard3.keys[int(tone2)].anim(duration=0.25).mark(),
            FadeIn(Group(i_scaleText[deg], i_scale[deg]), duration=0.25),
        )
        self.forward(0.5)

        for i, (tone1, tone2) in enumerate(it.pairwise(fz.MAJOR_SCALE_TONES_CO5)):
            deg = i * 4 % 7
            point1, point2 = i_circ.n2p(tone1), i_circ.n2p(tone2)
            i_line = Line(point1, point2).stroke.set(color=PINK).r.depth.set(-2).r
            i_co5.add(i_line)
            self.play_audio(midi2Audio(tone2), delay=1)
            self.prepare(
                Flash(point2),
                i_keyboard3.keys[int(tone2)].anim(duration=0.25).mark(),
                FadeIn(Group(i_scaleText[deg], i_scale[deg]), duration=0.25),
                at=1,
            )
            self.play(
                Create(i_line, duration=1),
                ShowPassingFlash(i_circ.createArc(tone1, 7), time_width=3, duration=2),
            )
            self.forward(0.5)
        self.forward(2)

        def createLabelUpdater(i: int):
            i_label = i_labels[i]

            def updaterFn(data: TypstMath, p: UpdaterParams):
                t = 1 + p.alpha * 6
                i_circ.addLabel(i * t, data, add=False)

            return GroupUpdater(i_label, updaterFn)

        def createDegUpdater(i: int):
            i_deg = i_degs[i]
            tone = fz.MAJOR_SCALE_TONES[i]

            def updateFn(data: TypstMath, p: UpdaterParams):
                t = 1 + p.alpha * 6
                i_circ.addLabel(tone * t, data, buff=0.25, side=INSIDE, add=False)

            return GroupUpdater(i_deg, updateFn)

        def createLineUpdater(i: int):
            i_line = i_co5[i]

            def updateFn(data: Line, p: UpdaterParams):
                t = 1 + p.alpha * 6
                tone1 = (i - 1) * 7 % 12
                tone2 = i * 7 % 12
                point1 = i_circ.n2p(tone1 * t)
                point2 = i_circ.n2p(tone2 * t)
                data.points.put_start_and_end_on(point1, point2)

            return DataUpdater(i_line, updateFn)

        # 从半音圈变换到五度圈
        self.play(
            *(createLabelUpdater(i) for i in range(1, 12)),
            *(createDegUpdater(i) for i in range(7)),
            *(createLineUpdater(i) for i in range(6)),
            *(
                Rotate(
                    i_tick,
                    angle=i_circ.n2a(6, diff=True) * i,
                    about_point=i_circ.getPosition(),
                )
                for i, i_tick in enumerate(i_ticks)
            ),
            *(
                Rotate(
                    i_sector,
                    angle=i_circ.n2a(6, diff=True) * i,
                    about_point=i_circ.getPosition(),
                )
                for i, i_sector in zip(fz.MAJOR_SCALE_TONES, i_sectors)
            ),
            duration=5,
        )
        self.forward(2)
        self.play(
            FadeOut(Group(i_co5, i_degs, i_scaleText, i_scale)),
            i_keyboard3.anim.unmark(),
            duration=0.5,
        )

        # 出示五度圈的前 5 个音，构成大调五声音阶
        pentatonicDegsCo5 = fz.STEPS_CO5[1:6]
        pentatonicDegs = np.sort(pentatonicDegsCo5)
        pentatonicTones = fz.MAJOR_SCALE_TONES[pentatonicDegs]
        pentatonicNames = ("宫", "商", "角", "徵", "羽")
        i_pentatonicScale = Group(*(i_scale[int(deg)] for deg in pentatonicDegs))
        i_pentatonicNames = Group(
            *(
                Text(pentatonicNames[i])
                .points.next_to(i_scaleText[int(deg)], DOWN, buff=0.25)
                .r
                for i, deg in enumerate(pentatonicDegs)
            )
        )

        # i_pentatonicDegs = Group(*(i_degs[int(deg)] for deg in pentatonicDegsCo5))
        i_pentatonicLines = i_co5[1:5]
        i_nonPentatonicLines = Group(i_co5[0], i_co5[-1])

        deg = fz.STEPS_CO5[1]
        tone = fz.MAJOR_SCALE_TONES_CO5[1]
        self.play_audio(midi2Audio(tone), delay=0.5)
        self.play(
            FadeIn(
                Group(
                    i_degs[int(deg)],
                    i_scale[int(deg)],
                    i_scaleText[int(deg)],
                )
            ),
            i_keyboard3.keys[int(tone)].anim.mark(),
            duration=0.5,
        )
        self.forward(0.5)

        i_pentatonicScaleText = Group(
            *(i_scaleText[int(deg)] for deg in pentatonicDegs)
        )
        for i, deg in enumerate(fz.STEPS_CO5[2:6]):
            tone = fz.MAJOR_SCALE_TONES_CO5[i + 2]
            self.play_audio(midi2Audio(tone), delay=0.5)
            self.play(
                FadeIn(
                    Group(
                        i_degs[int(deg)],
                        i_scale[int(deg)],
                        i_scaleText[int(deg)],
                    )
                ),
                Create(i_co5[i + 1]),
                i_keyboard3.keys[int(tone)].anim.mark(),
                duration=0.5,
            )
            self.forward(0.5)
        self.play(FadeIn(i_pentatonicNames))

        def animatePentatonicChange(co5Order: int, playScale: bool = True):
            nonlocal \
                i_scale, \
                i_pentatonicScale, \
                i_keyName, \
                i_pentatonicScaleText, \
                currentTone

            newTonic = fz.OPitch.co5(co5Order)
            newScale = fz.Scale(newTonic, fz.Modes.MAJOR)
            tone = newTonic.tone

            if tone != currentTone:
                diff = modShift(tone - currentTone, 12, 6)
                angle = i_circ.n2a(diff, diff=True)
                markColors = (
                    MARK_RED if p.acci > 0 else MARK_GREEN if p.acci < 0 else MARK_BLUE
                    for p in newScale.pitches[pentatonicDegs]
                )

                i_newKeyName = MajorName(newTonic).setPosition(keyNamePos)
                i_newScaleText = ScaleText(newScale, color=colorByAcci()).setPosition(
                    scaleTextPos
                )
                i_newPentatonicScaleText = Group(
                    *(i_newScaleText[int(deg)] for deg in pentatonicDegs)
                )

                def createDegUpdater(i: int) -> GroupUpdater:
                    i_deg = i_degsCo5[i]
                    ctone = currentTone

                    def updateFn(data: TypstMath, p: UpdaterParams):
                        tone = p.alpha * diff + i - 1 + ctone
                        i_circ.addLabel(tone, data, buff=0.25, side=INSIDE, add=False)

                    return GroupUpdater(i_deg, updateFn)

                i_newScale = Scale.fromNotation(
                    i_staff,
                    newScale,
                    buff=2,
                    acciSpaceRatio=0,
                    color=colorByAcci(),
                    noteheadType=createScaleNoteheadType(),
                ).shiftHpos(8)
                i_newPentatonicScale = Group(
                    *(i_newScale[int(i)] for i in pentatonicDegs)
                )
                self.play(
                    Transform(i_pentatonicScale, i_newPentatonicScale),
                    Transform(i_pentatonicScaleText, i_newPentatonicScaleText),
                    Transform(i_keyName, i_newKeyName),
                    Rotate(
                        Group(i_pentatonicLines, i_dot),
                        about_point=i_circ.getPosition(),
                        angle=angle,
                    ),
                    *(createDegUpdater(i) for i in range(1, 6)),
                    i_keyboard3.anim.setMarks(
                        fz.MAJOR_SCALE_TONES_CO5[1:6] + tone, markColors
                    ),
                    duration=2,
                )
                for i in (0, 6):
                    i_circ.addLabel(
                        diff + i + currentTone - 1,
                        i_degsCo5[i],
                        buff=0.25,
                        side=INSIDE,
                        add=False,
                    )
                i_nonPentatonicLines.points.rotate(
                    angle, about_point=i_circ.getPosition()
                )

            if playScale:
                self.forward(0.5)
                for i in pentatonicTones:
                    self.play_audio(midi2Audio(i + tone))
                    self.prepare(Flash(i_circ.n2p((i + tone) * 7 % 12)), duration=0.5)
                    self.forward(0.25)
                self.forward(1.25)

            if tone != currentTone:
                i_pentatonicScale = i_newPentatonicScale
                i_pentatonicScaleText = i_newPentatonicScaleText
                i_scale = i_newScale
                i_keyName = i_newKeyName
                currentTone = tone

        i_co5[0].points.reverse()

        def animateNonPentatonic(hide: bool = False, succession: bool = True):
            ag = []
            for i in (0, -1):
                deg = fz.STEPS_CO5[i]
                tone = fz.MAJOR_SCALE_TONES_CO5[i]
                i_line = i_co5[i]

                ag.append(
                    AnimGroup(
                        (Uncreate if hide else Create)(i_line),
                        (FadeOut if hide else FadeIn)(
                            Group(
                                i_degs[int(deg)],
                                i_scale[int(deg)],
                                i_scaleText[int(deg)],
                            )
                        ),
                        getattr(
                            i_keyboard3.keys[int(tone)].anim,
                            "unmark" if hide else "mark",
                        )(),
                    )
                )
            if not hide:
                self.play_audio(midi2Audio(5), delay=0.5)
                self.play_audio(midi2Audio(11), delay=1.5)
            self.play((Succession if succession else AnimGroup)(*ag))

        animateNonPentatonic()
        self.forward(2)
        animateNonPentatonic(hide=True, succession=False)
        animatePentatonicChange(0)
        animatePentatonicChange(-6)
        self.forward(2)
        animatePentatonicChange(0, False)
        self.forward(3)


class TL_MajorTriad(Timeline):
    CONFIG = Config(font=_textFont)

    def construct(self):
        i_staff = (
            Staff(**_staffConfig, staffLength=15).points.to_center().shift(0.5 * UP).r
        )
        i_staff2 = (
            Staff(**_staffConfig, staffLength=30).points.to_center().shift(0.5 * UP).r
        )
        i_clef = Clef(i_staff)
        i_clef2 = Clef(i_staff2)
        i_chord = Chord(i_staff, (-6, -4, -2), noteheadType="whole")
        i_chord2 = Chord(i_staff2, (-6, -4, -2), noteheadType="whole")

        i_scale = Scale(
            i_staff2,
            (-6, -4, -2),
            buff=3,
            noteheadType=it.chain(("whole",), it.repeat("black")),
        )

        i_clef.setHpos(0.5)
        i_clef2.setHpos(0.5)
        i_chord.after(i_clef, 4)
        i_chord2.after(i_clef2, 4)
        i_scale.after(i_chord2, 5)
        self.play(Create(i_staff.i_staffLines), FadeIn(i_clef))
        self.play_audio(midi2Audio(("C_0", "E_0", "G_0")), delay=0.5)
        self.play(Write(i_chord))
        self.forward(2)
        self.play(
            Transform(i_staff.i_staffLines, i_staff2.i_staffLines),
            Transform(i_clef, i_clef2),
            Transform(i_chord, i_chord2),
            FadeIn(i_scale),
        )
        colors = (RED, GREEN, BLUE)
        noteNames = ("C", "E", "G")
        ratio = (4, 5, 6)

        for mob_note, color, notename, ratioNumber in zip(
            i_scale, colors, noteNames, ratio
        ):
            i_noteName = (
                # fmt: off
                Text(notename)
                .points.move_to(mob_note.i_noteheads[0])
                .next_to(i_staff2.i_staffLines, DOWN, buff=0.5, coor_mask=UP)
                .r
                # fmt: on
            )
            i_ratioNumber = (
                # fmt: off
                Text(str(ratioNumber)).points.next_to(i_noteName, DOWN, buff=0.25).r
                # fmt: on
            )
            self.play_audio(midi2Audio(notename), delay=0.25)
            self.play(
                mob_note.i_noteheads[0].anim.fill.set(color=color),
                Write(i_noteName),
                Write(i_ratioNumber),
                duration=0.5,
            )
        self.forward(3)


class TL_ScaleTranslation(Timeline):
    CONFIG = Config(font=_textFont)

    def construct(self):
        tonic = fz.OPitch("C")
        newTonic = fz.OPitch("A")
        scale = fz.Scale(tonic, fz.Modes.MAJOR)
        newScale = fz.Scale(newTonic, fz.Modes.MAJOR)
        i_staff = (
            Staff(**_staffConfig, staffLength=32).points.to_center().shift(0.5 * UP).r
        )
        i_keyName = (
            MajorName(tonic).points.next_to(i_staff.i_staffLines, DOWN, buff=0.5).r
        )
        i_newKeyName = MajorName(newTonic).setPosition(i_keyName.getPosition())
        i_clef = Clef(i_staff).setHpos(0.5)
        i_scale1 = Scale.fromNotation(
            i_staff,
            scale,
            buff=2,
            acciSpaceRatio=0,
            color=colorByAcci(),
            noteheadType=createScaleNoteheadType(),
        ).shiftHpos(8)
        i_scale2 = Scale.fromNotation(
            i_staff,
            newScale,
            buff=2,
            acciSpaceRatio=0,
            color=colorByAcci(),
            noteheadType=createScaleNoteheadType(),
        ).shiftHpos(8)
        self.play(Create(i_staff.i_staffLines), FadeIn(i_clef))
        self.play(Write(i_scale1), Write(i_keyName), duration=1)
        self.forward(2)
        self.play(
            Transform(i_scale1, i_scale2),
            Transform(i_keyName, i_newKeyName),
            duration=1,
        )
        self.forward(2)


class TL_Accidentals(Timeline):
    CONFIG = Config(font=_textFont)

    def construct(self):
        i_keyboard = PianoKeyboard(keyRange=(-12, 36))
        i_staff = Staff(**_staffConfig, staffLength=66)

        i_keyboard.points.to_center().shift(1.5 * UP)
        i_staff.points.to_center().shift(1.25 * DOWN)

        notes = (
            fz.Pitch("C_0"),
            fz.Pitch("C+_0"),
            fz.Pitch("D_0"),
            fz.Pitch("D-_0"),
            fz.Pitch("G_0"),
            fz.Pitch("G+_0"),
            fz.Pitch("A_0"),
            fz.Pitch("A-_0"),
        )
        notes2 = (
            fz.Pitch("F_0"),
            fz.Pitch("E+_0"),
            fz.Pitch("E_0"),
            fz.Pitch("F-_0"),
            fz.Pitch("C_1"),
            fz.Pitch("B+_0"),
            fz.Pitch("B_0"),
            fz.Pitch("C-_1"),
        )
        i_clef = Clef(i_staff).setHpos(0.5)

        def CreateNotesAndNames(notes: Iterable[fz.PitchBase]):
            i_notes = Scale.fromNotation(
                i_staff,
                notes,
                color=colorByAcci(),
                buff=it.cycle((4, 7)),
                acciSpaceRatio=0,
            ).shiftHpos(12)
            i_noteNames = Group(
                *(NoteName(p.opitch, color=colorByAcci()) for p in notes)
            )
            for i_noteName, i_note in zip(i_noteNames, i_notes):
                x = i_note.getPosition()[0]
                y = i_staff.getPosition()[1]
                i_noteName.setPosition((x - 0.25, y + 0.5, 0))
            return i_notes, i_noteNames

        i_notes, i_noteNames = CreateNotesAndNames(notes)
        i_notes2, i_noteNames2 = CreateNotesAndNames(notes2)

        self.play(FadeIn(i_keyboard), Create(i_staff.i_staffLines), FadeIn(i_clef))
        for i_noteName, i_note in zip(i_noteNames, i_notes):
            x = i_note.getPosition()[0]
            y = i_staff.getPosition()[1]
            i_noteName.setPosition((x - 0.25, y + 0.5, 0))
        self.forward(1)
        for (note1, note2), (i_note1, i_note2), (i_noteName1, i_noteName2) in zip(
            it.batched(notes, 2), it.batched(i_notes, 2), it.batched(i_noteNames, 2)
        ):
            acciMarkColor = MARK_RED if note2.acci > 0 else MARK_GREEN
            self.play_audio(midi2Audio(note1), delay=0.25)
            self.play(
                i_keyboard.keys[note1].anim.mark(),
                Write(i_note1),
                Write(i_noteName1),
                duration=0.5,
            )
            self.forward(0.5)
            self.play_audio(midi2Audio(note2), delay=0.25)
            self.play(
                i_keyboard.keys[note2].anim.mark(acciMarkColor),
                Write(i_note2),
                Write(i_noteName2),
                duration=0.5,
            )
            self.forward(0.5)
            self.play(Indicate(i_note2.i_acci, scale_factor=1.5), duration=0.75)
            self.forward(1)
            self.play(i_keyboard.anim.unmark(), duration=0.5)
        self.play(FadeOut(Group(i_notes, i_noteNames)))
        self.forward(0.5)
        self.play(Write(i_notes2), Write(i_noteNames2))
        self.play(
            i_keyboard.keys[4, 11].anim.mark(MARK_GREEN),
            i_keyboard.keys[5, 12].anim.mark(MARK_RED),
            duration=0.5,
        )
        self.forward(2)


class TL_KeySigAccis(Timeline):
    CONFIG = Config(font=_textFont)

    def construct(self):
        i_noteNames = Group[NoteName]()
        for i in range(-1, 6):
            i_noteName = NoteName(
                fz.OPitch.co5(i), fs=DEFAULT_FONT_SIZE * 2
            ).setPosition((i, 0, 0))
            i_noteNames.add(i_noteName)
        i_noteNames.points.to_center()
        i_upperArrow = (
            Arrow(ORIGIN, RIGHT * i_noteNames.points.box.width, color=RED, buff=0)
            .points.next_to(i_noteNames, UP, buff=0.25)
            .r
        )
        i_lowerArrow = (
            Arrow(ORIGIN, LEFT * i_noteNames.points.box.width, color=GREEN, buff=0)
            .points.next_to(i_noteNames, DOWN, buff=0.25)
            .r
        )
        i_sharp = (
            MusicGlyph("accidentalSharp", font=_musicFont, sp=0.3)
            .setPosition(i_upperArrow.points.get_start())
            .points.next_to(i_upperArrow, LEFT, buff=0.25, coor_mask=RIGHT)
            .r
        )
        i_flat = (
            MusicGlyph("accidentalFlat", font=_musicFont, sp=0.3)
            .setPosition(i_lowerArrow.points.get_start())
            .points.next_to(i_lowerArrow, RIGHT, buff=0.25, coor_mask=RIGHT)
            .r
        )
        self.play(Write(i_noteNames), duration=1)
        self.forward(1)
        self.play(Create(i_upperArrow), Write(i_sharp))
        self.forward(1)
        self.play(Create(i_lowerArrow), Write(i_flat))
        self.forward(2)
        i_staff = (
            Staff(**_staffConfig, staffLength=48)
            .points.to_center()
            .shift(DOWN * 1.25)
            .r
        )
        i_clef = Clef(i_staff).setHpos(0.5)
        i_sharpKey = KeySig(i_staff, 7).setHpos(6)
        i_flatKey = KeySig(i_staff, -7).setHpos(28)
        self.play(
            Group(
                i_upperArrow, i_lowerArrow, i_sharp, i_flat, i_noteNames
            ).anim.points.shift(UP * 1.25),
            Create(i_staff.i_staffLines),
            FadeIn(i_clef),
        )
        for i_key, order in zip((i_sharpKey, i_flatKey), (1, -1)):
            color = RED if order > 0 else GREEN
            self.play(Write(i_key))
            self.forward(1)
            self.play(
                AnimGroup(
                    *(
                        AnimGroup(
                            Indicate(i_acci, scale_factor=1.5),
                            Indicate(
                                i_noteNames[i if order > 0 else -i - 1],
                                scale_factor=1.25,
                                color=color,
                            ),
                            duration=1,
                        )
                        for i, i_acci in enumerate(i_key)
                    ),
                    lag_ratio=0.5,
                ),
            )
            self.forward(1)
        self.forward(2)
        self.play(FadeOut(Group(i_sharpKey, i_flatKey)))

        def animateFindKeySig(co5Order: int, inverse: bool = False):
            tonic = fz.OPitch.co5(co5Order)
            scale = fz.Scale(tonic, fz.Modes.MAJOR)
            alteredPitch: fz.OPitch = scale[-1 if co5Order > 0 else 3]
            if co5Order > 0:
                alteredDegs = np.arange(6 - co5Order, 6) * 4 % 7
                alterColor = RED
                noteNameIdx = co5Order - 1
            else:
                alteredDegs = np.arange(-2 - co5Order, -2, -1) * 4 % 7
                alterColor = GREEN
                noteNameIdx = co5Order

            i_tonic = Note.fromNotation(i_staff, tonic).setHpos(16)
            i_altered = Note.fromNotation(
                i_staff, alteredPitch, color=alterColor
            ).setHpos(32)
            i_alteredNoteName = i_noteNames[noteNameIdx]
            i_alteredNoteNames = (
                i_noteNames[:co5Order] if co5Order > 0 else i_noteNames[co5Order:]
            )
            i_keyName = (
                MajorName(tonic)
                .setPosition(i_staff.i_staffLines.points.box.get(UR))
                .points.shift((-1, 0.4, 0))
                .r
            )
            i_scale = Scale.fromNotation(
                i_staff,
                scale,
                color=colorByAcci(),
                buff=4,
                acciSpaceRatio=0,
                noteheadType=createScaleNoteheadType(),
            ).shiftHpos(10)
            i_scale2 = Scale(
                i_staff,
                np.arange(7) + (tonic.step - 6),
                buff=4,
                acciSpaceRatio=0,
                noteheadType=createScaleNoteheadType(),
            ).shiftHpos(10)
            i_scale3 = Scale(
                i_staff,
                np.arange(7) + (tonic.step - 6),
                buff=3,
                acciSpaceRatio=0,
                noteheadType=createScaleNoteheadType(),
            ).shiftHpos(15)
            i_keySig = KeySig(i_staff, co5Order).setHpos(4)
            i_accis = Group(*(i_scale[int(deg)].i_acci for deg in alteredDegs))

            for deg in alteredDegs:
                for i_ in i_scale2, i_scale3:
                    i_[int(deg)].i_notehead.color.set(alterColor)

            if inverse:
                self.play(Write(i_keySig))
                i_whatKey = (
                    Text("?? 调")
                    .points.next_to(i_keySig, UP, buff=0.25)
                    .r.fill.set(YELLOW)
                    .r
                )
                self.prepare(FadeOut(i_whatKey, duration=0.5), at=1.5)
                self.play(
                    ShowPassingFlashAround(i_keySig, time_width=5),
                    FadeIn(i_whatKey, duration=0.5),
                    duration=2,
                )
                self.forward(1)
                i_altered.setHpos(36)
                i_tonic.setHpos(20)
                self.play(Indicate(i_keySig[-1], scale_factor=1.5))
                self.play(
                    # FadeOut(i_keySig),
                    FadeIn(i_altered),
                    i_alteredNoteName.anim.fill.set(color=alterColor),
                )
                self.forward(1)
                self.play(
                    Transform(i_altered, i_tonic, hide_src=False), FadeIn(i_keyName)
                )
                self.forward(1)
                self.play(
                    FadeOut(Group(i_altered, i_tonic), duration=0.5),
                    Write(i_scale3, duration=1),
                    *(i_.anim.fill.set(alterColor) for i_ in i_alteredNoteNames),
                )
                self.play(
                    ShowPassingFlashAround(
                        i_alteredNoteNames,
                        time_width=5,
                    ),
                    duration=2,
                )
                self.forward(1)
                self.play(
                    *(Transform(i_s, i_t) for i_s, i_t in zip(i_keySig, i_accis)),
                    Transform(i_scale3, i_scale2),
                )
                self.hide(i_scale2)
                self.show(i_scale)
                self.forward(2)
                self.play(
                    FadeOut(Group(i_scale, i_keyName)),
                    *(
                        i_noteName.anim.fill.set(color=WHITE)
                        for i_noteName in i_noteNames
                    ),
                )
            else:
                self.play(FadeIn(Group(i_tonic, i_keyName)), duration=0.5)
                self.forward(1)
                self.play(Transform(i_tonic, i_altered, hide_src=False))
                self.play(
                    Indicate(i_altered.i_acci, scale_factor=1.5),
                    i_alteredNoteName.anim.fill.set(color=alterColor),
                )
                self.forward(1)
                self.play(
                    FadeOut(Group(i_tonic, i_altered)),
                    i_alteredNoteName.anim.fill.set(color=WHITE),
                )
                self.play(Write(i_scale))
                self.hide(i_scale)
                self.show(i_scale2)
                self.play(
                    *(Transform(i_s, i_t) for i_s, i_t in zip(i_accis, i_keySig)),
                    Transform(i_scale2, i_scale3),
                )
                self.play(
                    AnimGroup(
                        *(
                            AnimGroup(
                                Indicate(i_acci, scale_factor=1.5, duration=0.5),
                                i_noteName.anim(duration=0.5).fill.set(
                                    color=alterColor
                                ),
                            )
                            for i_acci, i_noteName in zip(
                                i_keySig,
                                i_noteNames if co5Order > 0 else reversed(i_noteNames),
                            )
                        ),
                        lag_ratio=0.5,
                    )
                )

                self.play(
                    ShowPassingFlashAround(
                        i_alteredNoteNames,
                        time_width=5,
                    ),
                    ShowPassingFlashAround(i_keySig, time_width=5),
                    duration=2,
                )
                self.forward(2)
                self.play(
                    FadeOut(Group(i_scale3, i_keySig, i_keyName)),
                    *(
                        i_noteName.anim.fill.set(color=WHITE)
                        for i_noteName in i_noteNames
                    ),
                )

        animateFindKeySig(4)
        animateFindKeySig(2)
        animateFindKeySig(5, True)
        # animateFindKeySig(-2)
        # animateFindKeySig(-4)
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])


################################################################################
################################################################################
#########################################
