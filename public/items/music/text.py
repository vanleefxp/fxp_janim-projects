from pathlib import Path

from janim.imports import *
import pyrsistent as pyr
import fantazia as fz

from ... import PositionedVItem, createEmptyDot

__all__ = ["MusicGlyph", "NoteName", "ScaleText", "colorByAcci"]

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
MUSIC_FONT_SIZE_BASE = 270
_smuflName2Codepoint = None
_defaultMusicFonts = ("Leland", "Bravura")
_accidentalGlyphNames = pyr.pmap(
    {
        3: "accidentalTripleSharp",
        2: "accidentalDoubleSharp",
        1.5: "accidentalThreeQuarterTonesSharpStein",
        1: "accidentalSharp",
        0.5: "accidentalQuarterToneSharpStein",
        0: "accidentalNatural",
        -0.5: "accidentalQuarterToneFlatStein",
        -1: "accidentalFlat",
        -1.5: "accidentalThreeQuarterTonesFlatStein",
        -2: "accidentalDoubleFlat",
        -3: "accidentalTripleFlat",
    }
)


def _loadSmufl():
    import json

    global _smuflName2Codepoint, _loadSmufl
    jsonSrc = json.loads((DIR / "assets/glyphnames.json").read_text(encoding="UTF-8"))
    _smuflName2Codepoint = pyr.pmap(
        {k: chr(int(v["codepoint"][2:], base=16)) for k, v in jsonSrc.items()}
    )
    _loadSmufl = lambda: None


def _collectFonts(
    font: str | Iterable[str],
    weight: Weight | WeightName = "regular",
    style: Style | StyleName = "normal",
):
    if isinstance(font, str):
        font_names = [font]
    else:
        font_names = list(font)

    cfg_font = Config.get.font
    if isinstance(cfg_font, str):
        font_names.append(cfg_font)
    else:
        font_names.extend(cfg_font)
    fonts = [
        Font.get_by_info(get_font_info_by_attrs(name, weight, style))
        for name in font_names
    ]
    return fonts


def colorByAcci(sharpColor: JAnimColor = RED, flatColor: JAnimColor = GREEN):
    def _colorByAcci(pitch: fz.Pitch):
        if pitch.acci > 0:
            return sharpColor
        elif pitch.acci < 0:
            return flatColor
        else:
            return None

    return _colorByAcci


class MusicGlyph(TextChar):
    def __init__(
        self,
        char: int | str,
        font: str | Iterable[str] = _defaultMusicFonts,
        sp: float = 0.2,
        fs: float | None = None,
        stroke_alpha=0,
        **kwargs,
    ):
        fonts = _collectFonts(font)

        # get SMuFL character
        if isinstance(char, int):  # codepoint
            char = chr(char)
        elif isinstance(char, str):
            if len(char) > 1:  # multi-character string = SMuFL glyph name
                _loadSmufl()
                glyphName = char
                char = _smuflName2Codepoint.get(glyphName)
                if char is None:
                    raise ValueError(f"Invalid SMuFL glyph name: {glyphName}")
        else:
            raise ValueError(f"Invalid char: {char}")

        # font size
        if fs is None:
            fs = sp * MUSIC_FONT_SIZE_BASE

        super().__init__(char, fonts, fs, stroke_alpha=stroke_alpha, **kwargs)

    def getPosition(self) -> Vect:
        return self.get_mark_orig()

    def setPosition(self, p: Vect) -> Self:
        self.points.shift(np.array(p) - self.getPosition())
        return self


class NoteName(Group[Text | TextChar], PositionedVItem):
    def __init__(
        self,
        pitch: fz.PitchBase,
        acciPosition: int = -1,
        showNatural: bool = False,
        font: str | Iterable[str] = (),
        musicFont: str | Iterable[str] = _defaultMusicFonts,
        fs: float = DEFAULT_FONT_SIZE,
        acciScale: float = 1,
        acciRaise: float = 0.25,
        acciBuff: float = 0.25,
        octaveScale: float = 0.75,
        octaveRaise: float = -0.25,
        octaveBuff: float = 0.25,
        middleCOctave: int = 4,
        color: JAnimColor | Callable[[fz.Pitch], JAnimColor] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        i_degName = self._i_degName = Text(
            fz.DEG_NAMES[pitch.opitch.deg],
            font=font,
            font_size=fs,
            format=Text.Format.RichText,
        )
        if callable(color):
            color = color(pitch)
        if color is not None:
            i_degName.fill.set(color=color)

        i_zero = Text("0", font=font, font_size=fs)
        cap = i_zero.points.box.height
        ch = i_zero.points.box.width

        if pitch.acci == 0 and not showNatural:
            i_acci = self._i_acci = createEmptyDot()
        else:
            i_acci = self._i_acci = MusicGlyph(
                _accidentalGlyphNames[pitch.acci],
                font=musicFont,
                fs=fs * acciScale,
            )

        (
            i_acci.points.next_to(
                i_degName,
                np.array((acciPosition, 0, 0)),
                buff=acciBuff * ch,
                coor_mask=RIGHT,
            ).shift(UP * cap * acciRaise)
        )
        if isinstance(pitch, fz.Pitch):
            i_octave = Text(
                str(pitch.octave + middleCOctave),
                font=font,
                font_size=fs * octaveScale,
            )
        else:
            i_octave = createEmptyDot()
        i_octave.points.next_to(
            i_degName, RIGHT, buff=octaveBuff * ch, coor_mask=RIGHT
        ).shift(UP * cap * octaveRaise)
        self.add(i_degName, i_acci, i_octave)

    @property
    def i_degName(self) -> Text:
        return self._i_degName

    @property
    def i_acci(self) -> MusicGlyph:
        return self._i_acci

    def getPosition(self) -> Vect:
        return self.i_degName[0].get_mark_orig()


class ScaleText(Group[NoteName], PositionedVItem):
    def __init__(
        self,
        scale: fz.Scale,
        colWidth=0.5,
        **kwargs,
    ):
        super().__init__()
        for i, pitch in enumerate(scale):
            x = i * colWidth
            i_noteName = NoteName(pitch, **kwargs).setPosition((x, 0, 0))
            self.add(i_noteName)

    def getPosition(self):
        return self[0].getPosition()
