from __future__ import annotations
from typing import Self, Literal
from collections.abc import Mapping, Iterable, Sequence
from pathlib import Path
from enum import StrEnum
import itertools as it

from janim.imports import *
import pyrsistent as pyr
import numpy as np
import fantazia as fz

from .. import PositionedVItem
from ...utils.algorithm_utils import segStack
from .text import MusicGlyph, _defaultMusicFonts, _accidentalGlyphNames

__all__ = [
    "Staff",
    "StaffElement",
    "BarlineTypes",
    "PositionedVItem",
    "MusicGlyph",
    "Staff",
    "Clef",
    "Chord",
    "Note",
    "Barline",
    "KeySig",
    "Scale",
]

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


_staffDefaults = pyr.m(
    staffHeight=0.7,
    staffLength=24,
    staffLineThickness=0.075,
    changeClefScale=0.8,
    keySigAcciAdvance=1,
    acciColumnBuff=0.25,
    acciNoteheadBuff=0.5,
    ledgerLineThickness=0.1,
    # barlineThickness=(0.1, 0.3),
    # barlineDashLength=(0.4, 0.4),
    barlineBuff=0.3,
    doubleBarlineDistance=0.5,
)

_clefGlyphNames = pyr.pmap(
    {
        # G clef and variants
        "G": "gClef",
        "G_15": "gClef15mb",
        "G_8": "gClef8vb",
        "G^8": "gClef8va",
        "G^15": "gClef15ma",
        "GG": "gClef8vbOld",
        "G^C": "gClef8vbCClef",
        "G_(8)": "gClef8vbParens",
        # C clef and variants
        "C": "cClef",
        "C_8": "cClef8vb",
        # F clef and variants
        "F": "fClef",
        "F_15": "fClef15mb",
        "F_8": "fClef8vb",
        "F^8": "fClef8va",
        "F^15": "fClef15ma",
        # other clefs
        "||": "unpitchedPercussionClef1",
        "TAB": "6stringTabClef",
        "tab": "4stringTabClef",
    }
)


class BarlineTypes(StrEnum):
    THIN = "|"
    THICK = "."
    DASHED = "!"
    DOTTED = ";"
    COLON = ":"
    SHORT = ","
    TICK = "'"


_barlineGlyphNames = pyr.pmap(
    {
        "|": "barlineSingle",
        ".": "barlineHeavy",
        "!": "barlineDashed",
        ";": "barlineDotted",
        ":": "repeatDots",
        ",": "barlineShort",
        "'": "barlineTick",
    }
)


_clef2CVpos = pyr.m(G=1, F=-1, C=0)
_sharpPositions = np.array((3, 0, 4, 1, -2, 2, -1))
_flatPositions = np.array((-1, 2, -2, 1, -3, 0, -4))
_sharpPositions.flags.writeable = False
_flatPositions.flags.writeable = False


def _defaultClefVpos(sym: str):
    match sym[0]:
        case "G":
            return -2
        case "F":
            return 2
        case _:
            return 0


def _loadSmufl():
    import json

    global _smuflName2Codepoint, _loadSmufl
    jsonSrc = json.loads((DIR / "assets/glyphnames.json").read_text(encoding="UTF-8"))
    _smuflName2Codepoint = pyr.pmap(
        {k: chr(int(v["codepoint"][2:], base=16)) for k, v in jsonSrc.items()}
    )
    _loadSmufl = lambda: None


def _sepDict(d1: Mapping, d2: Mapping):
    d3 = dict(d1)
    d4 = {}
    for k, v in d2.items():
        if k in d1:
            d3[k] = v
        else:
            d4[k] = v
    return d3, d4


class StaffElement[T](Group[T]):
    def __init__(self, parent: Staff, add: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._parent = parent
        if add:
            self.parent.i_staffElements.add(self)

    @property
    def parent(self):
        return self._parent

    @property
    def sp(self):
        return self._parent.sp

    def getStartHpos(self) -> float:
        """Start position of the element on the staff measured in staff space (sp)."""
        x1 = self.points.box.get(LEFT)[0]
        x2 = self._parent.getPosition()
        return (x1 - x2) / self.sp

    def getEndHpos(self) -> float:
        """End position of the element on the staff measured in staff space (sp)."""
        x1 = self.points.box.get(RIGHT)[0]
        x2 = self._parent.getPosition()
        return (x1 - x2) / self.sp

    def before(self, other: "StaffElement | float", buff: float = 0) -> Self:
        """
        Place the current element after the other element or a horizontal position.
        """
        if not isinstance(other, Item):
            other = self.parent.getPosition() + other * self.sp * RIGHT
        self.points.next_to(other, LEFT, buff=buff * self.sp, coor_mask=RIGHT)
        return self

    def after(self, other: "StaffElement | float", buff: float = 0) -> Self:
        """
        Place the current element after the other element or a horizontal position.
        """
        if not isinstance(other, Item):
            other = self.parent.getPosition() + other * self.sp * RIGHT
        self.points.next_to(other, RIGHT, buff=buff * self.sp, coor_mask=RIGHT)
        return self

    def getHspan(self) -> float:
        return self.points.box.width / self.sp

    def shiftHpos(self, delta: float) -> Self:
        self.points.shift(delta * self.sp * RIGHT)
        return self


class PositionedStaffElement(StaffElement, PositionedVItem):
    def __init__(self, parent: Staff, **kwargs):
        super().__init__(parent, **kwargs)

    def getHpos(self):
        return (self.getPosition()[0] - self.parent.getPosition()[0]) / self.sp

    def setHpos(self, hpos: float | PositionedStaffElement) -> Self:
        if isinstance(hpos, PositionedStaffElement):
            hpos = hpos.getHpos()
        x = self.parent.getPosition()[0] + hpos * self.sp
        pos = self.getPosition()
        self.setPosition((x, *pos[1:]))
        return self


class Clef(PositionedStaffElement):
    def __init__(
        self,
        parent: Staff,
        clefType: str = "G",
        change: bool = False,
        vpos: int | None = None,
        **kwargs,
    ):
        sp = parent.sp
        if vpos is None:
            vpos = _defaultClefVpos(clefType)
        super().__init__(parent, **kwargs)

        i_glyph = self._i_glyph = MusicGlyph(
            char=_clefGlyphNames[clefType], font=parent.musicFont, sp=sp, **kwargs
        )
        self.add(self._i_glyph)

        if change:
            # TODO use specialized change clef glyph
            changeClefSize: float = self.parent.staffConfig["changeClefScale"]
            i_glyph.points.scale(changeClefSize, about_point=ORIGIN)
        i_glyph.setPosition(self.parent.getPosition())
        i_glyph.points.shift(vpos * sp / 2 * UP)

    def getPosition(self):
        return self._i_glyph.getPosition()


class KeySig(PositionedStaffElement):
    def __init__(
        self,
        parent,
        acciCount: int = 0,
        clefType: Literal["G", "F", "C"] = "G",
        cVpos: int | None = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        if clefType is not None:
            cVpos = _clef2CVpos[clefType]

        self._acciCount = acciCount
        self._cVpos = cVpos

        sp = self.sp
        acciAdvance = self.parent.staffConfig["keySigAcciAdvance"]
        musicFont = self.parent.musicFont

        if acciCount > 0:
            i_sharpTem = MusicGlyph("accidentalSharp", musicFont, sp)
            n = acciCount
            for i in range(n):
                x = i * acciAdvance * sp
                y = (_sharpPositions[i] + cVpos) * sp / 2
                i_sharp = i_sharpTem.copy().setPosition((x, y, 0))
                self.add(i_sharp)
        elif acciCount < 0:
            i_flatTem = MusicGlyph("accidentalFlat", musicFont, sp)
            n = -acciCount
            for i in range(n):
                x = i * acciAdvance * sp
                y = (_flatPositions[i] + cVpos) * sp / 2
                i_flat = i_flatTem.copy().setPosition((x, y, 0))
                self.add(i_flat)
        else:
            i_hiddenDot = Dot(radius=0.001)
            self.add(i_hiddenDot)

        self.points.shift(parent.getPosition())

    @property
    def cVpos(self) -> int:
        return self._cVpos

    @property
    def acciCount(self) -> int:
        return self._acciCount

    def getPosition(self):
        if self.acciCount == 0:
            return self[0].points.box.center
        else:
            return self[0].getPosition()


class Barline(PositionedStaffElement):
    def __init__(self, parent: Staff, lineStyle: str = "|", **kwargs):
        super().__init__(parent, **kwargs)
        sp = self.sp
        barlineBuff = self.parent.staffConfig["barlineBuff"]
        musicFont = self.parent.musicFont
        i_lastGlyph = MusicGlyph(_barlineGlyphNames[lineStyle[0]], musicFont, sp)
        i_lastGlyph.points.shift(2 * sp * DOWN)
        self.add(i_lastGlyph)
        for i, line in enumerate(lineStyle[1:]):
            i_glyph = MusicGlyph(_barlineGlyphNames[line], musicFont, sp)
            i_glyph.points.shift(2 * sp * DOWN)
            i_glyph.points.next_to(
                i_lastGlyph, RIGHT, buff=barlineBuff * sp, coor_mask=RIGHT
            )
            self.add(i_glyph)
            i_lastGlyph = i_glyph
        self.points.shift(parent.getPosition())

    def getPosition(self):
        return self[0].points.box.get(LEFT)


class Chord(PositionedStaffElement):
    def __init__(
        self,
        parent: Staff,
        vpos: Iterable[int],
        accis: None | Iterable[int | float | None] = None,
        noteheadType: str | Iterable[str] = "black",
        ledgerLineLength: float | tuple[float, float] | None = 1 / 3,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        sp = self.sp
        musicFont = self.parent.musicFont

        # input preprocess
        vpos = self._vpos = np.array(vpos)
        if isinstance(noteheadType, str):
            noteheadType = np.full(len(vpos), noteheadType)
        else:
            noteheadType = np.array(tuple(it.islice(noteheadType, len(vpos))))
        vposArgs = np.argsort(vpos)
        vpos = vpos[vposArgs]
        noteheadType = noteheadType[vposArgs]
        low, high = vpos[0], vpos[-1]
        if accis is None:
            accis = it.repeat(None)
        else:
            accis = np.array(tuple(accis), dtype=object)[vposArgs]

        # create groups
        i_noteheads = self._i_Noteheads = StaffElement(parent, add=False)
        i_ledgerLines = self._i_ledgerLines = StaffElement(parent, add=False)
        i_ledgerLines.add(Dot(radius=0.001))
        i_accis = self._i_accis = StaffElement(parent, add=False)
        # the single element is wrapped into a group due to a JAnim internal bug
        # hope this be fixed ASAP
        i_accis.add(Group(Dot(radius=0.001)))
        self.add(i_noteheads, i_ledgerLines, i_accis)

        @lru_cache
        def getNoteheadTem(noteheadType: str) -> MusicGlyph:
            i_noteheadTem = MusicGlyph(
                f"notehead{noteheadType[0].upper()}{noteheadType[1:]}", musicFont, sp
            )
            i_noteheadTem.points.next_to(ORIGIN, LEFT, buff=0)
            return i_noteheadTem

        # create ledger lines
        if ledgerLineLength is not None and (low < -5 or high > 5):
            headWidth = getNoteheadTem(
                "whole" if "whole" in noteheadType else "black"
            ).points.box.width
            th = self.parent.staffConfig["ledgerLineThickness"]
            if isinstance(ledgerLineLength, tuple):
                ledgerLeft, ledgerRight = ledgerLineLength
            else:
                ledgerLeft = ledgerRight = ledgerLineLength
            i_ledgerLineTem = Line(
                (-ledgerLeft * sp - headWidth, 0, 0),
                (ledgerRight * sp, 0, 0),
            )
            i_ledgerLineTem.radius.set(th * sp)

            # ledger lines below
            if low < -5:
                numLines = (-low - 4) // 2
                for i in range(numLines):
                    i_line = i_ledgerLineTem.copy()
                    y = (-i - 3) * sp
                    i_line.points.shift(UP * y)
                    i_ledgerLines.add(i_line)

            if high > 5:
                numLines = (high - 4) // 2
                for i in range(numLines):
                    i_line = i_ledgerLineTem.copy()
                    y = (i + 3) * sp
                    i_line.points.shift(UP * y)
                    i_ledgerLines.add(i_line)

        # create noteheads and accidentals
        lastFlipped = False
        lastVpos = None
        il_accidentals = []
        acciSegs = []
        self._accidentalMap = {}
        for i, (vp, acci) in enumerate(zip(vpos, accis)):
            y = vp * sp / 2
            i_noteheadTem = getNoteheadTem(noteheadType[i])
            i_notehead = i_noteheadTem.copy()
            i_notehead.points.shift(UP * y)
            if not lastFlipped and lastVpos is not None and vp - lastVpos == 1:
                i_notehead.points.shift(RIGHT * i_noteheadTem.points.box.width)
                lastFlipped = True
            else:
                lastFlipped = False
            i_noteheads.add(i_notehead)

            if acci is not None:
                acciGlyphName = _accidentalGlyphNames[acci]
                i_glyph = MusicGlyph(acciGlyphName, musicFont, sp)
                i_glyph.points.shift(UP * (sp * vp / 2))
                il_accidentals.append(i_glyph)
                self._accidentalMap[i] = i_glyph
                acciSegs.append(
                    np.array(
                        (
                            i_glyph.points.box.bottom[1],
                            i_glyph.points.box.top[1],
                        )
                    )
                )

            lastVpos = vp

        # determine accidental columns by greedy algorithm
        i_headsAndAccis = Group(i_noteheads)
        acciColumns = segStack(acciSegs)
        buff = self.parent.staffConfig["acciColumnBuff"]
        headBuff = self.parent.staffConfig["acciNoteheadBuff"]
        for column in acciColumns:
            i_column = StaffElement(parent, add=False)
            i_column.add(*(il_accidentals[i] for i in column))
            i_column.points.next_to(
                i_headsAndAccis, LEFT, buff=buff * sp, coor_mask=RIGHT
            )
            i_headsAndAccis.add(i_column)
            i_accis.add(i_column)

        i_accis.points.shift(LEFT * ((headBuff - buff) * sp))
        self.points.shift(parent.getPosition())

    @property
    def i_ledgerLines(self) -> StaffElement:
        return self._i_ledgerLines

    @property
    def i_noteheads(self) -> StaffElement:
        return self._i_Noteheads

    @property
    def i_accis(self) -> StaffElement:
        return self._i_accis

    def getAcci(self, i: int) -> MusicGlyph | None:
        return self._accidentalMap.get(i)

    def getPosition(self):
        return self.i_ledgerLines[0].points.box.center

    def copyWithoutAcci(self):
        i_cp = self.copy()
        i_dot = i_cp.i_accis[0]
        i_cp.i_accis.clear_children()
        i_cp.i_accis.add(i_dot)
        return i_cp


class Note(Chord):
    def __init__(
        self,
        parent: Staff,
        vpos: int,
        acci: int | float | None = None,
        noteheadType: str = "black",
        ledgerLineLength: float = 1 / 3,
        **kwargs,
    ):
        super().__init__(
            parent=parent,
            vpos=(vpos,),
            accis=(acci,),
            noteheadType=noteheadType,
            ledgerLineLength=ledgerLineLength,
            **kwargs,
        )

    @property
    def i_notehead(self) -> MusicGlyph:
        return self.i_noteheads[0]

    @property
    def i_acci(self) -> MusicGlyph | Dot:
        if len(self.i_accis) == 1:
            return self.i_accis[0][0]  # the invisible dot
        else:
            return self.i_accis[-1][0]  # the actual accidental


class Scale(StaffElement[Note]):
    @classmethod
    def fromNotation(
        cls,
        parent: Staff,
        scale: fz.Scale,
        cVpos: int = -6,
        color: JAnimColor | Callable[[fz.PitchBase], JAnimColor] | None = None,
        **kwargs,
    ):
        tonicDeg = scale.tonic.deg
        vpos = (tonicDeg + p.deg + cVpos for p in scale.mode)
        accis = ((None if p.acci == 0 else p.acci) for p in scale)
        i_scale = cls(parent, vpos, accis=accis, **kwargs)
        if color is not None:
            if callable(color):
                for i_note, p in zip(i_scale, scale):
                    c = color(p)
                    if c is not None:
                        i_note.i_notehead.color.set(c)
            else:
                for i_note in i_scale:
                    i_note.i_notehead.color.set(color)
        return i_scale

    def __init__(
        self,
        parent: Staff,
        vpos: Iterable[int],
        buff: float | Iterable[float] = 1.5,
        noteheadType: str | Iterable[str] = "black",
        accis: None | Iterable[int | None] = None,
        ledgerLineLength: float | tuple[float, float] | None = 1 / 3,
        acciSpaceRatio: float = 0.5,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        # input preprocess
        if isinstance(noteheadType, str):
            noteheadType = it.repeat(noteheadType)
        if not isinstance(buff, Iterable):
            buff = it.repeat(buff)
        if accis is None:
            accis = it.repeat(None)

        # create notes
        lastTarget = None
        for vp, nh, acci, b in zip(vpos, noteheadType, accis, it.chain((0,), buff)):
            i_note = Note(
                parent=parent,
                vpos=vp,
                noteheadType=nh,
                ledgerLineLength=ledgerLineLength,
                acci=acci,
                add=False,
            )
            if lastTarget is not None:
                i_note.setHpos(
                    lastTarget.getHpos()
                    + b
                    + i_note.i_noteheads.getHspan()
                    + acciSpaceRatio * i_note.i_accis.getHspan()
                )
            self.add(i_note)
            lastTarget = i_note

    def copyWithoutAcci(self) -> Self:
        i_cp = self.copy().clear_children()
        for i_note in self:
            i_cp.add(i_note.copyWithoutAcci())
        return i_cp


class Staff(Group, PositionedVItem):
    def __init__(self, musicFont=_defaultMusicFonts, **kwargs):
        self._staffConfig, kwargs = _sepDict(_staffDefaults, kwargs)
        super().__init__(**kwargs)

        self._musicFont = musicFont

        cfg = self._staffConfig = pyr.pmap(self._staffConfig)
        sp = self._sp = cfg["staffHeight"] / 4
        l = cfg["staffLength"]
        th = cfg["staffLineThickness"]

        self._i_staffLines = Group()
        self._i_staffElements = Group()
        for i in range(-2, 3):
            i_line = Line((0, sp * i, 0), (sp * l, sp * i, 0))
            i_line.radius.set(th * sp)
            self._i_staffLines.add(i_line)
        self.add(self._i_staffLines, self._i_staffElements)

    @property
    def sp(self) -> float:
        return self._sp

    @property
    def musicFont(self) -> str | Sequence[str]:
        return self._musicFont

    @property
    def i_staffLines(self) -> Group:
        return self._i_staffLines

    @property
    def i_staffElements(self) -> Group:
        return self._i_staffElements

    @property
    def staffConfig(self) -> Mapping[str, object]:
        return self._staffConfig

    def getPosition(self) -> Vect:
        return self.i_staffLines[2].points.box.get(LEFT)
