from __future__ import annotations
from collections.abc import Mapping, Iterator, Iterable
from typing import Any, overload
import itertools as it

from janim.imports import *
import pyrsistent as pyr
import numpy as np
import fantazia as fz

from .. import PositionedVItem

__all__ = [
    "PianoKey",
    "KeyGroup",
    "PianoKeyboardOctave",
    "PianoKeyboard",
    "MARK_RED",
    "MARK_GREEN",
    "MARK_BLUE",
    "MARK_GRAY",
]

_whiteKeyIdx = np.arange(-1, 6) * 7 % 12
_blackKeyIdx = np.arange(6, 11) * 7 % 12
_whiteKeyIdx.sort()
_blackKeyIdx.sort()
_blackKeyPos = np.array((0, 1, 3, 4, 5))
_whiteKeyIdx.flags.writeable = False
_blackKeyIdx.flags.writeable = False
_blackKeyPos.flags.writeable = False

MARK_RED = (RED_B, RED_D)
MARK_GREEN = (GREEN_B, GREEN_D)
MARK_BLUE = (BLUE_B, BLUE_D)
MARK_GRAY = (GREY_B, GREY_D)

_keyboardConfig = pyr.m(
    whiteWidth=0.35,
    whiteHeight=2,
    blackWidth=0.175,
    blackHeight=1.1,
    cornerWidth=0.08,
    blackDisplace=(-0.1, 0.1, -0.1, 0, 0.1),
)

type ColorPair = tuple[JAnimColor, JAnimColor]
type KeyMarkColor = ColorPair | JAnimColor
type SetMarksColorType = Iterable[KeyMarkColor] | Callable[[PianoKey], KeyMarkColor]


def _sepDict(d1: Mapping, d2: Mapping):
    d3 = dict(d1)
    d4 = {}
    for k, v in d2.items():
        if k in d1:
            d3[k] = v
        else:
            d4[k] = v
    return d3, d4


class PianoKey(PositionedVItem):
    def __init__(self, idx: int = 0, **kwargs):
        self._keyboardConfig, kwargs = _sepDict(_keyboardConfig, kwargs)
        cfg = self._keyboardConfig = pyr.pmap(self._keyboardConfig)

        self._idx = idx
        self._marked = False
        isBlack = self._isBlack = (idx % 12) in _blackKeyIdx

        cw = cfg["cornerWidth"]
        if isBlack:
            w, h = cfg["blackWidth"], cfg["blackHeight"]
        else:
            w, h = cfg["whiteWidth"], cfg["whiteHeight"]

        super().__init__(
            # fmt: off
            *PathBuilder()
                .move_to((0, 0, 0))
                .line_to((w, 0, 0))
                .line_to((w, -h + cw, 0))
                .conic_to((w, -h, 0), (w - cw, -h, 0))
                .line_to((cw, -h, 0))
                .conic_to((0, -h, 0), (0, -h + cw, 0))
                .close_path()
                .get()
            # fmt: on
        )
        (
            self._resetColor()
            .fill.set(alpha=1)
            .r.stroke.set(color=GREY)
            .r.radius.set(0.01)
            .r.depth.set(0 if isBlack else 1)
        )

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def isBlack(self) -> bool:
        return self._isBlack

    @property
    def marked(self) -> bool:
        return self._marked

    @property
    def keyboardConfig(self) -> Mapping[str, Any]:
        return self._keyboardConfig

    def getPosition(self):
        return self.points.box.get(UL)

    def mark(self, color: ColorPair | JAnimColor | None = MARK_BLUE):
        if color is None:
            self.unmark()
            return self
        self._marked = True
        if isinstance(color, tuple) and len(color) == 2:
            self.fill.set(color=color[1 if self.isBlack else 0])
        else:
            self.fill.set(color=color)
        return self

    def _resetColor(self):
        self.fill.set(color=BLACK if self.isBlack else WHITE, alpha=1)
        return self

    def unmark(self):
        if self._marked:
            self._resetColor()
            self._marked = False
        return self


class KeyGroup(Group[PianoKey]):
    def mark(self, color: tuple[JAnimColor, JAnimColor] | JAnimColor = MARK_BLUE):
        for i_key in self:
            i_key.mark(color)
        return self

    def unmark(self):
        for i_key in self:
            i_key.unmark()
        return self

    def setMarks(
        self,
        idx: int | Iterable[int] = (),
        color: SetMarksColorType = (MARK_BLUE,),
    ):
        self.unmark()
        if not isinstance(idx, Iterable):
            self[idx].mark(color)
            return self
        if callable(color):
            color = map(color, self)
        else:
            color = it.cycle(color)
        for i, c in zip(idx, color):
            self[i].mark(c)
        return self


class PianoKeyboardOctave(KeyGroup, PositionedVItem):
    def __init__(self, keyRange: tuple[int, int] = (0, 12), octave: int = 0, **kwargs):
        self._keyboardConfig, kwargs = _sepDict(_keyboardConfig, kwargs)
        super().__init__(**kwargs)

        cfg = self._keyboardConfig = pyr.pmap(self._keyboardConfig)
        self._keyRange = keyRange
        self._octave = octave

        w1, w2 = cfg["whiteWidth"], cfg["blackWidth"]
        blackDisplace = cfg["blackDisplace"]

        il_keys = np.empty(12, dtype=object)
        il_keys[:] = [PianoKey(idx=i + 12 * octave) for i in range(12)]
        il_whiteKeys = il_keys[_whiteKeyIdx]
        il_blackKeys = il_keys[_blackKeyIdx]

        i_whiteKey: PianoKey
        for i, i_whiteKey in enumerate(il_whiteKeys):
            dx = i * w1
            i_whiteKey.points.shift(dx * RIGHT)

        i_blackKey: PianoKey
        for i, i_blackKey, disp in zip(_blackKeyPos, il_blackKeys, blackDisplace):
            dx = (i + 1) * w1 + w2 * (disp - 0.5)
            i_blackKey.points.shift(dx * RIGHT)

        self._i_whiteKeys = Group(*il_whiteKeys)
        self._i_blackKeys = Group(*il_blackKeys)
        self._i_keys = Group(*il_keys)
        self.add(*il_keys[slice(*keyRange)])

    @property
    def i_keys(self):
        return self._i_keys

    @property
    def i_whiteKeys(self):
        return self._i_whiteKeys

    @property
    def i_blackKeys(self):
        return self._i_blackKeys

    @property
    def keyboardConfig(self) -> Mapping[str, Any]:
        return self._keyboardConfig

    @property
    def keyRange(self) -> tuple[int, int]:
        return self._keyRange

    @property
    def octave(self) -> int:
        return self._octave

    def getPosition(self):
        return self.points.box.get(UL)


class PianoKeyboard(Group[PianoKeyboardOctave], PositionedVItem):
    def __init__(
        self, keyRange: tuple[int, int] = (-12, 24), originOctave: int = 0, **kwargs
    ):
        self._keyboardConfig, kwargs = _sepDict(_keyboardConfig, kwargs)
        super().__init__(**kwargs)
        cfg = self._keyboardConfig = pyr.pmap(self._keyboardConfig)
        startKey, endKey = keyRange
        startOctave = keyRange[0] // 12
        endOctave = keyRange[1] // 12
        if endOctave % 12 != 0:
            endOctave += 1
        octaveWidth = cfg["whiteWidth"] * 7

        self._keyRange = keyRange
        self._octaveRange = (startOctave, endOctave)
        self._octaves = _PianoKeyboardOctaveAccessor(self)
        self._keys = _PianoKeyboardKeyAccessor(self)

        self._i_dot = Dot(radius=0)
        self.add(self._i_dot)
        self._i_octaves = Group()
        self._i_keys = KeyGroup()

        if startOctave == endOctave:
            # one octave only
            i_octave = PianoKeyboardOctave(
                keyRange=(startKey % 12, endKey % 12), octave=startOctave, **cfg
            )
            i_octave.points.shift((startOctave - originOctave) * octaveWidth * RIGHT)
            self.add(i_octave)
            self._i_keys.add(*i_octave)
            self._i_octaves.add(i_octave)
        else:
            # multiple octaves

            # first octave on the left
            i_octave = PianoKeyboardOctave(
                keyRange=(startKey % 12, 12), octave=startOctave, **cfg
            )
            i_octave.points.shift((startOctave - originOctave) * octaveWidth * RIGHT)
            self.add(i_octave)
            self._i_keys.add(*i_octave)
            self._i_octaves.add(i_octave)

            # middle octaves
            for octave in range(startOctave + 1, endOctave - 1):
                i_octave = PianoKeyboardOctave(keyRange=(0, 12), octave=octave, **cfg)
                i_octave.points.shift((octave - originOctave) * octaveWidth * RIGHT)
                self.add(i_octave)
                self._i_keys.add(*i_octave)
                self._i_octaves.add(i_octave)

            # last octave on the right
            i_octave = PianoKeyboardOctave(
                keyRange=(0, endKey % 12), octave=endOctave, **cfg
            )
            i_octave.points.shift((endOctave - originOctave - 1) * octaveWidth * RIGHT)
            self.add(i_octave)
            self._i_keys.add(*i_octave)
            self._i_octaves.add(i_octave)

    @property
    def keyRange(self) -> tuple[int, int]:
        return self._keyRange

    @property
    def octaveRange(self) -> tuple[int, int]:
        return self._octaveRange

    @property
    def keyboardConfig(self) -> Mapping[str, Any]:
        return self._keyboardConfig

    @property
    def octaves(self) -> _PianoKeyboardOctaveAccessor:
        return self._octaves

    @property
    def keys(self) -> _PianoKeyboardKeyAccessor:
        return self._keys

    @property
    def i_keys(self) -> KeyGroup:
        return self._i_keys

    @property
    def i_octaves(self) -> Group:
        return self._i_octaves

    def getPosition(self):
        return self._i_dot.points.box.center

    def mark(self, color: tuple[Color, Color] | Color = MARK_BLUE):
        self._i_keys.mark(color)

    def unmark(self):
        self._i_keys.unmark()

    @overload
    def setMarks(self, idx: int, color: KeyMarkColor): ...

    @overload
    def setMarks(
        self,
        idx: Iterable[int],
        color: Iterable[KeyMarkColor] | Callable[[PianoKey], KeyMarkColor],
    ): ...

    def setMarks(
        self, idx: Iterable[int] = (), color: SetMarksColorType = (MARK_BLUE,)
    ):
        for i_key in self._i_keys:
            i_key.unmark()
        if not isinstance(idx, Iterable):
            self.keys[i].mark(color)
            return self
        if callable(color):
            color = map(color, self._i_keys)
        else:
            color = it.cycle(color)
        for i, c in zip(idx, color):
            self.keys[i].mark(c)
        return self


class _PianoKeyboardOctaveAccessor:
    def __init__(self, parent: PianoKeyboard):
        self._parent = parent

    def __len__(self) -> int:
        return len(self._parent)

    @overload
    def __getitem__(self, idx: int) -> PianoKeyboardOctave: ...

    @overload
    def __getitem__(self, idx: slice | Iterable[int]) -> Group[PianoKeyboardOctave]: ...

    def __getitem__(
        self, idx: int | slice | Iterable[int]
    ) -> PianoKeyboardOctave | Group[PianoKeyboardOctave]:
        startOctave = self._parent.octaveRange[0]
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start - startOctave
            stop = -1 if idx.stop is None else idx.stop - startOctave
            idx = slice(start, stop, idx.step)
            return self._parent[idx]
        elif isinstance(idx, Iterable):
            return Group(*(self._parent[i - startOctave] for i in idx))
        else:
            return self._parent[idx - self._parent.octaveRange[0]]

    def __iter__(self) -> Iterator[PianoKeyboardOctave]:
        return iter(self._parent)


class _PianoKeyboardKeyAccessor:
    def __init__(self, parent: PianoKeyboard):
        self._parent = parent

    @overload
    def __getitem__(self, idx: int) -> PianoKey: ...

    @overload
    def __getitem__(self, idx: slice | Iterable[int | fz.PitchBase]) -> KeyGroup: ...

    def __getitem__(
        self, idx: int | fz.PitchBase | slice | Iterable[int | fz.PitchBase]
    ) -> PianoKey | KeyGroup:
        startKey = self._parent.keyRange[0]
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start - startKey
            stop = -1 if idx.stop is None else idx.stop - startKey
            idx = slice(start, stop, idx.step)
            return self._parent._i_keys[idx]
        elif isinstance(idx, Iterable):
            # currently numpy integers are not regarded as a valid index
            # this is probably a bug of JAnim
            return KeyGroup(*(self._parent._i_keys[int(i - startKey)] for i in idx))
        else:
            if isinstance(idx, fz.PitchBase):
                idx = round(idx.tone)
            return self._parent._i_keys[int(idx - startKey)]

    def __len__(self) -> int:
        return len(self._parent._i_keys)

    def __iter__(self) -> Iterator[PianoKey]:
        return iter(self._parent._i_keys)
