from pathlib import Path
import time
import subprocess
import shutil
import os

import toml
import music21 as m21
from scipy.io import wavfile
from janim.imports import *
import fantazia as fz

__all__ = ["midi2wav", "midi2Audio", "alda"]

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
_whole = m21.duration.Duration(4)
_soundfontConfig = toml.load(DIR / "assets/soundfont.toml")
_soundfontPath = Path(_soundfontConfig["soundfontPath"])


class alda(str): ...


def fromAldaFile(aldaPath: os.PathLike) -> Path:
    tempFolder = DIR / "_tmp"
    tempFolder.mkdir(exist_ok=True)
    midiPath = tempFolder / f"temp_{time.monotonic_ns()}.mid"
    aldaPath = Path(aldaPath)
    subprocess.run(
        (
            "alda",
            "export",
            "-f",
            str(aldaPath),
            "-o",
            str(midiPath),
        )
    )


def fromAlda(src: str) -> Path:
    tempFolder = DIR / "_tmp"
    tempFolder.mkdir(exist_ok=True)
    midiPath = tempFolder / f"temp_{time.monotonic_ns()}.mid"
    subprocess.run(
        (
            "alda",
            "export",
            "-c",
            src,
            "-o",
            str(midiPath),
        )
    )
    return midiPath


type PitchLike = fz.PitchBase | float | str


def toM21(pitch: PitchLike) -> m21.pitch.Pitch:
    if isinstance(pitch, (str, fz.OPitch)):
        pitch = fz.Pitch(pitch)
    if isinstance(pitch, fz.PitchBase):
        return fz.Pitch(pitch).m21()
    else:
        tone = pitch
        baseTone = round(tone)
        microtone = tone - baseTone
        return m21.pitch.Pitch(baseTone + 60, microtone=float(microtone * 100))


def pitch2Stream(
    p: PitchLike | Iterable[PitchLike], duration: float = 1
) -> m21.stream.Stream:
    stream = m21.stream.Stream()
    stream.append(m21.tempo.MetronomeMark(number=240 / duration))
    if isinstance(p, Iterable):
        chord = m21.chord.Chord(tuple(map(toM21, p)), duration=_whole)
        stream.append(chord)
    else:
        note = m21.note.Note(toM21(p), duration=_whole)
        stream.append(note)
    return stream


def fromM21(stream: m21.stream.Stream) -> Path:
    tempFolder = DIR / "_tmp"
    tempFolder.mkdir(exist_ok=True)
    midiPath = tempFolder / f"temp_{time.monotonic_ns()}.mid"
    stream.write("midi", midiPath)
    return midiPath


def midi2wav(
    midi: os.PathLike | str | m21.stream.Stream | PitchLike | Iterable[PitchLike],
    gain: float = 2,
):
    """
    Convert MIDI input to WAV using fluidsynth. The MIDI input can be a MIDI file path
    or an `alda` notation string.
    """
    # fmt: off
    tempFolder = DIR/"_tmp"
    # fmt: on
    tempFolder.mkdir(exist_ok=True)
    wavFilename = f"temp_{time.monotonic_ns()}"
    if isinstance(midi, os.PathLike):
        midiPath = Path(midi)
        if midiPath.suffix == ".alda":
            midiPath = fromAldaFile(midiPath)
    elif isinstance(midi, alda):
        midiPath = fromAlda(midi)
    elif isinstance(midi, m21.stream.Stream):
        midiPath = fromM21(midi)
    else:
        midiPath = fromM21(pitch2Stream(midi))

    wavPath = tempFolder / f"{wavFilename}.wav"

    # set audio format to 16-bit to be compatible with JAnim
    subprocess.run(
        (
            "fluidsynth",
            "-ni",
            "-g",
            str(gain),
            str(_soundfontPath),
            str(midiPath),
            "-r",
            str(Config.get.audio_framerate),
            "-F",
            str(wavPath),
        )
    )
    if not wavPath.exists():
        raise Exception(
            f"Failed to generate WAV from MIDI file: {midiPath}. "
            "Please check if `fluidsynth` is correctly installed and "
            "if the soundfont file is valid."
        )

    data = wavfile.read(wavPath)[1]
    nChannels = Config.get.audio_channels
    if nChannels > 1 and data.ndim == 1:
        data = data[:, np.newaxis] * np.ones(nChannels, dtype=np.int16)
    shutil.rmtree(tempFolder)
    return data


def midi2Audio(*args, **kwargs):
    data = midi2wav(*args, **kwargs)
    audio = Audio()
    audio.set_samples(data)
    return audio
