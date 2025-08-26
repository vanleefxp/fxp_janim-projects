import os
from pathlib import Path
from typing import SupportsIndex
from collections.abc import Callable, Sequence

import numpy as np
from numpy.lib.npyio import NpzFile
from scipy.interpolate import CubicSpline

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


class FourierFigure(Sequence[complex]):
    __slots__ = ("_coefs", "_coefFn", "_max_n")
    _coefs: np.ndarray[complex]
    _coefFn: CubicSpline

    def __new__(
        cls, file: str | os.PathLike, centralize: bool = True, normalize: bool = True
    ):
        result: NpzFile = np.load(file)

        interpolate_steps = int(result["interpolate_steps"])
        samples_positive = result["samples_positive"]
        samples_negative = result["samples_negative"]
        result.close()

        if centralize:
            samples_positive[0] = samples_negative[0] = 0

        max_n = len(samples_positive) // interpolate_steps

        samples = np.concat((samples_negative[:0:-1], samples_positive))
        n_values = (
            np.arange(-max_n * interpolate_steps + 1, max_n * interpolate_steps)
            / interpolate_steps
        )

        coefs = np.concat(
            (
                samples_positive[::interpolate_steps],
                samples_negative[-interpolate_steps:0:-interpolate_steps],
            )
        )

        if normalize:
            normalize_factor = np.linalg.norm(coefs)
            coefs /= normalize_factor
            samples /= normalize_factor

        coefFunc = CubicSpline(n_values, samples)
        # assert np.allclose(coefs, np.fft.ifftshift(coefFunc(np.arange(-max_n + 1, max_n))))

        self = cls._newHelper(coefs)
        self._coefFn = coefFunc
        self._max_n = max_n

        return self

    @classmethod
    def _newHelper(cls, coefs: np.ndarray[complex]):
        self = super().__new__(cls)
        self._coefs = coefs
        self._coefs.flags.writeable = False
        return self

    @property
    def coefs(self) -> np.ndarray[complex]:
        return self._coefs

    @property
    def coefFn(self) -> Callable[[float], complex]:
        return self._coefFn

    @property
    def max_n(self) -> int:
        return self._max_n

    def components(self, t: float) -> np.ndarray[complex]:
        n_values = np.fft.ifftshift(np.arange(-self.max_n + 1, self.max_n))
        if isinstance(t, np.ndarray):
            return np.exp(2j * np.pi * n_values * t[:, None]) * self._coefs
        else:
            return np.exp(2j * np.pi * n_values * t) * self._coefs

    def __call__(self, t: float) -> complex:
        components = self.components(t)
        if isinstance(t, np.ndarray):
            return np.sum(components, axis=1)
        else:
            return np.sum(components)

    def __len__(self) -> int:
        return len(self._coefs)

    def __getitem__(self, idx):
        if isinstance(idx, SupportsIndex):
            if idx >= self.max_n or idx <= -self.max_n:
                return 0
            return self._coefs[idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is not None or step is not None or stop is None or stop < 0:
                raise ValueError(f"Unsupported slice: {idx}")
            if stop == 0:
                result = self._newHelper(np.array((0,)))
                result._coefFn = self._coefFn
                result._max_n = 0
                return result
            elif stop == 1:
                result = self._newHelper(np.array((self._coefs[0],)))
                result._coefFn = self._coefFn
                result._max_n = 1
                return result
            elif stop >= self.max_n:
                return self
            else:
                result = self._newHelper(
                    np.concat((self._coefs[:stop], self._coefs[-stop + 1 :]))
                )
                result._coefFn = self._coefFn
                result._max_n = stop
                return result

    def __iter__(self):
        return iter(self._coefs)
