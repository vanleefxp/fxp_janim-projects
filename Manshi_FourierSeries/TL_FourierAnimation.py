from janim.imports import *
from functools import cached_property

with reloads():
    from common import *

BILI_PINK = "#fb7299"


def get_fourier_coef_mask(n):
    n_half = n // 2 + 1
    mask = np.empty(n, dtype=int)
    mask[:n_half] = np.arange(n_half)
    mask[-n_half + 1 :] = np.arange(-n_half + 1, 0)
    return mask


class FourierCoefs:
    def __init__(self, coefs):
        self.coefs = coefs
        coefs.flags.writeable = False

    def components(self, t: float) -> np.ndarray[complex]:
        mask = get_fourier_coef_mask(len(self.coefs))
        if isinstance(t, np.ndarray):
            return np.exp(2j * np.pi * mask * t[:, None]) * self.coefs
        else:
            return np.exp(2j * np.pi * mask * t) * self.coefs

    def __call__(self, t: float, n: int | None = None) -> complex:
        components = self.components(t)
        if n is not None:
            components = components[: n * 2 - 1]
        if isinstance(t, np.ndarray):
            return np.sum(components, axis=1)
        else:
            return np.sum(components)

    @cached_property
    def max_n(self) -> int:
        return len(self.coefs) // 2 + 1

    def __len__(self) -> int:
        return len(self.coefs)

    def __getitem__(self, key):
        return self.coefs[key]

    def __iter__(self):
        return iter(self.coefs)


def complex2point(c: complex):
    return np.array((c.real, c.imag, 0))


def toPointsFn(fn: Callable[[float], complex]) -> Callable[[float], Vect]:
    def pointsFn(t):
        return complex2point(fn(t))

    return pointsFn


def alternatingSignedInts(stop_abs=None):
    yield 0
    for i in it.count(1) if stop_abs is None else range(1, stop_abs):
        yield i
        yield -i


class TL_FourierSeriesAnim(Timeline):
    CONFIG = config

    def construct(self):
        coefs = np.load(DIR / "assets/data/fourier-coefs/cxk.npy")
        max_n = 50
        scale = 2.5
        startPoint = np.array((0, 0.5, 0))

        coefs = np.concat((coefs[:max_n], coefs[-max_n + 1 :]))
        fn = FourierCoefs(coefs)
        coefs_alternating = np.empty_like(coefs)
        coefs_alternating[0] = coefs[0]
        coefs_alternating[1::2] = coefs[1 : fn.max_n]
        coefs_alternating[2::2] = coefs[-1 : -fn.max_n : -1]
        print(coefs_alternating)
        fn_points = toPointsFn(fn)
        i_graph = ParametricCurve(
            lambda t: fn_points(t) * scale + startPoint, (0, 1, 0.001), color=BILI_PINK
        )

        i_vecs = Group[Vector](
            *(
                Line(
                    ORIGIN,
                    ORIGIN,
                    buff=0,
                    stroke_radius=interpolate(0.01, 0.0025, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i in alternatingSignedInts(max_n)
            )
        )
        i_circs = Group[Circle](
            *(
                Circle(
                    np.abs(coef) * scale,
                    stroke_radius=interpolate(0.01, 0.005, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i, coef in zip(alternatingSignedInts(), coefs_alternating)
            )
        )
        i_dots = Group[Dot](
            *(
                Dot(radius=interpolate(0.04, 0.01, 1 - np.exp(-np.abs(i) / 5)))
                for i in alternatingSignedInts(max_n)
            )
        )
        i_circs[0].set(stroke_alpha=0)
        i_drawing = Group(i_vecs, i_circs, i_dots)
        i_startPoint = Dot(startPoint, radius=0.04)

        def updateVectors(t: float, i_drawing=i_drawing):
            i_vecs, i_circs, i_dots = i_drawing
            components = fn.components(t)
            components_cp = components.copy()
            components[0] = components_cp[0]
            components[1::2] = components_cp[1 : fn.max_n]
            components[2::2] = components_cp[-1 : -fn.max_n : -1]
            components_cumsum = np.cumsum(components)

            for (pc_prev, pc_next), i_vec, i_circ, i_dot in zip(
                it.pairwise(it.chain((0j,), components_cumsum)), i_vecs, i_circs, i_dots
            ):
                p_prev = complex2point(pc_prev * scale) + startPoint
                p_next = complex2point(pc_next * scale) + startPoint
                i_vec.points.set_start_and_end(p_prev, p_next)
                i_circ.points.move_to(p_prev)
                i_dot.points.move_to(p_next)

        updateVectors(0)
        self.play(FadeIn(i_startPoint), duration=0.5)
        self.play(Create(i_drawing, auto_close_path=False))
        self.forward(1)
        anim_duration = 15
        self.play(
            GroupUpdater(
                i_drawing,
                lambda item, params: updateVectors(params.alpha, item),
                duration=anim_duration,
                rate_func=linear,
            ),
            Create(
                i_graph, auto_close_path=False, duration=anim_duration, rate_func=linear
            ),
        )
        self.forward(1)
        self.play(Uncreate(i_drawing, auto_close_path=False), FadeOut(i_startPoint))
        # print(coefs)
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
