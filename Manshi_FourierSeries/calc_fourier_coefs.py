from pathlib import Path
from collections.abc import Mapping, Callable

import svgpathtools as svgp
import numpy as np
from scipy.integrate import quad
from frozendict import frozendict

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()

_default_intergal_args = frozendict(
    limit=200,
    epsabs=1e-10,
    epsrel=1e-10,
)


def fourier_integral(fn: Callable[[float], complex], n, **kwargs) -> complex:
    return quad(
        lambda t: np.exp(-2j * np.pi * n * t) * fn(t),
        0,
        1,
        complex_func=True,
        **kwargs,
    )[0]


def get_fourier_coefs(
    svg_file,
    max_n: int = 200,
    interpolate_steps: int = 10,
    integral_args: Mapping | None = None,
):
    integral_args = (
        _default_intergal_args
        if integral_args is None
        else _default_intergal_args | integral_args
    )

    svgpath = svgp.svg2paths(svg_file)[0][0]
    xmin, xmax, ymin, ymax = svgpath.bbox()
    x0 = (xmin + xmax) / 2
    y0 = (ymin + ymax) / 2
    rx = (xmax - xmin) / 2
    ry = (ymax - ymin) / 2
    scale_ratio = 1 / max(rx, ry)

    # 将 SVG 图形归一化使其位于画面中间
    svgpath: svgp.Path = svgpath.translated(complex(-x0, -y0)).scaled(
        scale_ratio, -scale_ratio
    )

    # 为计算连续的傅里叶变换，需要进行插值处理，在每两个整数之间多取几个点
    samples_positive = np.empty(max_n * interpolate_steps, dtype=complex)
    samples_negative = np.empty(max_n * interpolate_steps, dtype=complex)

    print("正在计算积分")

    # n = 0 的时候单独处理
    print("n = 0")
    dc_integral = quad(svgpath.point, 0, 1, complex_func=True, **integral_args)[0]
    samples_positive[0] = samples_negative[0] = dc_integral

    n_values = np.linspace(0, 1, interpolate_steps, False)[1:]
    integ_values_positive = np.array(
        [fourier_integral(svgpath.point, n, **integral_args) for n in n_values]
    )
    integ_values_negative = np.array(
        [fourier_integral(svgpath.point, -n, **integral_args) for n in n_values]
    )
    samples_positive[1:interpolate_steps] = integ_values_positive
    samples_negative[1:interpolate_steps] = integ_values_negative

    for i in range(1, max_n):
        n_values = np.linspace(i, i + 1, interpolate_steps, False)
        integ_values_positive = np.array(
            [fourier_integral(svgpath.point, n, **integral_args) for n in n_values]
        )
        integ_values_negative = np.array(
            [fourier_integral(svgpath.point, -n, **integral_args) for n in n_values]
        )
        print(f"n = {i}")
        samples_positive[i * interpolate_steps : (i + 1) * interpolate_steps] = (
            integ_values_positive
        )
        print(f"n = {-i}")
        samples_negative[i * interpolate_steps : (i + 1) * interpolate_steps] = (
            integ_values_negative
        )

    return (samples_positive, samples_negative, interpolate_steps)


if __name__ == "__main__":
    for file in (DIR / "assets/image/fourier-anim-shapes").glob("*.svg"):
        samples_positive, samples_negative, interpolate_steps = get_fourier_coefs(file)
        np.savez(
            DIR / f"assets/data/fourier-coefs/{file.stem}.npz",
            samples_positive=samples_positive,
            samples_negative=samples_negative,
            interpolate_steps=interpolate_steps,
            allow_pickle=False,
        )


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
