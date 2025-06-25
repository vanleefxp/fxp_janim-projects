from janim.imports import *
import numpy as np


class TL_ConformalMappings(Timeline):
    def construct(self):
        targetMatrix = np.array(
            (
                (-1, 1j),
                (1, 1j),
            )
        )
        targetMatrixNormalized = targetMatrix / (np.linalg.det(targetMatrix) ** (1 / 2))

        def fn(z):
            try:
                return (1j - z) / (1j + z + 1e-5)
            except ZeroDivisionError:
                return 0j

        i_coord = NumberPlane()
        i_coord_orig = i_coord.store()

        def updateCoord(item: NumberPlane, p: UpdaterParams):
            item.restore(i_coord_orig)
            item.points.prepare_for_nonlinear_transform()

            t = p.alpha
            a0, b0, c0, d0 = 1, 0, 0, 1
            a1, b1, c1, d1 = targetMatrixNormalized.flatten()
            a, b, c, d = (
                a0 + (a1 - a0) * t,
                b0 + (b1 - b0) * t,
                c0 + (c1 - c0) * t,
                d0 + (d1 - d0) * t,
            )
            item.points.apply_complex_fn(lambda z: (a * z + b) / (c * z + d))

        self.show(i_coord)
        self.play(GroupUpdater(i_coord, updateCoord), duration=5)
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
