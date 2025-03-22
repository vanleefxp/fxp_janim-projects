import numpy as np


def udvec(angle: float) -> np.ndarray:
    return np.array(
        (
            np.cos(angle),
            np.sin(angle),
            0,
        )
    )
