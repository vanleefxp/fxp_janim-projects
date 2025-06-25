from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    @overload
    def nextPow2(n: int, strict: bool, withPow: Literal[True]) -> tuple[int, int]: ...

    @overload
    def nextPow2(n: int, strict: bool, withPow: Literal[False]) -> int: ...

    @overload
    def prevPow2(n: int, strict: bool, withPow: Literal[True]) -> tuple[int, int]: ...

    @overload
    def prevPow2(n: int, strict: bool, withPow: Literal[False]) -> int: ...


def nextPow2(
    n: int, strict: bool = True, withPow: bool = False
) -> tuple[int, int] | int:
    if n <= 0:
        return 1
    result = 1
    power = 0
    if strict:
        while result <= n:
            result <<= 1
            power += 1
    else:
        while result < n:
            result <<= 1
            power += 1
    if withPow:
        return result, power
    else:
        return result


def prevPow2(
    n: int, strict: bool = True, withPow: bool = False
) -> int | tuple[int, int]:
    result, power = nextPow2(n, False)
    if strict:
        while result >= n:
            result >>= 1
            power -= 1
    else:
        while result > n:
            result >>= 1
            power -= 1
    if withPow:
        return result, power
    else:
        return result


def gnext(n: float, k: float, rem: float = 0, strict: bool = True) -> float:
    """
    Returns the smallest number in the form of `k * t + r` greater than `n` where `t` is an
    integer.
    """
    if n % k == rem:
        return n + k if strict else n
    return ((n - rem) // k + 1) * k + rem


def gprev(n: float, k: float, rem: float = 0, strict: bool = True) -> float:
    """
    Returns the largest number in the form of `k * t + r` less than `n` where `t` is an integer.
    """
    if n % k == rem:
        return n - k if strict else n
    return (n - rem) // k * k + rem


def _clusterCenters(data: np.ndarray, labels):
    nClusters = np.max(labels) + 1
    centers = np.empty(nClusters, dtype=float)
    for i in range(nClusters):
        cluster = data[labels == i]
        centers[i] = np.mean(cluster)
    return centers


def fdiv(n: float, k: float, strict: bool = False) -> int:
    if strict and n % k == 0:
        return int(n // k) - 1
    return int(n // k)


def cdiv(n: float, k: float, strict: bool = False) -> int:
    if strict and n % k == 0:
        return int(n // k) + 1
    return int(-(-n // k))


def grange(
    start: float,
    stop: float,
    k: float,
    rem: float = 0,
    includeStart: bool = True,
    includeEnd: bool = True,
) -> Iterable[float]:
    """
    Yields all integer multiples of `k` with remainder `rem` in the range `[start, stop]`.
    """
    return map(
        lambda t: t * k + rem,
        range(
            cdiv(start - rem, k, not includeStart),
            fdiv(stop - rem, k, not includeEnd) + 1,
        ),
    )


def modShift(n: float, k: float, disp: float, includeRight: bool = True) -> float:
    result = n % k
    if result > disp or (not includeRight and result == disp):
        result -= k
    return result


def approxGCD(data: Iterable[float], tolerance: float) -> float:
    """
    find a value `k` such that all data elements are approximately multiples of `k`
    """
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=tolerance, min_samples=1)
    data = np.array(data, dtype=float)
    while len(data) > 1:
        # remove elements close to zero
        # otherwise the algorithm might never terminate
        data = data[abs(data) > tolerance]
        data.sort()
        data = data - np.insert(data, 0, 0)[:-1]
        clusterResult = dbscan.fit(data.reshape(-1, 1))
        labels = clusterResult.labels_
        data = _clusterCenters(data, labels)
    return data[0]
