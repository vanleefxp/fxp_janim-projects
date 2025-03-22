from collections.abc import Iterable

import numpy as np


def nextPowerOf2(n: int) -> int:
    if n <= 0:
        return 1
    result = 1
    power = 0
    while result < n:
        result <<= 1
        power += 1
    return result, power


def prevPowerOf2(n: int) -> int:
    result, power = nextPowerOf2(n)
    result >>= 1
    power -= 1
    return result, power


def gnext(n: float, k: float, rem: float = 0, strict: bool = True) -> float:
    """
    Returns the smallest integer multiple of `k` greater than `n`.
    """
    if n % k == rem:
        return n + k if strict else n
    return ((n - rem) // k + 1) * k + rem


def gprev(n: float, k: float, rem: float = 0, strict: bool = True) -> float:
    """
    Returns the largest integer multiple of `k` less than `n`.
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
