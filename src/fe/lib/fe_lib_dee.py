import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import math


def material_dee_get_iso(young: float, nu: float) -> np.MatrixType:
    coeff: float = young / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array([
        [1.0-nu, nu,     nu,     0.0],
        [nu,     1.0-nu, nu,     0.0],
        [nu,     nu,     1.0-nu, 0.0],
        [0.0,    0.0,    0.0,    0.5-nu],
    ]) * coeff


def material_dee_get_ortho(c11: float, c12: float, c22: float, c13: float, c23: float, c33: float, c44: float) -> np.MatrixType:
    return np.array([
        [c11, c12, c13, 0.0],
        [c12, c22, c23, 0.0],
        [c13, c23, c33, 0.0],
        [0.0, 0.0, 0.0, c44],
    ])


def material_dee_rotate(dee: np.MatrixType, angle_deg: float) -> np.MatrixType:
    c: float = math.cos(math.radians(angle_deg))
    s: float = math.sin(math.radians(angle_deg))
    rot_matrix_inv: np.MatrixType = np.array([
        [ c*c,  s*s,  0.0,  2.0*s*c],
        [ s*s,  c*c,  0.0, -2.0*s*c],
        [ 0.0,  0.0,  1.0,  0.0],
        [-s*c, +s*c,  0.0,  c*c-s*s],
    ])
    return rot_matrix_inv @ dee @ rot_matrix_inv.transpose()


def material_dee_convert_to_pe(dee: np.MatrixType) -> np.MatrixType:
    return dee.copy()


def material_dee_convert_to_ps(dee: np.MatrixType) -> np.MatrixType:
    dee_ps: typing.Callable[[int, int], float] = \
        lambda i, j: dee[(i, j)] - (dee[(i, 2)] * dee[(2, j)]) / dee[(2, 2)]
    return np.array([
        [dee_ps(0, 0), dee_ps(0, 1), 0.0, dee_ps(0, 3)],
        [dee_ps(1, 0), dee_ps(1, 1), 0.0, dee_ps(1, 3)],
        [0.0,          0.0,          0.0, 0.0],
        [dee_ps(3, 0), dee_ps(3, 1), 0.0, dee_ps(3, 3)], ])
