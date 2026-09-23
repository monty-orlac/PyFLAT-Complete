
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

from .fe_base import *
from . import lib
import pyflat_numpy as np

class MaterialTRUSS(MaterialBase, IMaterialTRUSS):
    __slots__ = ["__young"]

    NPARAM = 1
    TYPE = "MTRUSS"

    def __init__(self, label: int, young: float):
        super().__init__(label)
        self.__young: float = young
        return

    def get_dee_truss(self) -> float:
        return self.__young


class Material2DIso(MaterialBase, IMaterial2D, IMaterialTRUSS):
    __slots__ = ["__young", "__rpois", "__dee"]

    NPARAM = 2
    TYPE = "M2DISO"

    def __init__(self, label: int, young: float, rpois: float):
        super().__init__(label)
        self.__young: float = young
        self.__rpois: float = rpois
        self.__dee: np.MatrixType = lib.material_dee_get_iso(self.__young, self.__rpois)
        return

    def get_dee_truss(self) -> float:
        return self.__young

    def get_dee_matrix2d(self) -> np.MatrixType:
        return self.__dee

    @classmethod
    def get_instance(cls, label: int, params: cabc.Sequence[float]) -> typing.Self:
        if len(params) != cls.NPARAM:
            raise RuntimeError("Invalid number of initialization parameters.")
        return cls(label, *params)


class Material2DGeneral(MaterialBase, IMaterial2D):
    __slots__ = ["__dee"]

    def __init__(self, label: int, dee: np.MatrixType):
        super().__init__(label)
        self.__dee: np.MatrixType = dee
        return

    def get_dee_matrix2d(self) -> np.MatrixType:
        return self.__dee


class Material2DOrtho(Material2DGeneral):
    NPARAM = 8
    TYPE = "M2DORTHO"

    @classmethod
    def get_instance(cls, label: int, params: cabc.Sequence[float]) -> typing.Self:
        if len(params) != cls.NPARAM:
            raise RuntimeError("Invalid number of initialization parameters.")
        dee: np.MatrixType = lib.material_dee_get_ortho(*params[0:7])
        dee = lib.material_dee_rotate(dee, -params[7])
        return cls(label, dee)


class Material2DEngConstant(Material2DGeneral):
    NPARAM = 8
    TYPE = "M2DENG"

    @classmethod
    def get_instance(cls, label: int, params: cabc.Sequence[float]) -> typing.Self:
        if len(params) != cls.NPARAM:
            raise RuntimeError("Invalid number of initialization parameters.")
        yng1: float = params[0]
        yng2: float = params[1]
        yng3: float = params[2]
        nu12: float = params[3]
        nu21: float = nu12 * yng2 / yng1
        nu13: float = params[4]
        nu31: float = nu13 * yng3 / yng1
        nu23: float = params[5]
        nu32: float = nu23 * yng3 / yng2
        delta: float = 1.0 - nu12 * nu21 - nu23 * nu32 - nu13 * nu31 - 2.0 * nu21 * nu32 * nu13
        c11: float = (1.0 - nu23 * nu32) / delta * yng1
        c22: float = (1.0 - nu13 * nu31) / delta * yng2
        c33: float = (1.0 - nu12 * nu21) / delta * yng3
        c12: float = (nu12 + nu32 * nu13) / delta * yng2
        c13: float = (nu13 + nu12 * nu23) / delta * yng3
        c23: float = (nu23 + nu21 * nu13) / delta * yng3
        c44: float = params[6]
        dee: np.MatrixType = lib.material_dee_get_ortho(c11, c12, c22, c13, c23, c33, c44)
        dee = lib.material_dee_rotate(dee, -params[7])
        return cls(label, dee)
