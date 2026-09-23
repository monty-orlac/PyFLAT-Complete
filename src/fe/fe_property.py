import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

from .fe_base import *
from . import lib
import pyflat_numpy as np


class PropertyTR(PropertyBase, IPropertyTR):
    __slots__ = ["__area"]

    NPARAM = 1
    TYPE = "PTRUSS"

    MATERIAL_INTERFACES = [IMaterialTRUSS, ]

    @property
    def area(self) -> float:
        return self.__area

    def __init__(self, label: int, material: MaterialBase, area: float):
        super().__init__(label, material)
        self.__area: float = area
        return

    def get_dee_truss(self) -> float:
        assert isinstance(self._material, IMaterialTRUSS)
        return self._material.get_dee_truss()


class Property2D(PropertyBase, IPropertyPE, IPropertyPS):
    __slots__ = ["__thickness", "__dee_pe", "__dee_ps"]

    NPARAM = 1
    TYPE = "P2D"
    MATERIAL_INTERFACES = [IMaterial2D, ]

    @property
    def thickness(self) -> float:
        return self.__thickness

    def __init__(self, label: int, material: MaterialBase, thickness: float):
        super().__init__(label, material)
        self.__thickness: float = thickness

        assert isinstance(material, IMaterial2D)
        dee: np.MatrixType = material.get_dee_matrix2d()
        self.__dee_pe: np.MatrixType = lib.material_dee_convert_to_pe(dee)
        self.__dee_ps: np.MatrixType = lib.material_dee_convert_to_ps(dee)
        return

    def get_dee_pe(self) -> np.MatrixType:
        return self.__dee_pe

    def get_dee_ps(self) -> np.MatrixType:
        return self.__dee_ps


class PropertyAX(PropertyBase, IPropertyAX):
    __slots__ = ["__dee_ax"]

    NPARAM = 0
    TYPE = "PAX"
    MATERIAL_INTERFACES = [IMaterial2D, ]

    def __init__(self, label: int, material: MaterialBase):
        super().__init__(label, material)

        assert isinstance(material, IMaterial2D)
        dee: np.MatrixType = material.get_dee_matrix2d()
        self.__dee_ax: np.MatrixType = dee
        return

    def get_dee_ax(self) -> np.MatrixType:
        return self.__dee_ax
