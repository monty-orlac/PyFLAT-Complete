import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

from .fe_base import *
from . import lib
import pyflat_numpy as np

# **********************************************************************
# **********************************************************************
# **
# **    要素データの実装
# **
# **********************************************************************
# **********************************************************************

# **********************************************************
#   TRUSS: トラス要素
# **********************************************************
class ElementTRUSS(ElementBase):
    __slots__ = []

    NNODE = 2
    TYPE = "TRUSS"
    PROPERTY_INTERFACES = [IPropertyTR, ]

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyTR)
        matrix_local = lib.calc_element_truss(
            coords,
            self._property.get_dee_truss(),
            self._property.area)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyTR)
        result = lib.calc_element_truss(
            coords,
            self._property.get_dee_truss(),
            self._property.area,
            ans_elem)
        assert isinstance(result, list)
        return result


# **********************************************************
#   PExxx: 平面ひずみ要素
# **********************************************************
class _ElementPE(ElementBase, IElementEdge):
    __slots__ = []

    PROPERTY_INTERFACES = [IPropertyPE, ]

    def _calc_edge_load(self, coords_edge: np.MatrixType, value: float, direction: tuple[float, float] | None = None) -> np.VectorType:
        assert isinstance(self._property, IPropertyPE)
        load_edge: np.VectorType = lib.calc_edge_2d(
            coords_edge,
            self._property.thickness,
            value, direction)
        return load_edge


class _ElementPEFullInt(_ElementPE):
    __slots__ = []

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyPE)
        dee: np.MatrixType = self._property.get_dee_pe()
        matrix_local = lib.calc_element_2d_full(
            coords, dee, self._property.thickness)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyPE)
        dee: np.MatrixType = self._property.get_dee_pe()
        result = lib.calc_element_2d_full(
            coords, dee, self._property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result


class ElementPETRIA3(_ElementPEFullInt):
    __slots__ = []

    NNODE = 3
    TYPE = "PETRIA3"


class ElementPEQUAD4(_ElementPEFullInt):
    __slots__ = []

    NNODE = 4
    TYPE = "PEQUAD4"


class ElementPETRIA6(_ElementPEFullInt):
    __slots__ = []

    NNODE = 6
    TYPE = "PETRIA6"


class ElementPEQUAD4R(_ElementPE):
    __slots__ = []

    NNODE = 4
    TYPE = "PEQUAD4R"

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyPE)
        matrix_local = lib.calc_element_plquad4r(
            coords,
            self._property.get_dee_pe(),
            self._property.thickness,
            CommonConfig.HOURGLASS_STIFFNESS)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyPE)
        result = lib.calc_element_plquad4r(
            coords,
            self._property.get_dee_pe(),
            self._property.thickness,
            CommonConfig.HOURGLASS_STIFFNESS, ans_elem)
        assert isinstance(result, list)
        return result


class ElementPEQUAD4I(_ElementPE):
    __slots__ = []

    NNODE = 4
    TYPE = "PEQUAD4I"

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyPE)
        matrix_local = lib.calc_element_plquad4i(
            coords,
            self._property.get_dee_pe(),
            self._property.thickness)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyPE)
        result = lib.calc_element_plquad4i(
            coords,
            self._property.get_dee_pe(),
            self._property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result


# **********************************************************
#   PSxxx: 平面応力要素
# **********************************************************
class _ElementPS(ElementBase, IElementEdge):
    __slots__ = []

    PROPERTY_INTERFACES = [IPropertyPS, ]

    def _calc_edge_load(self, coords_edge: np.MatrixType, value: float, direction: tuple[float, float] | None = None) -> np.VectorType:
        assert isinstance(self._property, IPropertyPS)
        load_edge: np.VectorType = lib.calc_edge_2d(
            coords_edge,
            self._property.thickness,
            value, direction)
        return load_edge


class _ElementPSFullInt(_ElementPS):
    __slots__ = []

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyPS)
        dee: np.MatrixType = self._property.get_dee_ps()
        matrix_local = lib.calc_element_2d_full(
            coords, dee, self._property.thickness)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyPS)
        dee: np.MatrixType = self._property.get_dee_ps()
        result = lib.calc_element_2d_full(
            coords, dee, self._property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result


class ElementPSTRIA3(_ElementPSFullInt):
    __slots__ = []

    NNODE = 3
    TYPE = "PSTRIA3"


class ElementPSQUAD4(_ElementPSFullInt):
    __slots__ = []

    NNODE = 4
    TYPE = "PSQUAD4"


class ElementPSTRIA6(_ElementPSFullInt):
    __slots__ = []

    NNODE = 6
    TYPE = "PSTRIA6"


class ElementPSQUAD4R(_ElementPS):
    __slots__ = []

    NNODE = 4
    TYPE = "PSQUAD4R"

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyPS)
        matrix_local = lib.calc_element_plquad4r(
            coords,
            self._property.get_dee_ps(),
            self._property.thickness,
            CommonConfig.HOURGLASS_STIFFNESS)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyPS)
        result = lib.calc_element_plquad4r(
            coords,
            self._property.get_dee_ps(),
            self._property.thickness,
            CommonConfig.HOURGLASS_STIFFNESS, ans_elem)
        assert isinstance(result, list)
        return result


class ElementPSQUAD4I(_ElementPS):
    __slots__ = []

    NNODE = 4
    TYPE = "PSQUAD4I"

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyPS)
        matrix_local = lib.calc_element_plquad4i(
            coords,
            self._property.get_dee_ps(),
            self._property.thickness)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyPS)
        result = lib.calc_element_plquad4i(
            coords,
            self._property.get_dee_ps(),
            self._property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result

# **********************************************************
#   AXxxx: 軸対称要素
# **********************************************************
class _ElementAX(ElementBase, IElementEdge):
    __slots__ = []

    PROPERTY_INTERFACES = [IPropertyAX, ]

    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        assert isinstance(self._property, IPropertyAX)
        dee: np.MatrixType = self._property.get_dee_ax()
        matrix_local = lib.calc_element_2d_full(
            coords, dee, -1.0)
        assert not isinstance(matrix_local, list)
        return matrix_local

    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        assert isinstance(self._property, IPropertyAX)
        dee: np.MatrixType = self._property.get_dee_ax()
        result = lib.calc_element_2d_full(
            coords, dee, -1.0,
            ans_elem)
        assert isinstance(result, list)
        return result

    def _calc_edge_load(self, coords_edge: np.MatrixType, value: float, direction: tuple[float, float] | None = None) -> np.VectorType:
        assert isinstance(self._property, IPropertyAX)
        load_edge: np.VectorType = lib.calc_edge_2d(
            coords_edge,
            -1.0,
            value, direction)
        return load_edge


class ElementAXTRIA3(_ElementAX):
    __slots__ = []

    NNODE = 3
    TYPE = "AXTRIA3"


class ElementAXQUAD4(_ElementAX):
    __slots__ = []

    NNODE = 4
    TYPE = "AXQUAD4"


class ElementAXTRIA6(_ElementAX):
    __slots__ = []

    NNODE = 6
    TYPE = "AXTRIA6"
