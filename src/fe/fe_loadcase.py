from __future__ import annotations
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import dataclasses

from .fe_base import *
# from . import lib
import pyflat_numpy as np

# **********************************************************************
# **********************************************************************
# **
# **    荷重境界条件データの実装
# **
# **********************************************************************
# **********************************************************************
def _get_node_and_dof(model: ModelData, ipar: cabc.Sequence[int]) -> tuple[Node, int]:
    """カードの (節点ラベル, 自由度番号(1始まり)) から節点と自由度インデックスを取得する。"""
    node: Node = model.nodes.get_by_label(ipar[0])
    if not 1 <= ipar[1] <= len(node.dof):
        raise RuntimeError("Invalid DOF number {:d} is specified (valid: 1-{:d}).".format(ipar[1], len(node.dof)))
    return node, ipar[1] - 1


class PointLoad(LoadBase):
    __slots__ = ["__node", "__dof", "__value"]

    TYPE = "PLOAD"

    @classmethod
    def get_card_initializer(cls) -> str:
        return "iif"

    def __init__(self, node: Node, dof: int, value: float):
        self.__node: Node = node
        self.__dof: int = dof
        self.__value: float = value
        return

    def apply(self, model: ModelData, rhs: np.VectorType) -> None:
        idof: int = self.__node.dof[self.__dof]
        if idof >= 0:
            rhs[idof] += self.__value
        return

    @classmethod
    def get_instance(cls, model: ModelData, ipar: cabc.Sequence[int], fpar: cabc.Sequence[float]) -> typing.Self:
        node, dof = _get_node_and_dof(model, ipar)
        return cls(node, dof, fpar[0])


class BoundarySPC(BaseCondition):
    __slots__ = ["__node", "__dof", "__value"]

    TYPE = "DISP"

    @classmethod
    def get_card_initializer(cls) -> str:
        return "iif"

    def __init__(self, node: Node, dof: int, value: float):
        self.__node: Node = node
        self.__dof: int = dof
        self.__value: float = value
        return

    def apply(self) -> list[tuple[int, float]]:
        idof: int = self.__node.dof[self.__dof]
        if idof < 0:
            raise RuntimeError("Failed to define displacement boundary: The DOF has already fixed.")
        return [(idof, self.__value), ]

    @classmethod
    def get_list_spc(cls, bounds: cabc.Iterable[BoundarySPC]) -> dict[int, float]:
        ret: dict[int, float] = {}
        for bound in bounds:
            ret.update({idof: value for (idof, value) in bound.apply()})
        return ret

    @classmethod
    def get_instance(cls, model: ModelData, ipar: cabc.Sequence[int], fpar: cabc.Sequence[float]) -> typing.Self:
        node, dof = _get_node_and_dof(model, ipar)
        if node.dof[dof] == Node.DOF_VALUE_FIXED:
            raise RuntimeError("Failed to define displacement boundary: "
                "DOF {:d} of node {:d} is already fixed in the NODE section.".format(ipar[1], node.label))
        return cls(node, dof, fpar[0])


@dataclasses.dataclass(frozen=True)
class EdgeLoadBase(LoadBase):
    element: ElementBase
    edge_id: int
    value: float
    direction: tuple[float, float] | None

    def __post_init__(self):
        if not isinstance(self.element, IElementEdge):
            raise RuntimeError("Specified element #{:d} doesn't have edges.".format(self.element.label))
        # 辺番号の範囲チェック (範囲外なら RuntimeError を送出)
        # edge_id は入力ファイル上の辺番号 (1始まり)
        if not (1 <= self.edge_id <= self.element.get_num_edges()):
            raise RuntimeError(f"Invalid edge number {self.edge_id} is specified for {self.element.TYPE} element.")
        return

    def apply(self, model: ModelData, rhs: np.VectorType) -> None:
        assert isinstance(self.element, IElementEdge)
        coords: np.MatrixType = self.element.get_coord_matrix()
        load: np.VectorType = self.element.calc_edge_load(coords, self.edge_id, self.value, self.direction)
        lm: cabc.Sequence[int] = self.element.get_lm()
        for il, ig in enumerate(lm):
            if ig < 0:
                # ig = 0 ならその自由度は拘束条件により消去済みなので処理しない
                continue
            rhs[ig] += load[il]
        return


@dataclasses.dataclass(frozen=True)
class EdgeLoad(EdgeLoadBase):
    TYPE = "ELOAD"

    @classmethod
    def get_card_initializer(cls) -> str:
        return "iifff"

    @classmethod
    def get_instance(cls, model: ModelData, ipar: cabc.Sequence[int], fpar: cabc.Sequence[float]) -> typing.Self:
        elem: ElementBase = model.elements.get_by_label(ipar[0])
        return cls(elem, ipar[1], fpar[0], (fpar[1], fpar[2]))


@dataclasses.dataclass(frozen=True)
class EdgePressure(EdgeLoadBase):
    TYPE: typing.ClassVar[str] = "EPRESS"

    @classmethod
    def get_card_initializer(cls) -> str:
        return "iif"

    @classmethod
    def get_instance(cls, model: ModelData, ipar: cabc.Sequence[int], fpar: cabc.Sequence[float]) -> typing.Self:
        elem: ElementBase = model.elements.get_by_label(ipar[0])
        return cls(elem, ipar[1], fpar[0], None)
