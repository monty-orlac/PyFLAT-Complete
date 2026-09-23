import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import abc
import dataclasses
import enum

from . import lib
import pyflat_numpy as np

class AlgorithmReorder(enum.Enum):
    SIMPLE = "Simple sequential indexing."
    RCM = "Reverse Cuthill-McKee (RCM) method to optimize the sparsity pattern."


class CommonConfig:
    HOURGLASS_STIFFNESS: typing.ClassVar[float] = 0.01
    SMALL_STRESS_CUTOFF: typing.ClassVar[float] = 0.0
    SMALL_DISPLACEMENT_CUTOFF: typing.ClassVar[float] = 0.0
    REORDER_ALGORITHM: AlgorithmReorder = AlgorithmReorder.SIMPLE


class CardEntity(metaclass=abc.ABCMeta):
    __slots__ = []

    TYPE: typing.ClassVar[str]

    @classmethod
    @abc.abstractmethod
    def get_card_initializer(cls) -> str:
        ...


class LabeledEntity(CardEntity):
    __slots__ = ["__label"]

    @property
    def label(self) -> int:
        return self.__label
        
    def __init__(self, label: int):
        self.__label: int = label
        return


class LabeledEntityArray[T_ENTITY: LabeledEntity](cabc.Sequence[T_ENTITY]):
    __slots__ = ["__label_map", "__array", "__stype"]

    TYPE: typing.ClassVar[str]

    def __init__(self, array: list[T_ENTITY]):
        self.__array: list[T_ENTITY] = array
        self.__label_map: dict[int, int] = {it.label: index for index, it in enumerate(array)}
        return

    @typing.overload
    def __getitem__(self, index: int) -> T_ENTITY:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> cabc.Sequence[T_ENTITY]:
        ...
    
    def __getitem__(self, index: int | slice) -> T_ENTITY | cabc.Sequence[T_ENTITY]:
        if isinstance(index, slice):
            # スライスが渡された場合は、新しいMySequenceインスタンスを返すなどの処理
            return list(self.__array[index])
        return self.__array[index]

    def __len__(self) -> int:
        return len(self.__array)

    def get_index_by_label(self, label: int) -> int:
        index: int = self.__label_map.get(label, -1)
        if index < 0:
            raise RuntimeError("Specified {:s} label {:d} not found.".format(self.TYPE.lower(), label))
        return index

    def get_by_label(self, label: int) -> T_ENTITY:
        index: int = self.get_index_by_label(label)
        return self.__array[index]


# **********************************************************************
# **********************************************************************
# **
# **    モデルデータ
# **
# **********************************************************************
# **********************************************************************

# **********************************************************
#   Node: 節点データ
# **********************************************************
class Node(LabeledEntity):
    __slots__ = ["__index", "__coord", "__dof"]

    TYPE = "NODE"
    DOF_VALUE_FIXED: typing.ClassVar[int] = -1
    _DOF_VALUE_FREE: typing.ClassVar[int] = -2

    @classmethod
    def get_card_initializer(cls) -> str:
        return "uiiff"

    @property
    def coord(self) -> np.VectorType:
        return self.__coord

    @property
    def dof(self) -> list[int]:
        return self.__dof

    def __init__(self, label: int, index: int, coord: tuple[float, float], fixed: tuple[bool, bool]):
        super().__init__(label)
        self.__index: int = index
        self.__coord: np.VectorType = np.array(coord)
        self.__coord.flags.writeable = False   # self.coordを書き込み禁止にする
        self.__dof: list[int] = [self.DOF_VALUE_FIXED if fx else self._DOF_VALUE_FREE for fx in fixed]
        return


class NodeArray(LabeledEntityArray[Node]):
    __slots__ = []

    TYPE = "node"


# **********************************************************
#   MaterialBase: 材料特性データ
# **********************************************************
class MaterialBase(LabeledEntity, metaclass=abc.ABCMeta):
    __slots__ = []

    NPARAM: typing.ClassVar[int]

    @classmethod
    def get_card_initializer(cls) -> str:
        return "u" + "f" * cls.NPARAM

    @classmethod
    def get_instance(cls, label: int, params: cabc.Sequence[float]) -> typing.Self:
        if len(params) != cls.NPARAM:
            raise RuntimeError("Invalid number of parameters are specified: {:d} expected, but {:d} found.".format(cls.NPARAM, len(params)))
        return cls(label, *params)


class IMaterialTRUSS(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_dee_truss(self) -> float:
        ...


class IMaterial2D(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_dee_matrix2d(self) -> np.MatrixType:
        ...


# **********************************************************
#   PropertyBase: 要素特性データ
# **********************************************************
class PropertyBase(LabeledEntity, metaclass=abc.ABCMeta):
    __slots__ = ["_material"]

    NPARAM: typing.ClassVar[int]
    MATERIAL_INTERFACES: list[type]

    @classmethod
    def get_card_initializer(cls) -> str:
        return "uu" + "f" * cls.NPARAM

    def __init__(self, label: int, material: MaterialBase):
        super().__init__(label)
        self.__label: int = label
        self._material: MaterialBase = material
        return

    @classmethod
    def get_instance(cls, label: int, material: MaterialBase, params: cabc.Sequence[float]) -> typing.Self:
        if not isinstance(material, tuple(cls.MATERIAL_INTERFACES)):
            raise RuntimeError("The specified material label.{:d} (type:{:s}) cannot be used in property {:s}".format(
                material.label, material.TYPE, cls.TYPE))
        if len(params) != cls.NPARAM:
            raise RuntimeError("Invalid number of parameters are specified: {:d} expected, but {:d} found.".format(len(params), cls.NPARAM))
        return cls(label, material, *params)


class IPropertyTR(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def area(self) -> float:
        ...

    @abc.abstractmethod
    def get_dee_truss(self) -> float:
        ...


class IPropertyPE(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def thickness(self) -> float:
        ...

    @abc.abstractmethod
    def get_dee_pe(self) -> np.MatrixType:
        ...


class IPropertyPS(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def thickness(self) -> float:
        ...

    @abc.abstractmethod
    def get_dee_ps(self) -> np.MatrixType:
        ...


class IPropertyAX(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_dee_ax(self) -> np.MatrixType:
        ...


# **********************************************************
#   Element: 要素データ
# **********************************************************
class ElementBase(LabeledEntity, metaclass=abc.ABCMeta):
    __slots__ = ["_property", "__nodes"]

    NNODE: typing.ClassVar[int]
    PROPERTY_INTERFACES: list[type]

    @classmethod
    def get_card_initializer(cls) -> str:
        return "uu" + "u" * cls.NNODE

    @property
    def nodes(self) -> cabc.Sequence[Node]:
        return self.__nodes

    def __init__(self, label: int, property: PropertyBase, nodes: cabc.Sequence[Node]):
        assert len(nodes) == self.NNODE
        super().__init__(label)
        self._property: PropertyBase = property
        self.__nodes: cabc.Sequence[Node] = nodes
        return

    @classmethod
    def get_instance(cls, label: int, property: PropertyBase, nodes: cabc.Sequence[Node]) -> typing.Self:
        if not isinstance(property, tuple(cls.PROPERTY_INTERFACES)):
            raise RuntimeError("The specified property label.{:d} (type:{:s}) cannot be used in element {:s}".format(
                property.label, property.TYPE, cls.TYPE))
        if len(nodes) != cls.NNODE:
            raise RuntimeError("Invalid number of nodes are specified: {:d} expected, but {:d} found.".format(len(nodes), cls.NNODE))
        return cls(label, property, nodes)

    @abc.abstractmethod
    def calc_stiffness_matrix(self, coords: np.MatrixType) -> np.MatrixType:
        ...

    @abc.abstractmethod
    def calc_result(self, coords: np.MatrixType, ans_elem: np.VectorType) -> cabc.Sequence[np.VectorType]:
        ...

    def get_lm(self) -> cabc.Sequence[int]:
        return [dof for node in self.__nodes for dof in node.dof]

    def get_coord_matrix(self) -> np.MatrixType:
        return np.vstack([it.coord for it in self.__nodes])


class IElementEdge:
    NNODE: typing.ClassVar[int]

    @classmethod
    def get_num_edges(cls) -> int:
        return len(lib.get_element_edge_ids(cls.NNODE))

    @classmethod
    def _get_edge_ids(cls, iedge: int) -> cabc.Sequence[int]:
        edge_ids = lib.get_element_edge_ids(cls.NNODE)
        return edge_ids[iedge - 1]

    def calc_edge_load(self, coords: np.MatrixType, iedge: int, value: float, direction: tuple[float, float] | None = None) -> np.VectorType:
        ndim: int = coords.shape[1]
        ellabels = self._get_edge_ids(iedge)
        coords_edge: np.MatrixType = np.vstack([coords[i, :] for i in ellabels], )
        load_edge: np.VectorType = self._calc_edge_load(
            coords_edge, value, direction)
        load: np.VectorType = np.zeros(coords.shape[0] * ndim)
        for iedge, ielnode in enumerate(ellabels):
            iel: int = ndim * ielnode
            ied: int = ndim * iedge
            load[iel:iel+ndim] += load_edge[ied:ied+ndim]
        return load

    @abc.abstractmethod
    def _calc_edge_load(self, coords_edge: np.MatrixType, value: float, direction: tuple[float, float] | None = None) -> np.VectorType:
        ...


class ElementArray(LabeledEntityArray[ElementBase]):
    __slots__ = []

    TYPE = "element"


# **********************************************************************
# **********************************************************************
# **
# **    モデル全体を表すオブジェクト
# **
# **********************************************************************
# **********************************************************************
@dataclasses.dataclass(frozen=True)
class ModelData:
    nodes: NodeArray
    elements: ElementArray


# **********************************************************************
# **********************************************************************
# **
# **    荷重境界条件データ
# **
# **********************************************************************
# **********************************************************************

# **********************************************************
#   BaseCondition: 荷重境界条件データ
# **********************************************************
class BaseCondition(CardEntity, metaclass=abc.ABCMeta):
    __slots__ = []

    @classmethod
    @abc.abstractmethod
    def get_instance(cls, model: ModelData, ipar: cabc.Sequence[int], fpar: cabc.Sequence[float]) -> typing.Self:
        ...

class LoadBase(BaseCondition):
    __slots__ = []

    @abc.abstractmethod
    def apply(self, model: ModelData, rhs: np.VectorType) -> None:
        ...
