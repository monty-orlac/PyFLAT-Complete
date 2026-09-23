import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import abc
import math

import scipy.sparse

# **********************************************************************
# **********************************************************************
# **
# **    行列・ベクトルの表示に使用する各種関数の定義
# **
# **********************************************************************
# **********************************************************************
def print_vector(fp: typing.TextIO, rhs: np.VectorType):
    for i in range(len(rhs)):
        fp.write("  {:02d}: {:15.6e}\n".format(i + 1, rhs[i]))


class IMatrixPrint(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def _iter_component(self) -> cabc.Iterator[tuple[int, int, float]]:
        ...

    def print_matrix(self, fp: typing.TextIO) -> None:
        assert self.size <= 100

        valmax: float = 0.0
        for _, _, value in self._iter_component():
            valmax = max(valmax, abs(value))
                
        iexp = int(math.log10(valmax) + 0.01)
        fp.write(f"scaling factor for matrix elements:       x10^{iexp:d}\n")
        irnew: int
        icnew: int
        value: float
        comp_iterator: cabc.Iterator[tuple[int, int, float]] = self._iter_component()
        (irnew, icnew, value) = comp_iterator.__next__()
        for irow in range(self.size):
            fp.write(f"  {irow+1:2d}: [")
            for icol in range(self.size):
                if irow == irnew and icol == icnew:
                    fp.write("{:7.3f}".format(value / 10.0**iexp))
                    try:
                        (irnew, icnew, value) = comp_iterator.__next__()
                    except StopIteration:
                        irnew = self.size + 1
                    continue
                elif irow < icol:
                    fp.write("       ")
                else:
                    fp.write("    ...")
            fp.write(" ]\n")
        return

    @classmethod
    def _iter_component_lil(cls, matrix: scipy.sparse.lil_array) -> cabc.Iterator[tuple[int, int, float]]:
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
        for irow in range(matrix.shape[0]):
            for icol, value in zip(matrix.rows[irow], matrix.data[irow]):
                if irow < icol:
                    continue
                yield irow, icol, value


# **********************************************************************
# **********************************************************************
# **
# **    行列ソルバーのフレームワーク定義
# **
# **********************************************************************
# **********************************************************************

# **********************************************************
#  IFactorized: 分解済みの行列を保持するクラス
# **********************************************************
class IFactorized(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def solve(self, rhs: np.VectorType, fp: typing.TextIO | None = None) -> np.VectorType:
        ...


# **********************************************************
#  ISolver: 構築済み（分解前）の全体剛性行列を保持するクラス
# **********************************************************
class ISolver(IMatrixPrint, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def factorize(self, fp: typing.TextIO | None) -> IFactorized:
        ...

    @classmethod
    def _print_summary(cls, fp: typing.TextIO, nsize: int, nelem: int, header: str="  ") -> None:
        fp.write(f"{header:s}number of equations        ={nsize:12d}\n")
        fp.write(f"{header:s}number of matrix elements  ={nelem:12d}\n")
        density: float = float(nelem) / float(nsize ** 2) * 100.0
        fp.write(f"{header:s}matrix density             = {density:8.4f}[%]\n")
        return


# **********************************************************
#  IBuilder: 全体剛性行列の構築を行うクラス
# **********************************************************
class IBuilder(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def add_value(self, i: int, j: int, value: float) -> None:
        ...

    def assemble(self, lm: cabc.Sequence[int], matrix_local: np.MatrixType) -> None:
        assert matrix_local.shape == (len(lm), len(lm))

        for il, _ig in enumerate(lm):
            if _ig < 0:
                continue
            for jl, _jg in enumerate(lm[0:il+1]):
                if _jg < 0:
                    continue
                ig: int = max(_ig, _jg)
                jg: int = min(_ig, _jg)
                self.add_value(ig, jg, matrix_local[(il, jl)])
        return

    @abc.abstractmethod
    def complete(self) -> ISolver:
        ...


# **********************************************************
#  IShape: 非ゼロ構造を決定する段階の行列データを格納するクラス
# **********************************************************
class IShape(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def assemble(self, lm: cabc.Sequence[int]) -> None:
        ...

    @abc.abstractmethod
    def allocate(self, spcs: cabc.Mapping[int, float]) -> IBuilder:
        ...


class IGenerator(metaclass=abc.ABCMeta):

    TYPE: typing.ClassVar[str]
    DESC: typing.ClassVar[str]
    NEEDS_REORDER: typing.ClassVar[bool]

    @abc.abstractmethod
    def generate(self, nsize: int) -> IShape:
        ...
