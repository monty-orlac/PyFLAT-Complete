from .linalg_base import * 
import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import scipy


# **********************************************************************
# **********************************************************************
# **
# **    Scipyの直接法疎行列ソルバーを使った実装
# **
# **********************************************************************
# **********************************************************************
class ScipyDirectFactorized(IFactorized):
    __slots__ = ["__factorized"]

    @property
    def size(self) -> int:
        return self.__factorized.shape[0]

    def __init__(self, factorized: scipy.sparse.linalg.SuperLU):
        self.__factorized: scipy.sparse.linalg.SuperLU = factorized
        return

    def solve(self, rhs: np.VectorType, fp: typing.TextIO | None = None) -> np.VectorType:
        if fp is not None:
            fp.write("Performing forward/backward substitution...\n")
        ans = self.__factorized.solve(rhs)
        if fp is not None:
            fp.write("  Done.\n")
        return ans


class ScipyDirectSolver(ISolver):
    __slots__ = ["__matrix"]

    _APPLY_REORDER: bool = False

    @property
    def size(self) -> int:
        return self.__matrix.shape[0]

    def __init__(self, matrix: scipy.sparse.lil_array):
        self.__matrix: scipy.sparse.lil_array = matrix
        return

    def factorize(self, fp: typing.TextIO | None) -> IFactorized:
        if fp is not None:
            fp.write("Perform general LU decomposition...\n")
        factorized: scipy.sparse.linalg.SuperLU = scipy.sparse.linalg.splu(self.__matrix.tocsc(),
            permc_spec="MMD_AT_PLUS_A" if self._APPLY_REORDER else "NATURAL")

        # 特異性のチェック: 拘束不足で剛体変位が残っていると、ピボット (U の対角項) が
        # 元の対角項に対して丸め誤差程度 (1e-16 程度) まで小さくなる
        pivot_ratio = abs(factorized.U.diagonal()) / abs(self.__matrix.diagonal()[np.argsort(factorized.perm_c)])
        if pivot_ratio.min() < 1.0e-10:
            raise RuntimeError("Near-zero pivot detected. The stiffness matrix is singular (e.g. insufficient displacement constraints).")
        if fp is not None:
            fp.write("  Done.\n")
            fp.write("\n")
            fp.write("Matrix profile summary:\n")
            nelem: int = factorized.L.nnz + factorized.U.nnz - self.size
            self._print_summary(fp, self.size, nelem)

        return ScipyDirectFactorized(factorized)

    def _iter_component(self) -> cabc.Iterator[tuple[int, int, float]]:
        yield from self._iter_component_lil(self.__matrix)
        return


class ScipyDirectBuilder(IBuilder):
    __slots__ = ["__nsize", "__row", "__col", "__data", "__ndata"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int, ntriplet: int):
        self.__nsize: int = nsize
        self.__ndata: int = nsize
        self.__row = np.zeros(ntriplet, dtype=np.int32)
        self.__col: np.IntVectorType = np.zeros(ntriplet, dtype=np.int32)
        self.__data: np.VectorType = np.zeros(ntriplet, dtype=np.float64)

        for i in range(nsize):
            self.__row[i] = i
            self.__col[i] = i
        return

    def add_value(self, i: int, j: int, value: float) -> None:
        if i == j:
            self.__data[i] += value
            return
        assert self.__ndata < len(self.__row) - 1
        self.__row[self.__ndata] = i
        self.__col[self.__ndata] = j
        self.__data[self.__ndata] = value
        self.__ndata += 1

        self.__row[self.__ndata] = j
        self.__col[self.__ndata] = i
        self.__data[self.__ndata] = value
        self.__ndata += 1
        return

    def complete(self) -> ISolver:
        matrix = scipy.sparse.coo_array(
            (self.__data, (self.__row, self.__col)),
            shape=(self.__nsize, self.__nsize), dtype=np.float64).tolil()
        return ScipyDirectSolver(matrix)


class ScipyDirectShape(IShape):
    __slots__ = ["__nsize", "__ntriplet_nodiag"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int):
        self.__nsize: int = nsize
        self.__ntriplet_nodiag: int = 0
        return

    def assemble(self, lm: cabc.Sequence[int]) -> None:
        nloc: int = sum(v >= 0 for v in lm)
        self.__ntriplet_nodiag += nloc * (nloc - 1)
        return

    def allocate(self, spcs: cabc.Mapping[int, float]) -> IBuilder:
        ntriplet: int = self.__ntriplet_nodiag + self.__nsize
        return ScipyDirectBuilder(self.__nsize, ntriplet)


class ScipyDirectGenerator(IGenerator):
    __slots__ = []

    TYPE = "DIRECT"
    DESC = "Fast sparse direct solver (via SciPy)."
    NEEDS_REORDER = True

    def generate(self, nsize: int) -> IShape:
        return ScipyDirectShape(nsize)
