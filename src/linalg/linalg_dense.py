from .linalg_base import * 
import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import scipy


# **********************************************************************
# **********************************************************************
# **
# **    Scipyの密行列ソルバーを使った実装 (Cholesky分解)
# **
# **********************************************************************
# **********************************************************************
class ScipyDenseFactorized(IFactorized):
    __slots__ = ["__nsize", "__factorized"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int, factorized: tuple[np.MatrixType, bool]):
        self.__nsize: int = nsize
        self.__factorized: tuple[np.MatrixType, bool] = factorized
        return

    def solve(self, rhs: np.VectorType, fp: typing.TextIO | None = None) -> np.VectorType:
        if fp is not None:
            fp.write("Performing forward/backward substitution...\n")
        ans: np.VectorType = scipy.linalg.cho_solve(self.__factorized, rhs, overwrite_b=True)
        if fp is not None:
            fp.write("  Done.\n")
        return ans


class ScipyDenseSolver(IBuilder, ISolver):
    __slots__ = ["__matrix"]

    @property
    def size(self) -> int:
        return self.__matrix.shape[0]

    def __init__(self, nsize: int):
        self.__matrix: np.MatrixType = np.zeros((nsize, nsize))
        return

    def add_value(self, i: int, j: int, value: float) -> None:
        assert i >= j
        self.__matrix[i, j] += value
        return
    
    def complete(self) -> ISolver:
        return self
    
    def factorize(self, fp: typing.TextIO | None) -> IFactorized:
        if fp is not None:
            fp.write("Matrix profile summary:\n")
            self._print_summary(fp, self.size, self.size**2)
            fp.write("\n")

        if fp is not None:
            fp.write("Perform Cholesky decomposition...\n")
        factorized: tuple[np.MatrixType, bool] = scipy.linalg.cho_factor(self.__matrix, lower=True, overwrite_a=True)  # type: ignore
        if fp is not None:
            fp.write("  Done.\n")

        return ScipyDenseFactorized(self.size, factorized)

    def _iter_component(self) -> cabc.Iterator[tuple[int, int, float]]:
        matrix: np.MatrixType = self.__matrix
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
        for irow in range(matrix.shape[0]):
            for icol in range(irow + 1):
                yield irow, icol, matrix[irow, icol]
        return


class ScipyDenseShape(IShape):
    __slots__ = ["__nsize"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int):
        self.__nsize: int = nsize
        return
    
    def assemble(self, lm: cabc.Sequence[int]) -> None:
        return

    def allocate(self, spcs: cabc.Mapping[int, float]) -> IBuilder:
        return ScipyDenseSolver(self.__nsize)


class ScipyDenseGenerator(IGenerator):
    __slots__ = []

    TYPE = "DENSE"
    DESC = "Dense matrix solver using Cholesky decomposition (via SciPy)."
    NEEDS_REORDER = False

    def generate(self, nsize: int) -> IShape:
        return ScipyDenseShape(nsize)
