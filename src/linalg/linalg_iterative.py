from .linalg_base import * 
import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]

import dataclasses

import scipy.sparse                 # pyright: ignore[reportUnusedImport]
import scipy.sparse.linalg


# **********************************************************************
# **********************************************************************
# **
# **    Scipyの反復法疎行列ソルバーを使った実装 (CG法を利用)
# **
# **********************************************************************
# **********************************************************************
class ScipyCGFactorized(IFactorized):
    __slots__ = ["__matrix", "__precon", "__diag_scale"]

    # 対角スケーリングを有効にするためのスイッチ
    # 系に剛性が極端に違う複数の物体が存在している場合には有効。
    _APPLY_DIAGONAL_SCALING: bool = True

    # 不完全LU分解を有効にするためのスイッチ
    _APPLY_PRECONDITIONER: bool = True

    @property
    def size(self) -> int:
        return self.__matrix.shape[0]

    def __init__(self, matrix_csc: scipy.sparse.csc_array):

        # 対角スケーリングの適用
        self.__diag_scale: scipy.sparse.dia_matrix[np.float64] | None = None
        if self._APPLY_DIAGONAL_SCALING:
            diag_array = matrix_csc.diagonal()
            diag_array_inv_sqrt = 1.0 / np.sqrt(np.where(diag_array > 0, diag_array, 1.0))
            self.__diag_scale = scipy.sparse.diags(diag_array_inv_sqrt, dtype=np.float64)
            matrix_csc = (self.__diag_scale @ matrix_csc @ self.__diag_scale).tocsc()  # type: ignore

        # 不完全LU分解の実施
        self.__precon: scipy.sparse.linalg.LinearOperator | None = None
        if self._APPLY_PRECONDITIONER:
            # 不完全LU分解の scipy.sparse.linalg.spilu() は対称性を前提としないLU分解。
            # 対称性を極力維持するため、下記の設定を行っている。
            #   - permc_spec='MMD_AT_PLUS_A' により対称性を保つ並び替えを強制
            #   - options 内の diag_pivot_thresh=1.0 でピボット交換による対称性の崩れを防ぐ
            # 精度 drop_tol を小さくすると不完全LU分解にかかる時間が長くなるが、
            # 反復回数を減らすことができる (デフォルト 1.0e-4)
            ilu = scipy.sparse.linalg.spilu(matrix_csc, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=1.0, drop_tol=1.0e-4)
            self.__precon = scipy.sparse.linalg.LinearOperator(shape=matrix_csc.shape, dtype=np.float64, matvec=ilu.solve)

        self.__matrix: scipy.sparse.csc_array[np.float64] = matrix_csc
        return

    def solve(self, rhs: np.VectorType, fp: typing.TextIO | None = None) -> np.VectorType:

        # CG法の反復回数をモニタリングするためのカウンター
        # scipy.sparse.linalg.cg()のコールバック関数として渡す
        @dataclasses.dataclass
        class IterationCounter:
            count: int = 0

            def __call__(self, _xk: np.VectorType) -> None:
                self.count += 1  # 呼ばれるたびにカウントアップ
                return

        if fp is not None:
            fp.write("Executing iterative solver...\n")

        # 対角スケーリング
        if self.__diag_scale is not None:
            rhs = self.__diag_scale @ rhs   # pyright: ignore[reportAssignmentType]

        ans: np.VectorType
        info: int
        counter = IterationCounter()
        ans, info = scipy.sparse.linalg.cg(self.__matrix, rhs,
            M=self.__precon, callback=counter, rtol=1.0e-8, atol=0.0)

        # 対角スケーリングをもとに戻す
        if self.__diag_scale is not None:
            ans = self.__diag_scale @ ans   # pyright: ignore[reportAssignmentType]

        if info > 0:
            raise RuntimeError("Convergence to tolerance not achieved.")
        if fp is not None:
            fp.write(f"  Done. Total {counter.count:d} iteration(s) were executed.\n")
        return ans


class ScipyCGSolver(ISolver):
    __slots__ = ["__matrix"]

    @property
    def size(self) -> int:
        return self.__matrix.shape[0]

    def __init__(self, matrix: scipy.sparse.lil_array):
        self.__matrix: scipy.sparse.lil_array = matrix
        return

    def factorize(self, fp: typing.TextIO | None) -> IFactorized:
        matrix: scipy.sparse.csc_array = self.__matrix.tocsc()
        if fp is not None:
            fp.write("  Done.\n")
            fp.write("\n")
            fp.write("Matrix profile summary:\n")
            self._print_summary(fp, self.size, matrix.nnz)
        return ScipyCGFactorized(matrix)

    def _iter_component(self) -> cabc.Iterator[tuple[int, int, float]]:
        yield from self._iter_component_lil(self.__matrix)
        return


class ScipyCGBuilder(IBuilder):
    __slots__ = ["__nsize", "__row", "__col", "__data", "__ndata"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int, ntriplet: int):
        self.__nsize: int = nsize
        self.__ndata: int = nsize
        self.__row: np.IntVectorType = np.zeros(ntriplet, dtype=np.int32)
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
        return ScipyCGSolver(matrix)


class ScipyCGShape(IShape):
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
        return ScipyCGBuilder(self.__nsize, ntriplet)


class ScipyCGGenerator(IGenerator):
    __slots__ = []

    TYPE = "ITERATIVE"
    DESC = "CG (Conjugate Gradient) iterative solver (via SciPy)."
    NEEDS_REORDER = False

    def generate(self, nsize: int) -> IShape:
        return ScipyCGShape(nsize)
