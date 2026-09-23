from .linalg_base import * 
import pyflat_numpy as np           # pyright: ignore[reportUnusedImport]
import typing                       # pyright: ignore[reportUnusedImport]
import collections.abc as cabc      # pyright: ignore[reportUnusedImport]


# ------------------------------
#  numba による高速化
# ------------------------------
# 本ファイル中では計算量の多い関数に対して numba による高速化を適用している。
# numba を使わない場合は、下記の設定行並びに @numba.njit デコレータ行を
# 全て削除orコメントアウトするだけで良い。

import numba

# NumbaのLLVMコード生成において AVX最適化を有効にする
numba.config.NUMBA_ENABLE_AVX = 1


# **********************************************************************
# **********************************************************************
# **
# **   Skyline格納形式の行列に対する主要アルゴリズム
# **
# **********************************************************************
# **********************************************************************
def _get_index(mtptr: np.IntVectorType, irow: int, icol: int) -> int:
    assert icol <= irow
    assert irow < len(mtptr) - 1
    index: int = mtptr[irow + 1] - (irow + 1) + icol
    if index < mtptr[irow]:
        return -1
    return index


# **********************************************************
#  修正Cholesky分解の実行
# **********************************************************
@numba.njit(cache=True)
def _factorize(nsize: int, mtptr: np.ndarray, mtval: np.ndarray) -> None:
    # 配列のサイズチェック
    assert len(mtptr) == nsize + 1
    assert len(mtval) == mtptr[nsize]

    # ------------------------------
    #  修正Cholesky分解の実行
    # ------------------------------
    # 対角項を連続したメモリに格納し、
    # 計算本体でのメモリアクセスを効率化する
    mtdiag = np.empty(nsize, dtype=np.float64)
    for i in range(nsize):
        mtdiag[i] = mtval[mtptr[i + 1] - 1]

    # 全体剛性行列の下三角部分の成分 (i, j) (i > j) をループ
    for i in range(1, nsize):
        # 成分 (i, j) (j = kih, ..., i-1) の計算
        # ki0 ... 行列の仮想的な(i,0)成分
        #   行列の(i,j)成分は mtval[ki0+j] となる
        # ih ... 行列のi行目で最も対角成分から離れた非ゼロ値の列インデックス
        ki0 = mtptr[i + 1] - (i + 1)
        ih = mtptr[i] - ki0

        for j in range(ih, i):
            kj0 = mtptr[j + 1] - (j + 1)
            jh = mtptr[j] - kj0

            # A(i,j) -= sum(A(i,k) * A(k,k) * A(k,j)) for k = 0..jの計算
            ks = max(ih, jh)
            if j > ks:
                vik = mtval[ki0+ks:ki0+j]
                vkk = mtdiag[ks:j]
                vkj = mtval[kj0+ks:kj0+j]

                tot = 0.0
                for k in range(j - ks):
                    tot += vik[k] * vkk[k] * vkj[k]
                mtval[ki0 + j] -= tot

            # A(i,j) /= A(j,j)
            mtval[ki0 + j] /= mtdiag[j]

        # 成分 (i, i) の計算
        if i > ih:
            mik = mtval[ki0+ih:ki0+i]
            mkk = mtdiag[ih:i]

            tot = 0.0
            for k in range(i - ih):
                tot += mik[k] * mkk[k] * mik[k]
            mtdiag[i] -= tot
            mtval[ki0 + i] -= tot

    # ------------------------------
    # 行列が正則かどうかを判定する
    # ------------------------------
    for i in range(nsize):
        if mtdiag[i] <= 0.0:
            raise RuntimeError("Non-positive pivot detected. The stiffness matrix is not positive definite.")
    return


# **********************************************************
#  分解済みの行列を使った求解
# **********************************************************
@numba.njit(cache=True)
def _lu_solve(nsize: int, mtptr: np.ndarray, mtval: np.ndarray, rhs: np.ndarray) -> np.ndarray:

    # ------------------------------
    #  下三角行列を解く
    #    LDL^Tx = b   --> DL^Tx = b'
    # ------------------------------
    for i in range(1, nsize):
        # ki0 ... 行列の仮想的な(i,0)成分
        #   行列の(i,j)成分は mtval[ki0-j] となる
        # ih ... 行列のi行目で最も対角成分から離れた非ゼロ値の列インデックス
        ki0 = mtptr[i + 1] - (i + 1)
        ih = mtptr[i] - ki0
        if ih == i:
            continue

        # v(i) -= sum(A(i,k) * v(k))   for k = h, h+1, ..., i-1
        rhs[i] -= np.dot(mtval[ki0 + ih : ki0 + i], rhs[ih:i])

    # ------------------------------
    #  対角行列を解く
    #    DL^Tx = b   --> L^Tx = b'
    # ------------------------------
    for i in range(0, nsize):
        # v(i) /= A(i,i)
        rhs[i] /= mtval[mtptr[i + 1] - 1]

    # ------------------------------
    #  上三角行列を解く
    #    L^Tx = b   --> x = b'
    # ------------------------------
    for i in range(nsize - 1, 0, -1):
        # ki0 ... 行列の仮想的な(i,0)成分
        #   行列の(i,j)成分は mtval[ki0-j] となる
        # ih ... 行列のi行目で最も対角成分から離れた非ゼロ値の列インデックス
        ki0 = mtptr[i + 1] - (i + 1)
        ih = mtptr[i] - ki0
        if ih == i:
            continue

        # v(k) -= A(k,i) * v(i)    for k = h, h+1, ..., i-1
        val_slice = mtval[ki0+ih:ki0+i]
        rhs_i = rhs[i]
        for k in range(i - ih):
            rhs[ih + k] -= val_slice[k] * rhs_i

    return rhs


# **********************************************************************
# **********************************************************************
# **
# **   Skyline格納形式の行列ソルバーの実装
# **
# **********************************************************************
# **********************************************************************
class SkylineFactorized(IFactorized):
    __slots__ = ["__nsize", "__mtptr", "__mtval"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int, mtptr: np.IntVectorType, mtval: np.VectorType):
        self.__nsize: int = nsize
        self.__mtptr: np.IntVectorType = mtptr
        self.__mtval: np.VectorType = mtval
        return

    def solve(self, rhs: np.VectorType, fp: typing.TextIO | None = None) -> np.VectorType:
        return _lu_solve(self.__nsize, self.__mtptr, self.__mtval, rhs)


class SkylineSolver(IBuilder, ISolver):
    __slots__ = ["__nsize", "__mtptr", "__mtval"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int, mtptr: np.IntVectorType):
        assert len(mtptr) == nsize + 1
        self.__nsize: int = nsize
        self.__mtptr: np.IntVectorType = mtptr
        self.__mtval: np.VectorType = np.zeros(mtptr[nsize])
        return

    def add_value(self, i: int, j: int, value: float) -> None:
        index: int = _get_index(self.__mtptr, i, j)
        assert index >= 0
        self.__mtval[index] += value
        return

    def complete(self) -> ISolver:
        return self

    def _iter_component(self) -> cabc.Iterator[tuple[int, int, float]]:
        for i in range(self.size):
            # ki0 ... 行列の仮想的な(i,0)成分
            #   行列の(i,j)成分は mtval[ki0-j] となる
            # ih ... 行列のi行目で最も対角成分から離れた非ゼロ値の列インデックス
            ki0: int = self.__mtptr[i+1] - (i+1)
            ih: int = self.__mtptr[i] - ki0
            for j in range(ih, i+1):
                yield i, j, self.__mtval[ki0+j]
        return

    def factorize(self, fp: typing.TextIO | None) -> IFactorized:
        if fp is not None:
            fp.write("Matrix profile summary:\n")
            nelem: int = len(self.__mtval) * 2 - self.__nsize
            self._print_summary(fp, self.__nsize, nelem)
            fp.write("\n")

        if fp is not None:
            fp.write("Perform Cholesky decomposition...\n")
        _factorize(self.__nsize, self.__mtptr, self.__mtval)
        factorized: SkylineFactorized = SkylineFactorized(self.__nsize, self.__mtptr, self.__mtval)
        if fp is not None:
            fp.write("  Done.\n")
        return factorized


class SkylineShape(IShape):
    __slots__ = ["__nsize", "__height"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int):
        self.__nsize: int = nsize
        self.__height: np.IntVectorType = np.zeros(nsize+1, dtype=np.int32)
        return

    def assemble(self, lm: cabc.Sequence[int]) -> None:
        # 全自由度拘束されていたら終了
        if all(it < 0 for it in lm):
            return

        # 配列 lm[] のうち None 以外で最初の値を minindex に保存
        minidx: int = min([idx for idx in lm if idx >= 0])

        # 配列 lm[] のうち None 以外のものでループ
        for idx in lm:
            if idx < 0:
                continue
            # 成分 (index, minindex) は非ゼロになるため、
            # 対角項からの高さを記録
            self.__height[idx] = max(self.__height[idx], idx - minidx)
        return

    def allocate(self, spcs: cabc.Mapping[int, float]) -> IBuilder:
        mtptr: np.IntVectorType = self.__height.copy()
        for index in spcs:
            mtptr[index] = 0
        mtptr[self.__nsize] = mtptr.sum() + self.__nsize
        for i in range(self.__nsize - 1, -1, -1):
            mtptr[i] = mtptr[i + 1] - mtptr[i] - 1
        return SkylineSolver(self.__nsize, mtptr)


class SkylineGenerator(IGenerator):
    __slots__ = []

    TYPE = "SKYLINE"
    DESC = "Sparse solver implementation using Skyline storage format."
    NEEDS_REORDER = True

    def generate(self, nsize: int) -> IShape:
        return SkylineShape(nsize)
