from numpy import *   # pyright: ignore[reportWildcardImportFromLibrary]
type MatrixType = ndarray[tuple[int, int], dtype[float64]]
type VectorType = ndarray[tuple[int, ], dtype[float64]]
type IntMatrixType = ndarray[tuple[int, int], dtype[int32]]
type IntVectorType = ndarray[tuple[int, ], dtype[int32]]

set_printoptions(precision=4, linewidth=1000)
