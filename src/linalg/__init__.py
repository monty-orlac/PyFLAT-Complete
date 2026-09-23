from .linalg_base import *
from . import linalg_dense      # pyright: ignore[reportUnusedImport]
from . import linalg_direct     # pyright: ignore[reportUnusedImport]
from . import linalg_iterative  # pyright: ignore[reportUnusedImport]
from . import linalg_skyline    # pyright: ignore[reportUnusedImport]


SOLVER_TYPES: dict[str, type[IGenerator]] = {
    cls.TYPE: cls for cls in IGenerator.__subclasses__() if hasattr(cls, "TYPE")
}
