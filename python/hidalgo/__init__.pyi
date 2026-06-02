import numpy as np
import numpy.typing as npt

def rca(exports: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def bundle_from_exports(
    exports: npt.NDArray[np.float64], threshold: float = 1.0,
    max_iters: int = 100000, tol: float = 1e-12,
) -> dict: ...
def bundle_from_rca(
    rca: npt.NDArray[np.float64], threshold: float = 1.0,
    max_iters: int = 100000, tol: float = 1e-12,
) -> dict: ...
def bundle_from_m(
    m: npt.NDArray[np.float64], max_iters: int = 100000, tol: float = 1e-12,
) -> dict: ...
