import os
import pathlib
import pytest


def _default_data_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get(
        "HIDALGO_TRADEWEAVE_DATA",
        str(pathlib.Path.home() / "tradeweave" / "data" / "parquet"),
    ))


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    d = _default_data_dir()
    if not (d / "rca_matrix").exists():
        pytest.skip(f"TradeWeave parquet not found at {d}; set HIDALGO_TRADEWEAVE_DATA")
    return d
