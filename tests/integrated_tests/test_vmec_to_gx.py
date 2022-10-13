from pathlib import Path
from glob import glob
import json

import pytest
import numpy as np
import xarray as xr

from VMEC2GK import vmec_to_bishop, bishop_to_gx


@pytest.fixture(scope="module")
def test_params():
    """
    Get parameters defining this test. Used by 'input_data' and 'expected' fixtures.
    """
    test_data_dir = Path(__file__).parent / "test_data" / "b2f0aea"
    with open(test_data_dir / "test.json") as f:
        params = json.load(f)
    # Set files to absolute paths
    for key in ["input", "gs2_output", "gx_output"]:
        params[key] = (test_data_dir / params[key]).resolve()
    return params


@pytest.fixture(scope="module")
def input_data(test_params):
    """
    Reads VMEC file given data in test_params, returns bishop_dict.
    """
    return vmec_to_bishop(
        test_params["input"],
        surf_idx=test_params["surf_idx"],
        nperiod=test_params["nperiod"],
        norm_scheme=test_params["norm_scheme"],
    )


@pytest.fixture(scope="module")
def expected(test_params):
    """
    Reads GX netcdf of expected data. Should be used to compare against current code
    outputs.
    """
    with xr.open_dataset(test_params["gx_output"]) as ds:
        return ds.load()


def test_vmec_to_gx(tmp_path, input_data, expected):
    # Make directory to store new GX file
    output_dir = tmp_path / "GX_nc_files"
    output_dir.mkdir()
    # Create GX file
    bishop_to_gx(input_data, output_dir=output_dir)
    # Ensure file was created successfully
    output_files = glob(str(output_dir / "*.nc"))
    assert len(output_files) == 1
    # Compare actual to expected
    with xr.open_dataset(output_files[0]) as actual:
        # Compare dims
        for key in expected.dims:
            assert np.array_equal(expected[key], actual[key])
        # Compare data vars
        for key in expected.data_vars:
            # Ensure shapes are the same
            assert np.array_equal(expected[key].shape, actual[key].shape)
            # Ensure values are close enough
            assert np.allclose(expected[key], actual[key], equal_nan=True)
