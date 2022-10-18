from pathlib import Path
from glob import glob
from itertools import chain
import json

import pytest
import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal, assert_allclose

from VMEC2GK import (
    vmec_to_bishop,
    bishop_to_gx,
    bishop_to_gs2,
    read_gs2_grid_file,
    plot_ballooning_scan,
)


@pytest.fixture(scope="module")
def test_params():
    """
    Get parameters defining this test. Used by 'input_data' and 'expected' fixtures.
    """
    test_data_dir = Path(__file__).parent / "test_data" / "b2f0aea"
    with open(test_data_dir / "test.json") as f:
        params = json.load(f)
    # Set files to absolute paths
    for key in ["input", "gs2_output", "gx_output", "ball_scan"]:
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


@pytest.fixture
def expected_gx(test_params):
    """
    Reads GX netcdf of expected data. Should be used to compare against current code
    outputs.
    """
    with xr.open_dataset(test_params["gx_output"]) as ds:
        return ds.load()


@pytest.fixture
def expected_gs2(test_params):
    """
    Reads GS2 grid file of expected data. Should be used to compare against current code
    outputs.
    """
    return read_gs2_grid_file(test_params["gs2_output"])


@pytest.fixture
def expected_ball_scan(test_params):
    """
    Reads expected ball scan data file. Should be used to compare against current code
    outputs.
    """
    with xr.open_dataset(test_params["ball_scan"]) as ds:
        return ds.load()


def _check_datasets(actual, expected):
    # Compare attrs
    for k, v in expected.attrs.items():
        assert np.isclose(getattr(actual, k), v, rtol=1e-5, atol=1e-8)
    # Compare dims and data vars
    for k in chain(expected.dims, expected.data_vars):
        # Ensure shapes are the same
        assert_array_equal(expected[k].shape, actual[k].shape)
        # Ensure values are close enough
        assert_allclose(expected[k], actual[k], equal_nan=True, rtol=1e-5, atol=1e-8)


def test_vmec_to_gx(tmp_path, input_data, expected_gx):
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
        _check_datasets(actual, expected_gx)


def test_vmec_to_gs2(tmp_path, input_data, expected_gs2):
    # Make directory to store new GX file
    output_dir = tmp_path / "GS2_grid_files"
    output_dir.mkdir()
    # Create GX file
    bishop_to_gs2(input_data, output_dir=output_dir)
    # Ensure file was created successfully
    output_files = glob(str(output_dir / "grid.out*"))
    assert len(output_files) == 1
    # Compare actual to expected
    _check_datasets(read_gs2_grid_file(output_files[0]), expected_gs2)


def test_vmec_ball_scan(tmp_path, input_data, expected_ball_scan):
    # Make directory to store new GX file
    output_dir = tmp_path / "ball_scan_files"
    output_dir.mkdir()
    # Create GX file
    plot_ballooning_scan(
        input_data, output_dir=output_dir, len_shat_grid=10, len_dpdpsi_grid=10
    )
    # Ensure plot was successful
    plot_files = glob(str(output_dir / "*.png"))
    assert len(plot_files) == 1
    # Ensure data file was created successfully
    output_files = glob(str(output_dir / "*.nc"))
    assert len(output_files) == 1
    # Compare actual to expected
    with xr.open_dataset(output_files[0]) as actual:
        _check_datasets(actual, expected_ball_scan)
