from itertools import product

import pytest
import numpy as np

from VMEC2GK.utils import extract_essence, derm

# TODO test ifft_routine


@pytest.mark.parametrize("mode", (0, 1))
def test_extract_essence(mode):
    arr = np.arange(144).reshape((6, 24))
    result = extract_essence(arr, 12, mode)
    # Expect resulting array to contain numbers 0-11, 24-33, 48-59, etc
    for idx, row in enumerate(result):
        if mode:
            assert np.array_equal(row, np.arange(idx * 24, idx * 24 + 12))
        else:
            assert np.array_equal(row, np.flip(np.arange(idx * 24, idx * 24 + 12)))


# Derm inputs and answers
# Use a Gaussian with unit max-magnitude, a mean of 0, and a variance of 10
gauss_x = np.linspace(-5.0, 5.0, 101)
gauss_dx = gauss_x[1] - gauss_x[0]
gauss_var = 10
gauss_1d = np.exp(-(gauss_x**2) / gauss_var)
gauss_2d = np.outer(gauss_1d, gauss_1d)
# Derm answers
# Multiply by 2*dx because it takes a central finite difference and derm does not divide
# by the grid spacing.
gauss_deriv_1d = 2 * gauss_dx * (-2 * gauss_x * gauss_1d / gauss_var)[np.newaxis, :]
gauss_deriv_2d = 2 * gauss_dx * (-2 * gauss_x * gauss_2d / gauss_var)


@pytest.mark.parametrize(
    "arr,ch,par,expected",
    [
        (gauss_1d, "l", "e", gauss_deriv_1d),
        (gauss_1d, "r", "e", gauss_deriv_1d),
        (gauss_1d, "l", "o", gauss_deriv_1d),
        (gauss_1d, "r", "o", gauss_deriv_1d),
        (gauss_2d, "l", "e", gauss_deriv_2d),
        (gauss_2d, "r", "e", gauss_deriv_2d),
        (gauss_2d, "l", "o", gauss_deriv_2d),
        (gauss_2d, "r", "o", gauss_deriv_2d),
    ],
)
def test_derm(arr, ch, par, expected):
    temp = np.copy(arr)
    result = derm(temp, ch, par)
    # Ensure arr is unchanged (and make sure we're not just looking at the same object)
    assert np.array_equal(arr, temp)
    assert temp is not arr
    # To simplify these checks, take the transpose of result if ch == "r".
    if ch == "r":
        result = result.T
    # Ensure result is of the expected shape
    assert np.array_equal(result.shape, expected.shape)
    # Separate results and expected into bulk and boundaries
    result_bulk = result[:, 1:-1]
    expected_bulk = expected[:, 1:-1]
    result_boundaries = np.array([result[:, 0], result[:, -1]])
    expected_boundaries = np.array([expected[:, 0], expected[:, -1]])
    # Check that the bulk is correct to a reasonable accuracy
    assert np.allclose(result_bulk, expected_bulk, rtol=1e-3)
    # Check that the boundaries are correct
    if ch == "l" and par == "e":
        # Expect the boundaries to be zero
        assert np.all(result_boundaries == 0)
    else:
        # Use large rtol to account for boundary elements being a simple first
        # order finite difference.
        assert np.allclose(result_boundaries, expected_boundaries, rtol=1e-1)
