from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from VMEC2GK.utils import extract_essence, derm, dermv

# TODO test ifft_routine

# -------------------------
# extract_essence tests


@pytest.mark.parametrize("mode", (0, 1))
def test_extract_essence(mode):
    arr = np.arange(144).reshape((6, 24))
    result = extract_essence(arr, 12, mode)
    # Expect resulting array to contain numbers 0-11, 24-33, 48-59, etc
    for idx, row in enumerate(result):
        if mode:
            assert_array_equal(row, np.arange(idx * 24, idx * 24 + 12))
        else:
            assert_array_equal(row, np.flip(np.arange(idx * 24, idx * 24 + 12)))


# -------------------------
# derm tests


class TestDerm:
    """
    We test derm and dermv using known functions and their derivatives. As finite
    difference routines, they should return the correct values within a few orders of
    magnitude.
    """

    def _gaussian_1d(self, x, mean, var):
        """Gaussian function with maximum magnitude 1.0"""
        return np.exp(-0.5 * (x - mean) ** 2 / var)

    def _gaussian_derivative_1d(self, x, mean, var):
        """First derivative of _gaussian_1d"""
        return self._gaussian_1d(x, mean, var) * (mean - x) / var

    def _gaussian_2d(self, x, y, mean_x, mean_y, var_x, var_y):
        """
        2D Gaussian function with maximum magnitude 1.0.
        Should be supplied with 2D arrays for x and y, as the grid may be irregular.
        """
        return np.exp(
            -0.5 * (x - mean_x) ** 2 / var_x - 0.5 * (y - mean_y) ** 2 / var_y
        )

    def _gaussian_derivative_2d(self, x, y, mean_x, mean_y, var_x, var_y, dim="x"):
        """First derivative of _gaussian_2d in the 'dim' direction"""
        if dim != "x" and dim != "y":
            raise ValueError("dim must be 'x' or 'y'")
        prefix = (mean_x - x) / var_x if dim == "x" else (mean_y - y) / var_y
        return prefix * self._gaussian_2d(x, y, mean_x, mean_y, var_x, var_y)

    def _check_results(self, result, expected, ch, par):
        """
        Perform assertions to ensure the result of a derm/dermv call matches the
        expected. Expects the derivative to be in the y-dimension, so the results
        should be transposed if
        """
        # To keep the indexing simple in this function, take the transpose of both the
        # result and the expected values if ch == 'r'
        if ch == "r":
            result, expected = result.T, expected.T
        # Ensure result is of the expected shape
        assert_array_equal(result.shape, expected.shape)
        # Separate results and expected into bulk and boundaries
        result_bulk = result[:, 1:-1]
        expected_bulk = expected[:, 1:-1]
        result_bounds = np.array([result[:, 0], result[:, -1]])
        expected_bounds = np.array([expected[:, 0], expected[:, -1]])
        # Check that the bulk is correct to a reasonable accuracy
        assert_allclose(result_bulk, expected_bulk, rtol=1e-3, atol=1e-8)
        # Check that the boundaries are correct
        if ch == "l" and par == "e":
            # Expect boundaies set to zero
            assert np.all(result_bounds == 0)
        else:
            # Use large rtol to account for boundary elements being a simple first
            # order finite difference.
            assert_allclose(result_bounds, expected_bounds, rtol=1e-1, atol=1e-8)

    @pytest.mark.parametrize("ch,par", product(("l", "r"), ("e", "o")))
    def test_derm_1d(self, ch, par):
        x = np.linspace(5.0, 15.0, 101)
        mean, var = 10, 10
        dx = x[1] - x[0]
        arr = self._gaussian_1d(x, mean, var)
        arr_copy = np.copy(arr)
        result = derm(arr, ch, par)
        # Ensure the input is unchanged,
        assert_array_equal(arr, arr_copy)
        # Check that the results are correct
        # NOTE: derm performs a central finite difference and does not divide by grid
        # spacing, so the result carries a factor of 2*dx. The result will have a new
        # axis in the 0 position if ch == 'l', e.g. (12,) -> (1, 12), or a new axis in
        # the 1 position if ch == 'r', e.g. (12,) -> (12, 1)
        new_axis = ch == "r"
        expected = np.expand_dims(
            2 * dx * self._gaussian_derivative_1d(x, mean, var), axis=new_axis
        )
        self._check_results(result, expected, ch, par)

    @pytest.mark.parametrize("ch,par", product(("l", "r"), ("e", "o")))
    def test_derm_2d(self, ch, par):
        x, y = np.meshgrid(
            np.linspace(5.0, 15.0, 101), np.linspace(20.0, 30.0, 121), indexing="ij"
        )
        mean_x, var_x = 10, 10
        mean_y, var_y = 25, 8
        dx = x[1, 0] - x[0, 0]
        dy = y[0, 1] - y[0, 0]
        arr = self._gaussian_2d(x, y, mean_x, mean_y, var_x, var_y)
        arr_copy = np.copy(arr)
        result = derm(arr, ch, par)
        # Ensure the input is unchanged,
        assert_array_equal(arr, arr_copy)
        # Check that the results are correct
        # NOTE: derm performs a central finite difference and does not divide by grid
        # spacing, so the result carries a factor of 2*dx.
        spacing = dx if ch == "r" else dy
        dim = "x" if ch == "r" else "y"
        expected = (
            2
            * spacing
            * self._gaussian_derivative_2d(x, y, mean_x, mean_y, var_x, var_y, dim)
        )
        self._check_results(result, expected, ch, par)

    @pytest.mark.parametrize("ch,par", product(("l", "r"), ("e", "o")))
    def test_dermv_regular_1d(self, ch, par):
        """
        Repeats test_derm_1d, though uses dermv instead. Expected results vary only in
        that they don't require multiplication by 2*dx.
        """
        new_axis = int(ch == "r")
        x = np.linspace(5.0, 15.0, 101)
        grid = np.expand_dims(x, axis=new_axis)
        mean, var = 10, 10
        arr = self._gaussian_1d(x, mean, var)
        arr_copy = np.copy(arr)
        result = dermv(arr, grid, ch, par)
        # Ensure the input is unchanged,
        assert_array_equal(arr, arr_copy)
        # Check that the results are correct
        expected = np.expand_dims(
            self._gaussian_derivative_1d(x, mean, var), axis=new_axis
        )
        self._check_results(result, expected, ch, par)

    @pytest.mark.parametrize("ch,par", product(("l", "r"), ("e", "o")))
    def test_dermv_regular_2d(self, ch, par):
        """
        Repeats test_derm_2d, though uses dermv instead. Expected results vary only in
        that they don't require multiplication by 2*dx.
        """
        x, y = np.meshgrid(
            np.linspace(5.0, 15.0, 101), np.linspace(20.0, 30.0, 121), indexing="ij"
        )
        grid = x if ch == "r" else y
        mean_x, var_x = 10, 10
        mean_y, var_y = 25, 8
        arr = self._gaussian_2d(x, y, mean_x, mean_y, var_x, var_y)
        arr_copy = np.copy(arr)
        result = dermv(arr, grid, ch, par)
        # Ensure the input is unchanged,
        assert_array_equal(arr, arr_copy)
        # Check that the results are correct
        dim = "x" if ch == "r" else "y"
        expected = self._gaussian_derivative_2d(x, y, mean_x, mean_y, var_x, var_y, dim)
        self._check_results(result, expected, ch, par)

    @pytest.mark.parametrize("ch,par", product(("l",), ("e", "o")))
    def test_dermv_irregular_1d(self, ch, par):
        """
        Similar to test_derm_regular_1d, though features a slightly irregular grid.
        Uses higher resolution to maintain accuracy.
        """
        # FIXME Not working when ch == "r". Should this test case use a regular grid
        # in all cases?
        new_axis = int(ch == "r")
        x_regular = np.linspace(5.0, 15.0, 301)
        dx_regular = x_regular[1] - x_regular[0]
        np.random.seed(0)
        x = x_regular + 0.2 * dx_regular * np.random.random_sample(x_regular.shape)
        grid = np.expand_dims(x, axis=new_axis)
        mean, var = 10, 10
        arr = self._gaussian_1d(x, mean, var)
        arr_copy = np.copy(arr)
        result = dermv(arr, grid, ch, par)
        # Ensure the input is unchanged,
        assert_array_equal(arr, arr_copy)
        # Check that the results are correct
        expected = np.expand_dims(
            self._gaussian_derivative_1d(x, mean, var), axis=new_axis
        )
        self._check_results(result, expected, ch, par)

    @pytest.mark.parametrize("ch,par", product(("l", "r"), ("e", "o")))
    def test_dermv_irregular_2d(self, ch, par):
        """
        Similar to test_derm_regular_2d, though features a slightly irregular grid.
        Uses higher resolution to maintain accuracy.
        """
        # Create regular grids
        x_regular, y_regular = np.meshgrid(
            np.linspace(5.0, 15.0, 301), np.linspace(20.0, 30.0, 361), indexing="ij"
        )
        # Add random small distortions in the derivative dimension
        np.random.seed(0)
        if ch == "r":
            dx_regular = x_regular[1, 0] - x_regular[0, 0]
            x = x_regular + 0.2 * dx_regular * np.random.random_sample(x_regular.shape)
            y = y_regular
            grid = x
        else:
            dy_regular = y_regular[0, 1] - y_regular[0, 0]
            y = y_regular + 0.2 * dy_regular * np.random.random_sample(y_regular.shape)
            x = x_regular
            grid = y
        # Continue as usual
        mean_x, var_x = 10, 10
        mean_y, var_y = 25, 8
        arr = self._gaussian_2d(x, y, mean_x, mean_y, var_x, var_y)
        arr_copy = np.copy(arr)
        result = dermv(arr, grid, ch, par)
        # Ensure the input is unchanged,
        assert_array_equal(arr, arr_copy)
        # Check that the results are correct
        dim = "x" if ch == "r" else "y"
        expected = self._gaussian_derivative_2d(x, y, mean_x, mean_y, var_x, var_y, dim)
        self._check_results(result, expected, ch, par)
