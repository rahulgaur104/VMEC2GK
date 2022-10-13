from itertools import product

import pytest
import numpy as np

from VMEC2GK.utils import extract_essence


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
