import pytest
import numpy as np
from q3dfm.helper_mat import helper_mat


@pytest.mark.parametrize(
    "fq, is_diff, r, m, expected",
    [
        # Happy path tests
        (
            2,
            True,
            2,
            6,
            np.array(
                [
                    [0.5, 0.0, 1.0, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 1.0, 0.0, 0.5],
                ]
            ),
        ),
        (
            2,
            False,
            2,
            4,
            np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]]),
        ),
        # Edge cases
        (1, True, 1, 1, np.array([[1.0]])),
        (1, False, 1, 1, np.array([[1.0]])),
        (2, True, 0, 4, np.zeros((0, 4))),
        # Error cases
        (2, True, -1, 4, ValueError),
        (2, False, 2, -1, ValueError),
    ],
    ids=[
        "happy_path_diff",
        "happy_path_no_diff",
        "edge_case_fq_1_diff",
        "edge_case_fq_1_no_diff",
        "edge_case_zero_rows",
        "error_case_negative_rows",
        "error_case_negative_columns",
    ],
)
def test_helper_mat(fq, is_diff, r, m, expected):
    # Act
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            helper_mat(fq, is_diff, r, m)
    else:
        result = helper_mat(fq, is_diff, r, m)

        # Assert
        np.testing.assert_array_equal(result, expected)
