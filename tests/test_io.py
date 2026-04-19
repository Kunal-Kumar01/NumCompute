"""Tests for the IO module."""

from pathlib import Path

import numpy as np
import pytest

from numcompute.io import load_data


def test_load_data_reads_numeric_csv_and_skips_header(tmp_path: Path) -> None:
	
	csv_file = tmp_path / "numeric_data.csv"
	csv_file.write_text(
		"col1,col2,col3\n"
		"1,2,3\n"
		"4,,6\n"
		"7,8,9\n",
		encoding="utf-8",
	)

	result = load_data(str(csv_file), skip_header=1)

	expected = np.array(
		[
			[1.0, 2.0, 3.0],
			[4.0, np.nan, 6.0],
			[7.0, 8.0, 9.0],
		]
	)

	assert isinstance(result, np.ndarray)
	np.testing.assert_array_equal(result, expected)


def test_load_data_raises_file_not_found_error_for_missing_file(tmp_path: Path) -> None:
	"""load_data should fail fast when the CSV path does not exist."""

	missing_file = tmp_path / "does_not_exist.csv"

	with pytest.raises(FileNotFoundError, match=r"file not found"):
		load_data(str(missing_file))
