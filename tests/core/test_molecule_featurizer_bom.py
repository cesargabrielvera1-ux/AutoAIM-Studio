"""Tests for MoleculeFeaturizer .xyz parser edge cases (v2.0).

Covers BOM (Byte Order Mark), Windows line endings, and malformed files.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.core.molecule_featurizer import MoleculeFeaturizer


class TestXYZParserEdgeCases:
    """Test _parse_xyz robustness against real-world file formats."""

    @pytest.fixture
    def mf(self):
        return MoleculeFeaturizer()

    @pytest.fixture
    def tmpdir(self):
        return Path(tempfile.mkdtemp())

    def test_bom_utf8(self, mf, tmpdir):
        """Windows-generated .xyz with UTF-8 BOM must parse correctly."""
        bom_content = b'\xef\xbb\xbf3\nWater\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n'
        path = tmpdir / "bom.xyz"
        path.write_bytes(bom_content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 3

    def test_windows_line_endings(self, mf, tmpdir):
        """\r\n line endings must be handled."""
        content = "3\r\nWater\r\nO 0.0 0.0 0.0\r\nH 1.0 0.0 0.0\r\nH 0.0 1.0 0.0\r\n"
        path = tmpdir / "win.xyz"
        path.write_text(content, newline='')
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 3

    def test_tabs_separator(self, mf, tmpdir):
        """Tab-separated coordinates must parse."""
        content = "3\nWater\nO\t0.0\t0.0\t0.0\nH\t1.0\t0.0\t0.0\nH\t0.0\t1.0\t0.0\n"
        path = tmpdir / "tabs.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1

    def test_extra_leading_spaces(self, mf, tmpdir):
        """Leading spaces in coordinate lines must be trimmed."""
        content = "3\nWater\n  O   0.000   0.000   0.000\n  H   1.000   0.000   0.000\n  H   0.000   1.000   0.000\n"
        path = tmpdir / "spaces.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1

    def test_trailing_empty_lines(self, mf, tmpdir):
        """Trailing empty lines must not cause issues."""
        content = "3\nWater\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n\n\n"
        path = tmpdir / "trailing.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1

    def test_comment_after_n_atoms(self, mf, tmpdir):
        """Second line treated as comment, not parsed."""
        content = "3\nThis is a comment line with numbers 123\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n"
        path = tmpdir / "comment.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 3

    def test_empty_file_raises(self, mf, tmpdir):
        """Empty file must raise ValueError with clear message."""
        path = tmpdir / "empty.xyz"
        path.write_text("")
        with pytest.raises(ValueError, match="Empty .xyz file"):
            mf.featurize_file(str(path))

    def test_only_comment_raises(self, mf, tmpdir):
        """File with only n_atoms + comment must raise ValueError."""
        path = tmpdir / "no_atoms.xyz"
        path.write_text("3\nWater\n")
        with pytest.raises(ValueError, match="No valid atoms"):
            mf.featurize_file(str(path))

    def test_wrong_n_atoms_parses_available(self, mf, tmpdir):
        """If n_atoms > actual lines, parse available atoms."""
        content = "5\nWater\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n"
        path = tmpdir / "wrong_count.xyz"
        path.write_text(content)
        # Should still parse the 3 available atoms
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 3

    def test_element_with_numeric_prefix(self, mf, tmpdir):
        """Numeric prefixes like '1C' must be stripped."""
        content = "3\nTest\n1C 0.0 0.0 0.0\n2H 1.0 0.0 0.0\n3H 0.0 1.0 0.0\n"
        path = tmpdir / "numbered.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_carbons'].iloc[0] == 1

    def test_single_atom(self, mf, tmpdir):
        """Single-atom molecule must not crash."""
        content = "1\nSingle atom\nH 0.0 0.0 0.0\n"
        path = tmpdir / "single.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 1

    def test_unknown_element(self, mf, tmpdir):
        """Unknown element symbol does NOT crash — returns default values."""
        content = "2\nUnknown\nXx 0.0 0.0 0.0\nXx 1.0 0.0 0.0\n"
        path = tmpdir / "unknown.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        # Should succeed with n_atoms=2, but periodic properties default to 0
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 2
        # Periodic properties for unknown element should be 0
        assert df['mol_mean_atomic_number'].iloc[0] == 0

    def test_bom_with_large_coords(self, mf, tmpdir):
        """BOM file with scientific notation coordinates."""
        bom_content = b'\xef\xbb\xbf2\nTest\nH 0.0 0.0 0.0\nH 1.0e-10 0.0 0.0\n'
        path = tmpdir / "sci_bom.xyz"
        path.write_bytes(bom_content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1

    def test_malformed_no_valid_atoms(self, mf, tmpdir):
        """File with no valid atom lines must raise ValueError."""
        path = tmpdir / "bad_atoms.xyz"
        path.write_text("not_a_number\nComment\nfoo 0 0 0\nbar 1 1 1\n")
        with pytest.raises(ValueError, match="No valid atoms parsed"):
            mf.featurize_file(str(path))

    def test_malformed_standard_xyz_bad_atom_count(self, mf, tmpdir):
        """Standard XYZ where atom count exceeds actual atoms must still parse available."""
        path = tmpdir / "bad_count.xyz"
        path.write_text("5\nWater\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n")
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 3

    def test_featurize_files_skips_bad_ones(self, mf, tmpdir):
        """featurize_files must skip invalid files and process valid ones."""
        # Create one valid and one invalid file
        valid = tmpdir / "water.xyz"
        valid.write_text("3\nWater\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n")
        invalid = tmpdir / "bad.xyz"
        invalid.write_text("")

        df = mf.featurize_files([str(valid), str(invalid)])
        assert df.shape[0] == 1  # Only valid one
        assert df['mol_source_file'].iloc[0] == "water"

    def test_featurize_files_all_invalid_returns_empty(self, mf, tmpdir):
        """featurize_files with all invalid must return empty DataFrame."""
        bad1 = tmpdir / "bad1.xyz"
        bad1.write_text("")
        bad2 = tmpdir / "bad2.xyz"
        bad2.write_text("only_header\n")

        df = mf.featurize_files([str(bad1), str(bad2)])
        assert df.empty


class TestRawXYZFormat:
    """Test parsing of raw coordinate files (no atom count header).

    GaussView, Spartan, and other chemistry packages often export XYZ files
    without the standard two-line header (atom count + comment).
    """

    @pytest.fixture
    def mf(self):
        return MoleculeFeaturizer()

    @pytest.fixture
    def tmpdir(self):
        return Path(tempfile.mkdtemp())

    def test_raw_format_no_header(self, mf, tmpdir):
        """GaussView/Spartan raw format: no atom count, no comment."""
        content = "C 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\nH 0.0 0.0 1.0\n"
        path = tmpdir / "raw.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 4
        assert df['mol_n_carbons'].iloc[0] == 1
        assert df['mol_n_hydrogens'].iloc[0] == 3

    def test_raw_format_mixed_whitespace(self, mf, tmpdir):
        """Raw format with varying whitespace."""
        content = "C    0.0000    0.0000    0.0000\nH    1.0890    0.0000    0.0000\n"
        path = tmpdir / "raw_mixed.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 2

    def test_gaussian_format_with_index(self, mf, tmpdir):
        """Gaussian format: atom index + element + x y z."""
        content = "6\nmethane\n  1  C   0.000000  0.000000  0.000000\n  2  H   1.089000  0.000000  0.000000\n"
        path = tmpdir / "gaussian.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 2

    def test_raw_format_with_leading_trailing_spaces(self, mf, tmpdir):
        """Raw format with leading/trailing spaces per line."""
        content = "  C  0.0  0.0  0.0  \n  H  1.0  0.0  0.0  \n"
        path = tmpdir / "raw_spaces.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 2

    def test_raw_format_empty_lines_ignored(self, mf, tmpdir):
        """Raw format with empty lines between coordinates."""
        content = "C 0.0 0.0 0.0\n\nH 1.0 0.0 0.0\n\n"
        path = tmpdir / "raw_empty_lines.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 2

    def test_raw_format_no_valid_atoms_raises(self, mf, tmpdir):
        """Raw format with no parseable atom lines must raise ValueError."""
        content = "foo bar baz\nqux quux corge\n"
        path = tmpdir / "raw_invalid.xyz"
        path.write_text(content)
        with pytest.raises(ValueError, match="No valid atoms parsed"):
            mf.featurize_file(str(path))

    def test_realistic_gaussview_export(self, mf, tmpdir):
        """Realistic GaussView-style export with aligned columns."""
        content = (
            "C         -7.46978   1.23456   0.00000\n"
            "C         -9.46999   2.34567   1.00000\n"
            "H         -5.12345   3.45678   2.00000\n"
        )
        path = tmpdir / "gaussview.xyz"
        path.write_text(content)
        df = mf.featurize_file(str(path))
        assert df.shape[0] == 1
        assert df['mol_n_atoms'].iloc[0] == 3
        assert df['mol_n_carbons'].iloc[0] == 2
        assert df['mol_n_hydrogens'].iloc[0] == 1
