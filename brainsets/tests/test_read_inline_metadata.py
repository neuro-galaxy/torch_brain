"""Tests for _read_inline_metadata function in cli_prepare module."""

import pytest
from pathlib import Path
import tempfile

from brainsets._cli.cli_prepare import _read_inline_metadata


def _write_temp_script(content: str) -> Path:
    """Write content to a temporary file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestReadInlineMetadata:
    """Tests for _read_inline_metadata function."""

    def test_no_metadata_block(self):
        """Returns None when no brainset-pipeline block exists."""
        script = """\
# Just a regular Python script
import os

def main():
    pass
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result is None

    def test_valid_metadata_with_python_version(self):
        """Parses python-version correctly."""
        script = """\
# /// brainset-pipeline
# python-version = "3.10"
# ///

import os
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {"python-version": "3.10"}

    def test_valid_metadata_with_dependencies(self):
        """Parses dependencies list correctly."""
        script = """\
# /// brainset-pipeline
# dependencies = ["numpy", "pandas>=1.5.0"]
# ///

import numpy as np
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {"dependencies": ["numpy", "pandas>=1.5.0"]}

    def test_valid_metadata_with_both_keys(self):
        """Parses both python-version and dependencies."""
        script = """\
# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["scipy", "torch"]
# ///

from scipy import stats
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {
            "python-version": "3.11",
            "dependencies": ["scipy", "torch"],
        }

    def test_empty_dependencies(self):
        """Handles empty dependencies list."""
        script = """\
# /// brainset-pipeline
# dependencies = []
# ///
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {"dependencies": []}

    def test_metadata_in_middle_of_file(self):
        """Finds metadata block even if not at start of file."""
        script = """\
#!/usr/bin/env python
# Some comment

# /// brainset-pipeline
# python-version = "3.9"
# ///

import sys
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {"python-version": "3.9"}

    def test_ignores_other_metadata_types(self):
        """Ignores non-brainset-pipeline metadata blocks (e.g., PEP 723 'script')."""
        script = """\
# /// script
# requires-python = ">=3.9"
# dependencies = ["requests"]
# ///

import requests
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result is None

    def test_multiple_blocks_raises_error(self):
        """Raises ValueError if there are multiple brainset-pipeline blocks.

        PEP723 Says: When there are multiple comment blocks of the same TYPE defined,
        tools MUST produce an error.
        """
        script = """\
# /// brainset-pipeline
# python-version = "3.10"
# ///

# some code

# /// brainset-pipeline
# python-version = "3.11"
# ///
"""
        filepath = _write_temp_script(script)
        with pytest.raises(ValueError, match="Multiple brainset-pipeline blocks found"):
            _read_inline_metadata(filepath)

    def test_unsupported_key_raises_error(self):
        """Raises ValueError when unsupported keys are present."""
        script = """\
# /// brainset-pipeline
# python-version = "3.10"
# unsupported-key = "value"
# ///
"""
        filepath = _write_temp_script(script)
        with pytest.raises(ValueError, match="Unsupported key"):
            _read_inline_metadata(filepath)

    def test_multiple_unsupported_keys_listed_in_error(self):
        """Error message lists all unsupported keys."""
        script = """\
# /// brainset-pipeline
# python-version = "3.10"
# foo = "bar"
# baz = "qux"
# ///
"""
        filepath = _write_temp_script(script)
        with pytest.raises(ValueError, match=r"(foo|baz).*(foo|baz)"):
            _read_inline_metadata(filepath)

    def test_multiline_dependencies(self):
        """Parses multiline TOML array syntax."""
        script = """\
# /// brainset-pipeline
# dependencies = [
#     "numpy>=1.20",
#     "pandas",
#     "scipy>=1.10.0",
# ]
# ///
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {"dependencies": ["numpy>=1.20", "pandas", "scipy>=1.10.0"]}

    def test_coexists_with_pep723_script_block(self):
        r"""Can coexist with a separate PEP 723 script block.

        We want to stay semi-complaint with PEP 723, which requires that tools
        that don't use a particular metadata block, should just ignore it.
        """
        script = """\
# /// script
# requires-python = ">=3.9"
# ///

# /// brainset-pipeline
# python-version = "3.10"
# dependencies = ["nlb-tools"]
# ///

import nlb_tools
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result == {
            "python-version": "3.10",
            "dependencies": ["nlb-tools"],
        }

    def test_empty_file(self):
        """Returns None for empty file."""
        script = ""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result is None

    def test_only_closing_marker(self):
        """Returns None when only closing marker exists (no opening)."""
        script = """\
# some comment
# ///
"""
        filepath = _write_temp_script(script)
        result = _read_inline_metadata(filepath)
        assert result is None
