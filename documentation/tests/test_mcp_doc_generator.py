"""Test suite for mcp_doc_generator.

These tests serve as a lightweight sanity check ensuring the DocGenerator's
core functionality works as expected. They are *not* exhaustive; contributors
should add more cases as MCP modules evolve.
"""

from pathlib import Path

import pytest

# Adjust import path for local testing without package installation
import sys, importlib

SCRIPT_PATH = Path(__file__).parent.parent / "mcp_doc_generator.py"
MODULE_NAME = "mcp_doc_generator"

spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)  # type: ignore
sys.modules[MODULE_NAME] = module  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore

DocGenerator = module.DocGenerator  # pylint: disable=invalid-name


def test_metadata_extraction(tmp_path: Path):
    """Verify that DocGenerator extracts module-level docstring."""
    sample_code = '"""Sample MCP description."""\nclass Dummy:\n    pass\n'
    src_file = tmp_path / "dummy.py"
    src_file.write_text(sample_code, encoding="utf-8")

    gen = DocGenerator(source_dir=tmp_path, output_dir=tmp_path)
    metadata = gen._extract_metadata(src_file)  # pylint: disable=protected-access

    assert metadata.name == "dummy"
    assert metadata.description == "Sample MCP description."


def test_markdown_generation(tmp_path: Path):
    """Ensure Markdown file is created for a simple module."""
    sample_code = '"""Another description."""\nclass Dummy:\n    pass\n'
    src_file = tmp_path / "dummy2.py"
    src_file.write_text(sample_code, encoding="utf-8")

    output_dir = tmp_path / "docs"
    gen = DocGenerator(source_dir=tmp_path, output_dir=output_dir)
    gen.run()

    md_file = output_dir / "dummy2.md"
    assert md_file.exists(), "Markdown file not generated."
    assert "Another description." in md_file.read_text(encoding="utf-8") 