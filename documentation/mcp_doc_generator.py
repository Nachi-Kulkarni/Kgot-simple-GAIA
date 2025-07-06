"""mcp_doc_generator.py
Automatic documentation generator for MCP (Modular Cognitive Process) Python modules.

This script introspects Python source files that define MCPs, extracts relevant metadata (docstrings, argument schemas, etc.), and automatically generates Markdown documentation files.  
Future extensions will add LLM-powered best-practice analysis and multi-format export (e.g., via MkDocs or Sphinx).

Usage (CLI):
    python documentation/mcp_doc_generator.py --source-dir alita-kgot-enhanced --output-dir docs/mcp

TODO:
    • Integrate LLM calls for best-practice generation
    • Add unit tests for metadata extraction and markdown generation
"""

from __future__ import annotations

import argparse
import ast
import inspect
import logging
import os
from pathlib import Path
from textwrap import indent
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
# NOTE: The broader codebase uses Winston (JS) for logging. In this Python
# module we rely on the standard `logging` package, while maintaining log level
# parity with Winston (debug, info, warning, error, critical). If a unified
# cross-language logging backend is introduced in the future, adapt this section
# to forward Python logs to Winston via a bridge such as logstash or HTTP.
# -----------------------------------------------------------------------------

LOGGER_NAME = "mcp_doc_generator"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)  # Capture all levels; handlers decide filtering

# Console handler - INFO and above
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_formatter = logging.Formatter(
    "%(asctime)s • %(levelname)s • %(name)s • %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_console_handler.setFormatter(_console_formatter)
logger.addHandler(_console_handler)

# File handler - full DEBUG log
_log_dir = Path("logs/mcp_doc_generator")
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "combined.log")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_console_formatter)
logger.addHandler(_file_handler)

# -----------------------------------------------------------------------------
# Data Classes & Helper Structures
# -----------------------------------------------------------------------------

class MCPMetadata:
    """Container for extracted MCP metadata."""

    def __init__(self, name: str, description: str) -> None:
        self.name: str = name
        self.description: str = description
        self.args: List[Dict[str, Any]] = []  # Each item: {name, type, description}
        self.returns: Optional[str] = None
        self.dependencies: List[str] = []

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def add_arg(self, name: str, type_: str, description: str | None = None) -> None:
        self.args.append({"name": name, "type": type_, "description": description or ""})

    def set_returns(self, description: str) -> None:
        self.returns = description


# -----------------------------------------------------------------------------
# Core DocGenerator Class
# -----------------------------------------------------------------------------

class DocGenerator:
    """Generate Markdown documentation for MCP modules.

    This class can be instantiated programmatically *or* invoked via CLI entry
    point in `main()` below.
    """

    def __init__(self, source_dir: Path, output_dir: Path) -> None:
        self.source_dir: Path = source_dir
        self.output_dir: Path = output_dir
        logger.debug("Initialized DocGenerator with source_dir=%s, output_dir=%s", source_dir, output_dir)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(self) -> None:
        """Walk the `source_dir`, process each Python file, and generate docs."""
        python_files = list(self.source_dir.rglob("*.py"))
        logger.info("Discovered %d Python files under %s", len(python_files), self.source_dir)

        # Preload global dependency list from nearest requirements.txt
        global_dependencies = self._read_requirements(self.source_dir)
        if global_dependencies:
            logger.debug("Loaded %d dependencies from requirements.txt", len(global_dependencies))

        for file_path in python_files:
            try:
                metadata = self._extract_metadata(file_path)
                metadata.dependencies = global_dependencies  # Same deps for all MCPs in repo context
                self._write_markdown(metadata)
                logger.info("✅ Generated docs for %s", file_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("❌ Failed to generate docs for %s: %s", file_path, exc)

    # ---------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------

    def _extract_metadata(self, file_path: Path) -> MCPMetadata:
        """Parse the given Python file to extract MCP metadata.

        The strategy is intentionally lightweight:
            1. Use `ast` to obtain the module-level docstring (overall description).
            2. Identify classes that expose an `args_schema` attribute (common in BaseTool subclasses) and
               gather their arg definitions via `inspect`.
            3. Fallback to function-level parsing if no classes found.
        """
        logger.debug("Parsing file %s", file_path)
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        module_docstring = ast.get_docstring(tree) or ""  # May be empty
        module_name = file_path.stem
        metadata = MCPMetadata(name=module_name, description=module_docstring)

        # Inspect class definitions for args_schema
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                logger.debug("Found class %s in %s", class_name, file_path.name)

                # Extract class docstring
                class_doc = ast.get_docstring(node) or ""
                if class_doc and not metadata.description:
                    metadata.description = class_doc  # Prefer detailed class docstring

                # Detect args_schema within class body
                for body_item in node.body:
                    if (
                        isinstance(body_item, ast.Assign)
                        and len(body_item.targets) == 1
                        and getattr(body_item.targets[0], "id", None) == "args_schema"
                    ):
                        try:
                            # Use eval in restricted namespace to resolve schema reference
                            logger.debug("Attempting to evaluate args_schema in %s.%s", module_name, class_name)
                            namespace: Dict[str, Any] = {}
                            exec(source, namespace)  # pylint: disable=exec-used
                            cls_obj = namespace[class_name]
                            args_schema = getattr(cls_obj, "args_schema", None)
                            if args_schema is not None:
                                self._extract_args_from_schema(args_schema, metadata)
                        except Exception as exc:  # pylint: disable=broad-except
                            logger.warning("Could not extract args_schema for %s: %s", class_name, exc)

        return metadata

    # ------------------------------------------------------------------
    def _extract_args_from_schema(self, schema_cls: Any, metadata: MCPMetadata) -> None:
        """Populate `metadata.args` using a pydantic schema class."""
        logger.debug("Extracting args from schema %s", schema_cls)
        try:
            # Handle pydantic BaseModel
            if hasattr(schema_cls, "__fields__"):
                for arg_name, field in schema_cls.__fields__.items():
                    arg_type = str(field.type_)
                    arg_desc = field.field_info.description or ""
                    metadata.add_arg(arg_name, arg_type, arg_desc)
            else:
                # Fallback: inspect annotations
                annotations = getattr(schema_cls, "__annotations__", {})
                for arg_name, arg_type in annotations.items():
                    metadata.add_arg(arg_name, str(arg_type))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to extract args from schema %s: %s", schema_cls, exc)

    # ------------------------------------------------------------------
    def _write_markdown(self, metadata: MCPMetadata) -> None:
        """Render and save Markdown documentation for a single MCP."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{metadata.name}.md"
        logger.debug("Writing Markdown to %s", output_path)

        md_lines = [f"# {metadata.name}\n"]

        if metadata.description:
            md_lines.append(metadata.description + "\n")

        md_lines.append("## Parameters\n")
        if metadata.args:
            for arg in metadata.args:
                md_lines.append(f"* **{arg['name']}** (*{arg['type']}*) – {arg['description']}")
        else:
            md_lines.append("(None detected)\n")

        md_lines.append("\n## Returns\n")
        md_lines.append(metadata.returns or "(Not specified)\n")

        if metadata.dependencies:
            md_lines.append("\n## Dependencies\n")
            for dep in metadata.dependencies:
                md_lines.append(f"* {dep}")

        md_lines.append("\n## Example Usage\n")
        usage_block = (
            f"```python\n"
            f"from {metadata.name} import *  # Adjust import as necessary\n"
            f"# TODO: Provide concrete example\n"
            f"```\n"
        )
        md_lines.append(usage_block)

        md_lines.append("\n## Best Practices\n")
        md_lines.append(self._generate_best_practices(metadata))

        output_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ------------------------------------------------------------------
    def _generate_best_practices(self, metadata: MCPMetadata) -> str:
        """Stub for future LLM-powered analysis of best practices."""
        logger.debug("Generating best practices placeholder for %s", metadata.name)
        # Placeholder text until LLM integration is available.
        return (
            "* 🛈 *This section will be automatically populated by an LLM in a future "
            "release, providing guidelines and performance tips based on code and "
            "usage context.*"
        )

    # ------------------------------------------------------------------
    def _read_requirements(self, start_path: Path) -> List[str]:
        """Traverse upwards to locate a requirements.txt and return its contents.

        Args:
            start_path: Path from where the upward search begins.

        Returns:
            List of dependency strings (e.g., ["langchain>=0.1.0", "pydantic==1.10.0"])
        """
        logger.debug("Searching for requirements.txt starting at %s", start_path)
        current = start_path.resolve()
        for _ in range(5):  # Limit upward search depth to avoid traversing entire FS
            req_file = current / "requirements.txt"
            if req_file.exists():
                deps = [line.strip() for line in req_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
                return deps
            if current.parent == current:
                break  # Reached filesystem root
            current = current.parent
        logger.debug("No requirements.txt found near %s", start_path)
        return []


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    """CLI wrapper for DocGenerator."""

    parser = argparse.ArgumentParser(description="Automatic MCP documentation generator")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Root directory that contains MCP Python modules",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/mcp"),
        help="Directory where generated Markdown files will be stored",
    )

    args = parser.parse_args()
    logger.info("Starting documentation generation: source=%s output=%s", args.source_dir, args.output_dir)

    generator = DocGenerator(source_dir=args.source_dir, output_dir=args.output_dir)
    generator.run()
    logger.info("Documentation generation completed.")


# -----------------------------------------------------------------------------
# Module Guard
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main() 