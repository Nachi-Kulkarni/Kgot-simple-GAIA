"""winston_config.py
Compatibility wrapper for Python code that expects a module named
`config.logging.winston_config`.

Historically, the project used a JavaScript‐based Winston configuration
(`winston_config.js`).  Some Python modules were written to import
`get_logger` from this path, even though the actual implementation now
lives in `logger.py`.

To avoid a large-scale refactor (and to maintain backward-compatibility
with existing code and documentation) we simply re-export the
`get_logger` helper from `logger.py`.
"""

from __future__ import annotations

# Re-export the canonical implementation so callers can do:
#   from config.logging.winston_config import get_logger
# without breaking.
from .logger import get_logger, CustomLogger  # noqa: F401

__all__ = [
    "get_logger",
    "CustomLogger",
] 