"""Test package marker for `core.tests`.

This file makes the `core.tests` directory a proper Python package so
that relative imports (e.g., ``from ..unified_system_controller``)
inside test modules resolve correctly during pytest collection.
""" 