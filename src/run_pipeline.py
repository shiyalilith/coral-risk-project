#!/usr/bin/env python3
"""Backward-compatible wrapper for the training entry point."""

from __future__ import annotations

try:
    from .train import main
except ImportError:
    from train import main


if __name__ == "__main__":
    raise SystemExit(main())
