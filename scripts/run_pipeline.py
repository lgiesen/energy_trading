#!/usr/bin/env python3
"""Deprecated compatibility wrapper.

Use `scripts/run_full_pipeline.py` as canonical pipeline entrypoint.
"""
from __future__ import annotations

import logging

from scripts.run_full_pipeline import main

LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.warning("`scripts/run_pipeline.py` is deprecated. Forwarding to `scripts/run_full_pipeline.py`.")
    main()
