"""Entry point for ``python -m localsec``."""

from __future__ import annotations

import sys

from localsec.cli import main

if __name__ == "__main__":
    sys.exit(main())
