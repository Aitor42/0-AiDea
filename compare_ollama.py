"""Compatibility import for ``FINAL.compare_ollama``."""

import sys

from FINAL import compare_ollama

sys.modules[__name__] = compare_ollama
