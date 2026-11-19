"""Compatibility import for ``FINAL.translator``."""

import sys

from FINAL import translator as translation

sys.modules[__name__] = translation
