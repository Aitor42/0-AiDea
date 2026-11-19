"""Compatibility import for ``FINAL.permutator``."""

import sys

from FINAL import permutator as parameter_variants

sys.modules[__name__] = parameter_variants
