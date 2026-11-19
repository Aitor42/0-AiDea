"""Entrada de compatibilidad para la matriz de experimentos local."""

import sys

import run_experiment_matrix

sys.modules[__name__] = run_experiment_matrix
