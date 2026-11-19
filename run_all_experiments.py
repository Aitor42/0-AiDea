"""Ejecuta los bancos de experimentos de cada modelo, experimento a uno.

Recorre todos los clasificadores del catálogo (excepto Random Forest, que
conserva su banco ``rf_experiments.py``) y ejecuta cada experimento de su
registro con el mismo protocolo: solo el split de entrenamiento, CV de 5 folds
con ``seed=42`` y holdout intacto.

Propiedades del driver:

- **Reanudable:** lee el JSONL de cada modelo y se salta los pares
  ``(modelo, experimento)`` que ya tienen fila, de modo que se puede cortar y
  relanzar sin repetir trabajo.
- **Aislado por experimento:** un fallo o un tiempo de ejecución excesivo se
  registra y se continúa con el siguiente; no se pierde el resto del banco.
- **Ordenado de barato a caro** para que los resultados aparezcan pronto.

Uso (desde la raíz del proyecto):

    python run_all_experiments.py
    python run_all_experiments.py --models=knn,decision_tree
    python run_all_experiments.py --timeout=3600
"""

from __future__ import annotations

import importlib
import json
import signal
import sys
import time
import traceback

import model_experiments as me
from training import load_train_data, make_split

# Orden de ejecución: primero los modelos baratos y al final los boosting y el
# MLP, que son los que dominan el tiempo total.
MODELS = (
    "naive_bayes",
    "complement_nb",
    "bernoulli_nb",
    "decision_tree",
    "lightgbm",
    "knn",
    "xgboost",
    "mlp",
    "catboost",
)

# Tope por experimento (5 folds). Protege el barrido frente a una
# configuración patológica sin tocar las que terminan normalmente.
DEFAULT_TIMEOUT = 1800


class _ExperimentTimeout(Exception):
    """El experimento superó el tope de tiempo configurado."""


def _on_alarm(signum, frame):  # noqa: ARG001 - firma exigida por signal
    raise _ExperimentTimeout()


def executed(model: str) -> set[str]:
    """Nombres de experimento que ya tienen fila en el JSONL del modelo."""
    path = me.results_path(model)
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(json.loads(line)["name"])
    return names


def parse_args(argv):
    """Interpreta ``--models`` y ``--timeout``; ignora el resto."""
    models = MODELS
    timeout = DEFAULT_TIMEOUT
    for arg in argv:
        if arg.startswith("--models="):
            models = tuple(arg.split("=", 1)[1].split(","))
        elif arg.startswith("--timeout="):
            timeout = int(arg.split("=", 1)[1])
    unknown = [name for name in models if name not in MODELS]
    if unknown:
        raise SystemExit(f"modelo desconocido: {unknown[0]}. Válidos: {list(MODELS)}")
    return models, timeout


def main(argv) -> int:
    models, timeout = parse_args(list(argv))

    df = load_train_data(me.PACKAGE / "data/train.csv")
    train, _ = make_split(df, val_size=0.2, seed=42)
    del df
    print(f"partición de selección: {len(train)} pares", flush=True)

    failures: list[str] = []
    skipped: list[str] = []
    total_started = time.perf_counter()

    for model in models:
        experiments = importlib.import_module(f"{model}_experiments").EXPERIMENTS
        already = executed(model)
        pending = [name for name in experiments if name not in already]
        print(
            f"\n### {model}: {len(pending)} pendientes de {len(experiments)} "
            f"(ya hechos: {len(already)})",
            flush=True,
        )

        for name in pending:
            started = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] -> {model}:{name}", flush=True)
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(timeout)
            try:
                me.run(name, model, train, experiments[name])
            except _ExperimentTimeout:
                elapsed = time.perf_counter() - started
                print(f"TIMEOUT {model}:{name} (> {timeout}s)", flush=True)
                failures.append(f"{model}:{name} (timeout)")
                skipped.append(f"{model}:{name}")
            except Exception:  # noqa: BLE001 - el driver no debe abortar
                elapsed = time.perf_counter() - started
                print(f"FALLO {model}:{name} tras {elapsed:.0f}s", flush=True)
                traceback.print_exc()
                failures.append(f"{model}:{name}")
            else:
                elapsed = time.perf_counter() - started
                print(f"[{time.strftime('%H:%M:%S')}] ok {elapsed:.0f}s", flush=True)
            finally:
                signal.alarm(0)

    total = time.perf_counter() - total_started
    print(f"\nbarrido terminado en {total / 60:.1f} min", flush=True)
    print(f"fallos: {failures if failures else 'ninguno'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
