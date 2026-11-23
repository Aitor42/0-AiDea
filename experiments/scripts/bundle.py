"""Empaqueta el código reutilizable para un notebook ejecutado en Kaggle."""

from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
MINILM_PROTOCOL = Path("experiments/results/validation/minilm_grouped/protocol.json")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def make_code_bundle(destination: Path, *, include_protocol: bool = False) -> dict[str, str]:
    sources = sorted((ROOT / "src/nli_project").rglob("*.py"))
    if include_protocol:
        sources.extend((ROOT / MINILM_PROTOCOL, ROOT / MINILM_PROTOCOL.with_name("grouped_seeds.csv")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        hashes = {}
        for path in sources:
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, arcname=name)
            hashes[name] = digest(path)
    return hashes


if __name__ == "__main__":
    parser = ArgumentParser(description="Preparar el ZIP para el notebook MiniLM.")
    parser.add_argument("destination", type=Path, nargs="?",
                        default=ROOT / "experiments/.local/minilm-source.zip")
    args = parser.parse_args()
    make_code_bundle(args.destination, include_protocol=True)
    print(args.destination)
