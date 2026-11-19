"""Comprueba los datos, el checkpoint y el CSV de MiniLM."""

import argparse
import csv
import hashlib
from pathlib import Path
from shutil import copyfile


TEST_ROWS = 5195
DATA_SHA256 = {
    "train.csv": "3f1222fb390532a5ac1703f40bbf59cc65e2eb1a3a19bafd5d5a844c548e6182",
    "test.csv": "4bab157073c61c875cc8ff3a78608ab13731a55a0236de8d6ad0c4a39a8121a5",
    "sample_submission.csv": "1320cfc2e8348495c34436f99a46034f9bc4e9121142945ed62349eb315e4cd8",
}
MODEL_SHA256 = {
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "config.json": "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e",
    "config_sentence_transformers.json": "b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8",
    "modules.json": "8f4b264b80206c830bebbdcae377e137925650a433b689343a63bdc9b3145460",
    "pytorch_model.bin": "16cc9e54df6e083272378abec2d75dc34d7a48b5276db3ccc050d18de672ac59",
    "sentence_bert_config.json": "70f4448f31320443fe3557cacea5abf2dcc4915dda8c80646bec9f3bb0aa5a1f",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
    "unigram.json": "71b44701d7efd054205115acfa6ef126c5d2f84bd3affe0c59e48163674d19a6"
}
ORIGINAL_TOKENIZER = "cac868512d555e9ac37a33c28300d36931ae4b5ddff43f1ca9bdacfe6fcd9dc8"
FIXED_TOKENIZER = "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b"


def verify_official_data(data_dir: Path) -> None:
    for name, expected in DATA_SHA256.items():
        raw = (data_dir / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{name} no coincide con los datos oficiales usados en la prueba.")


def prepare_checkpoint(source: Path, destination: Path) -> None:
    """Prepara una copia comprobada sin tocar el checkpoint de entrada."""
    raw = (source / "tokenizer_config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() == ORIGINAL_TOKENIZER:
        end = raw.rstrip().rfind(b"}")
        end_value = len(raw[:end].rstrip())
        raw = raw[:end_value] + b', "tokenizer_class": "PreTrainedTokenizerFast"' + raw[end_value:]
    if hashlib.sha256(raw).hexdigest() != FIXED_TOKENIZER:
        raise ValueError("tokenizer_config.json no coincide con el parche validado.")

    for name, expected in MODEL_SHA256.items():
        original = source / name
        with original.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"El archivo del checkpoint no coincide: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(original.resolve())
        except OSError:
            copyfile(original, target)
    (destination / "tokenizer_config.json").write_bytes(raw)


def validate_submission(submission: Path, sample: Path) -> None:
    """Valida el CSV persistido contra el orden de la plantilla oficial."""
    with sample.open(encoding="utf-8-sig", newline="") as stream:
        ids = [row["id"] for row in csv.DictReader(stream)]
    if len(ids) != TEST_ROWS or len(set(ids)) != TEST_ROWS or not all(ids):
        raise ValueError("La plantilla debe contener 5.195 IDs únicos y no vacíos.")

    with submission.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        if next(rows, None) != ["id", "prediction"]:
            raise ValueError("El CSV debe tener exactamente las columnas id,prediction.")
        for index, (row, expected_id) in enumerate(zip(rows, ids, strict=True), start=2):
            if len(row) != 2 or row[0] != expected_id or row[1] not in {"0", "1", "2"}:
                raise ValueError(f"ID, orden o predicción inválidos en la línea {index}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificar submission.csv frente a la plantilla de Kaggle.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("sample", type=Path)
    args = parser.parse_args()
    validate_submission(args.submission, args.sample)
    print(f"CSV válido: {TEST_ROWS} filas, IDs en orden y predicciones 0, 1 o 2.")
