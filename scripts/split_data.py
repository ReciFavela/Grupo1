"""
Divide data/PET e data/NOT_PET em train/val/test estratificado.

Uso (na raiz do projeto):
    python scripts/split_data.py
    python scripts/split_data.py --copy      # copia em vez de mover
    python scripts/split_data.py --dry-run   # só mostra contagens
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config

SPLITS = ("train", "val", "test")


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def count_split(split_dir: Path) -> dict[str, int]:
    counts = {}
    for classe in config.CLASSES:
        counts[classe] = len(list_images(split_dir / classe))
    counts["total"] = sum(counts.values())
    return counts


def split_already_done() -> bool:
    return (
        config.TRAIN_DIR.exists()
        and count_split(config.TRAIN_DIR)["total"] > 0
    )


def legacy_sources_exist() -> bool:
    return all(
        (config.DATA_DIR / classe).exists()
        for classe in config.CLASSES
    )


def stratified_split(files: list[Path]):
    train_files, temp_files = train_test_split(
        files,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    relative_test_size = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=relative_test_size,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    return train_files, val_files, test_files


def ensure_dirs():
    for split in SPLITS:
        for classe in config.CLASSES:
            (config.DATA_DIR / split / classe).mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, copy: bool, dry_run: bool):
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def run_split(copy: bool = False, dry_run: bool = False):
    if split_already_done():
        print("Split já existe. Resumo atual:")
        for split in SPLITS:
            counts = count_split(config.DATA_DIR / split)
            print(f"  {split}: {counts}")
        print("\nPara refazer, remova data/train, data/val e data/test primeiro.")
        return

    if not legacy_sources_exist():
        print("Erro: esperado data/PET e data/NOT_PET com imagens.")
        print("Baixe o dataset e coloque nessa estrutura antes de rodar o split.")
        sys.exit(1)

    ensure_dirs()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "ratios": {
            "train": config.TRAIN_RATIO,
            "val": config.VAL_RATIO,
            "test": config.TEST_RATIO,
        },
        "classes": {},
        "mode": "copy" if copy else "move",
    }

    total_moved = 0

    for classe in config.CLASSES:
        source_dir = config.DATA_DIR / classe
        files = list_images(source_dir)

        if not files:
            print(f"Aviso: nenhuma imagem em {source_dir}")
            continue

        train_files, val_files, test_files = stratified_split(files)

        manifest["classes"][classe] = {
            "source_total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        for split_name, split_files in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            dest_dir = config.DATA_DIR / split_name / classe
            for src in split_files:
                dst = dest_dir / src.name
                if dst.exists():
                    print(f"Pulando (já existe): {dst}")
                    continue
                transfer_file(src, dst, copy=copy, dry_run=dry_run)
                total_moved += 1

        print(
            f"{classe}: {len(files)} -> "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    if dry_run:
        print("\nDry-run concluído. Nenhum arquivo foi alterado.")
        return

    if not copy:
        for classe in config.CLASSES:
            legacy_dir = config.DATA_DIR / classe
            if legacy_dir.exists() and not list_images(legacy_dir):
                shutil.rmtree(legacy_dir)

    manifest["total_transferred"] = total_moved
    config.SPLIT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nSplit concluído. Resumo final:")
    for split in SPLITS:
        counts = count_split(config.DATA_DIR / split)
        print(f"  {split}: {counts}")

    print(f"\nManifesto salvo em: {config.SPLIT_MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Divide o dataset em train/val/test estratificado."
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copia arquivos em vez de mover (mantém data/PET e data/NOT_PET).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra contagens sem mover/copiar arquivos.",
    )
    args = parser.parse_args()

    run_split(copy=args.copy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
