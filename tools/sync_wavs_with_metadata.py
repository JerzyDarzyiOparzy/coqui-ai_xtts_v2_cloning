#!/usr/bin/env python3
"""
Usuwanie nagrań osieroconych, czyli takich, do których nie ma wpisu w metadata.csv.

Po ręcznym czyszczeniu transkrypcji zbiór rozjeżdża się w obie strony: zostają
pliki bez opisu i wpisy bez pliku. Trainer przewraca się dopiero na tych drugich,
pierwsze po prostu zajmują miejsce, dlatego skrypt raportuje obie różnice,
a kasuje wyłącznie nadmiarowe nagrania.

Domyślnie usuwa. Przed pierwszym uruchomieniem warto sprawdzić raport przez --dry-run.
"""

import argparse
from pathlib import Path


def load_ids(meta_path: Path, sep: str) -> set:
    ids = set()
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(line.split(sep, 1)[0].strip())
    return ids


def load_wav_stems(wavs_dir: Path) -> set:
    return {p.stem for p in wavs_dir.glob("*.wav")}


def print_diff(meta_ids: set, wav_ids: set, title: str):
    only_in_meta = sorted(meta_ids - wav_ids)
    only_in_wavs = sorted(wav_ids - meta_ids)

    print(f"\n== {title} ==")
    print("Wpisow w metadata:      ", len(meta_ids))
    print("Nagran w katalogu:      ", len(wav_ids))
    print("Wpisy bez nagrania:     ", len(only_in_meta))
    print("Nagrania bez wpisu:     ", len(only_in_wavs))

    if only_in_meta:
        print("Przyklady wpisow bez nagrania:", only_in_meta[:10])
    if only_in_wavs:
        print("Przyklady nagran bez wpisu:  ", only_in_wavs[:10])


def parse_args():
    ap = argparse.ArgumentParser(description="Synchronizacja katalogu wavs z metadata.csv")
    ap.add_argument("--dataset-dir", required=True, help="Katalog datasetu")
    ap.add_argument("--meta", default="metadata.csv", help="Nazwa pliku metadanych w katalogu datasetu")
    ap.add_argument("--wavs-dir", default="wavs", help="Nazwa katalogu z nagraniami wzgledem katalogu datasetu")
    ap.add_argument("--sep", default="|", help="Separator kolumn w metadata.csv")
    ap.add_argument("--dry-run", action="store_true", help="Tylko raport, bez usuwania")
    return ap.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta_path = dataset_dir / args.meta
    wavs_dir = dataset_dir / args.wavs_dir

    if not meta_path.is_file():
        raise SystemExit(f"Brak pliku metadanych: {meta_path.resolve()}")
    if not wavs_dir.is_dir():
        raise SystemExit(f"Brak katalogu z nagraniami: {wavs_dir.resolve()}")

    meta_ids = load_ids(meta_path, args.sep)
    print_diff(meta_ids, load_wav_stems(wavs_dir), "PRZED")

    to_delete = sorted(p for p in wavs_dir.glob("*.wav") if p.stem not in meta_ids)
    print("\nDo usuniecia:", len(to_delete))

    if args.dry_run:
        for p in to_delete[:50]:
            print("   ", p.name)
        if len(to_delete) > 50:
            print(f"    ... oraz {len(to_delete) - 50} wiecej")
        print("\nTryb --dry-run, nic nie usunieto.")
        return

    for p in to_delete:
        p.unlink(missing_ok=True)

    print_diff(meta_ids, load_wav_stems(wavs_dir), "PO")


if __name__ == "__main__":
    main()
