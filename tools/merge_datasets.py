#!/usr/bin/env python3
"""
Łączenie kilku zbiorów w jeden.

Nagrania pochodzące z różnych materiałów źródłowych trafiają do osobnych
katalogów, a dopiero przed treningiem są scalane. Identyfikatory z różnych
źródeł potrafią się powtarzać, dlatego każdy zbiór dostaje własny prefiks.

Kolejność wierszy jest losowana ze stałym ziarnem. Bez tego trainer dostawał
materiał posortowany źródłami, przez co walidacja wycinana z końca zbioru
pochodziła w całości z jednego nagrania.
"""

import argparse
import random
import shutil
from pathlib import Path


def read_metadata(path: Path, sep: str):
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split(sep)
        if len(parts) < 3:
            raise SystemExit(f"Mniej niz 3 kolumny w {path}, linia {i}: {line!r}")
        if not parts[0].strip():
            raise SystemExit(f"Puste id w {path}, linia {i}")

        rows.append((parts[0].strip(), parts[1], parts[2]))
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description="Laczenie zbiorow w formacie LJSpeech")
    ap.add_argument("--src", nargs="+", required=True, help="Katalogi zrodlowe, kazdy z metadata.csv i wavs/")
    ap.add_argument("--out", required=True, help="Katalog wynikowy")
    ap.add_argument("--meta-name", default="metadata.csv")
    ap.add_argument("--wavs-name", default="wavs")
    ap.add_argument("--sep", default="|")
    ap.add_argument("--seed", type=int, default=2137, help="Ziarno losowania kolejnosci wierszy")
    ap.add_argument("--dry-run", action="store_true", help="Tylko raport, bez kopiowania")
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_wavs = out_dir / args.wavs_name
    out_meta = out_dir / args.meta_name

    all_rows = []
    used_ids = set()
    missing_wavs = 0

    for idx, src in enumerate(Path(s) for s in args.src):
        meta_path = src / args.meta_name
        wavs_dir = src / args.wavs_name
        prefix = f"ds{idx + 1}_"

        if not meta_path.is_file():
            raise SystemExit(f"Brak pliku: {meta_path}")
        if not wavs_dir.is_dir():
            raise SystemExit(f"Brak katalogu: {wavs_dir}")

        for clip_id, text, norm in read_metadata(meta_path, args.sep):
            src_wav = wavs_dir / f"{clip_id}.wav"
            if not src_wav.exists():
                missing_wavs += 1
                continue

            new_id = prefix + clip_id
            if new_id in used_ids:
                raise SystemExit(f"Kolizja id mimo prefiksu: {new_id}")
            used_ids.add(new_id)

            all_rows.append((new_id, text, norm, src_wav))

    print("Wierszy z istniejacym nagraniem:", len(all_rows))
    print("Wpisow bez nagrania:            ", missing_wavs)

    random.Random(args.seed).shuffle(all_rows)

    if args.dry_run:
        print("\nTryb --dry-run, nic nie zapisano.")
        return

    out_wavs.mkdir(parents=True, exist_ok=True)
    for new_id, _, _, src_wav in all_rows:
        shutil.copy2(src_wav, out_wavs / f"{new_id}.wav")

    with out_meta.open("w", encoding="utf-8", newline="") as f:
        for new_id, text, norm, _ in all_rows:
            f.write(f"{new_id}{args.sep}{text}{args.sep}{norm}\n")

    print("\nZapisano metadata:", out_meta)
    print("Katalog nagran:   ", out_wavs)


if __name__ == "__main__":
    main()
