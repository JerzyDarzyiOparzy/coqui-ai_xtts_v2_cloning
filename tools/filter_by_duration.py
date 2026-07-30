#!/usr/bin/env python3
"""
Filtrowanie zbioru po długości nagrań.

Trainer odrzuca pliki dłuższe niż max_wav_length już w trakcie wczytywania,
a nagrania krótsze niż kilka sekund nie nadają się na próbkę warunkującą barwę
głosu. Odsianie ich wcześniej pozwala poznać rzeczywisty rozmiar zbioru przed
uruchomieniem treningu.

Tryb delete czyści zbiór w miejscu, tryb new zostawia oryginał nietknięty
i buduje obok nowy katalog. Metadane są w obu przypadkach dopasowywane do
pozostałych nagrań.

Wymaga ffprobe dostępnego w PATH.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def wav_duration_sec(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())


def parse_args():
    ap = argparse.ArgumentParser(description="Filtrowanie zbioru LJSpeech po dlugosci nagran")
    ap.add_argument("--dataset-dir", required=True, help="Katalog z metadata.csv i wavs/")
    ap.add_argument("--mode", choices=["delete", "new"], default="new",
                    help="delete: czysci zbior w miejscu, new: tworzy nowy zbior w --out-dir")
    ap.add_argument("--out-dir", help="Katalog nowego zbioru, wymagany dla --mode new")
    ap.add_argument("--min-s", type=float, default=2.0, help="Dolny prog dlugosci w sekundach")
    ap.add_argument("--max-s", type=float, default=12.0, help="Gorny prog dlugosci w sekundach")
    ap.add_argument("--sep", default="|", help="Separator kolumn w metadata.csv")
    ap.add_argument("--dry-run", action="store_true", help="Tylko raport, bez zmian na dysku")
    ap.add_argument("--backup", action="store_true",
                    help="Dla --mode delete: kopia metadata.csv i przeniesienie usuwanych nagran do deleted_wavs/")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.mode == "new" and not args.out_dir:
        raise SystemExit("Dla --mode new wymagany jest --out-dir")

    dataset = Path(args.dataset_dir)
    wavs_dir = dataset / "wavs"
    meta_path = dataset / "metadata.csv"

    if not wavs_dir.is_dir():
        raise SystemExit(f"Brak katalogu: {wavs_dir}")
    if not meta_path.is_file():
        raise SystemExit(f"Brak pliku: {meta_path}")

    meta_lines = [ln for ln in meta_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    deleted_dir = dataset / "deleted_wavs"
    if args.mode == "delete" and args.backup and not args.dry_run:
        shutil.copy2(meta_path, dataset / "metadata.csv.bak")
        deleted_dir.mkdir(parents=True, exist_ok=True)

    out_wavs_dir = None
    out_meta_path = None
    if args.mode == "new":
        out_dir = Path(args.out_dir)
        out_wavs_dir = out_dir / "wavs"
        out_meta_path = out_dir / "metadata.csv"
        if not args.dry_run:
            out_wavs_dir.mkdir(parents=True, exist_ok=True)

    kept_lines = []
    removed = []
    missing = []
    copied = 0

    for line in meta_lines:
        clip_id = line.split(args.sep)[0].strip()
        if not clip_id:
            continue

        wav_path = wavs_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            missing.append(clip_id)
            continue

        try:
            duration = wav_duration_sec(wav_path)
        except Exception:
            removed.append((clip_id, "ffprobe_error"))
            continue

        if duration < args.min_s:
            removed.append((clip_id, "za_krotkie"))
            continue
        if duration > args.max_s:
            removed.append((clip_id, "za_dlugie"))
            continue

        kept_lines.append(line)

        if args.mode == "new" and not args.dry_run:
            shutil.copy2(wav_path, out_wavs_dir / f"{clip_id}.wav")
            copied += 1

    print("Tryb:                ", args.mode)
    print("Zakres:              ", f"od {args.min_s} do {args.max_s} s")
    print("Wierszy w metadata:  ", len(meta_lines))
    print("Wpisow bez nagrania: ", len(missing))
    print("Zachowanych:         ", len(kept_lines))
    print("Odrzuconych:         ", len(removed))

    for reason in ("za_krotkie", "za_dlugie", "ffprobe_error"):
        print(f"   {reason}:", sum(1 for _, r in removed if r == reason))

    if args.dry_run:
        print("\nTryb --dry-run, nic nie zmieniono.")
        return

    if args.mode == "delete":
        for clip_id, _ in removed:
            wav_path = wavs_dir / f"{clip_id}.wav"
            if not wav_path.exists():
                continue
            if args.backup:
                wav_path.replace(deleted_dir / wav_path.name)
            else:
                wav_path.unlink(missing_ok=True)

        meta_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        print("\nZaktualizowano:", meta_path)
        if args.backup:
            print("Kopia metadata:", dataset / "metadata.csv.bak")
            print("Usuniete nagrania:", deleted_dir)
    else:
        out_meta_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        print("\nSkopiowanych nagran:", copied)
        print("Zapisano metadata:  ", out_meta_path)


if __name__ == "__main__":
    main()
