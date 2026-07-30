#!/usr/bin/env python3
"""
Statystyki zbioru: długości nagrań, długości transkrypcji, duplikaty.

Raport służy do sprawdzenia, czy zbiór po całym czyszczeniu nadaje się do
treningu. Istotne są trzy rzeczy: czy nagrania mieszczą się w progach przyjętych
w konfiguracji, czy transkrypcje nie przekraczają limitu tokenizera oraz czy
w zbiorze nie powtarzają się te same zdania, bo powtórzenia zawyżają wagę
pojedynczych fraz w treningu.

Wymaga ffprobe dostępnego w PATH.
"""

import argparse
import math
import statistics
import subprocess
from collections import Counter, defaultdict
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


def print_num_stats(name: str, values):
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        print(f"\n{name}: brak danych")
        return

    print(f"\n{name}")
    print("   Liczba: ", len(values))
    print(f"   Min: {values[0]:.3f}   Mediana: {statistics.median(values):.3f}"
          f"   Srednia: {statistics.mean(values):.3f}   Max: {values[-1]:.3f}")


def print_histogram(values, bin_width: float, max_bins: int, title: str, unit: str = ""):
    values = [v for v in values if v is not None]
    if not values:
        print(f"\n{title}: brak danych")
        return

    bins = Counter(int(v // bin_width) for v in values)
    keys = sorted(bins)

    print(f"\n{title}")
    for k in keys[:max_bins]:
        print(f"{k * bin_width:8.2f} do {(k + 1) * bin_width:8.2f}{unit} : {bins[k]}")

    if len(keys) > max_bins:
        print(f"... jeszcze {len(keys) - max_bins} przedzialow")


def read_metadata(meta_path: Path, sep: str):
    rows = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(sep)
        rows.append((
            parts[0].strip() if parts else "",
            parts[1] if len(parts) > 1 else "",
            parts[2] if len(parts) > 2 else "",
        ))
    return rows


def find_duplicates(id_to_value: dict):
    value_to_ids = defaultdict(list)
    for clip_id, value in id_to_value.items():
        value_to_ids[value].append(clip_id)
    return {v: ids for v, ids in value_to_ids.items() if v != "" and len(ids) > 1}


def report_duplicates(label: str, dups: dict, examples: int):
    rows = sum(len(ids) for ids in dups.values())
    print(f"{label}: grup={len(dups)}, wierszy w grupach={rows}")

    if not (examples and dups):
        return

    print(f"\nPrzykladowe powtorzenia ({label}):")
    for value, ids in sorted(dups.items(), key=lambda x: len(x[1]), reverse=True)[:examples]:
        preview = value.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"   x{len(ids)} ids={ids[:10]} tekst='{preview}'")


def parse_args():
    ap = argparse.ArgumentParser(description="Statystyki zbioru w formacie LJSpeech")
    ap.add_argument("--wavs-dir", required=True, help="Katalog z plikami .wav")
    ap.add_argument("--metadata", help="Sciezka do metadata.csv, pominiecie wylacza analize tekstow")
    ap.add_argument("--sep", default="|", help="Separator kolumn w metadata.csv")
    ap.add_argument("--bin", type=float, default=0.5, help="Szerokosc przedzialu histogramu dlugosci nagran w sekundach")
    ap.add_argument("--char-bin", type=int, default=10, help="Szerokosc przedzialu histogramu dlugosci tekstu w znakach")
    ap.add_argument("--max-bins", type=int, default=40, help="Ile przedzialow wypisac")
    ap.add_argument("--dup-examples", type=int, default=5, help="Ile przykladow powtorzen wypisac, 0 wylacza")
    ap.add_argument("--max-norm-chars", type=int, default=220, help="Prog dlugosci trzeciej kolumny do zaraportowania")
    return ap.parse_args()


def main():
    args = parse_args()

    wavs = sorted(Path(args.wavs_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"Brak plikow .wav w {args.wavs_dir}")

    durations = []
    for wav in wavs:
        try:
            durations.append(wav_duration_sec(wav))
        except Exception as e:
            print("Blad odczytu", wav.name, e)

    durations = sorted(d for d in durations if d and d > 0)
    if not durations:
        raise SystemExit("Nie udalo sie odczytac dlugosci zadnego pliku.")

    total = sum(durations)
    print("Plikow:        ", len(durations))
    print(f"Laczny czas:    {total:.2f} s ({total / 60:.2f} min)")
    print_num_stats("Dlugosc nagran w sekundach", durations)
    print_histogram(durations, args.bin, args.max_bins, "Histogram dlugosci nagran", "s")

    if not args.metadata:
        return

    meta_path = Path(args.metadata)
    if not meta_path.is_file():
        raise SystemExit(f"Brak pliku metadanych: {meta_path}")

    rows = read_metadata(meta_path, args.sep)
    print(f"\nWierszy w metadata: {len(rows)}")

    id_to_text = {clip_id: text for clip_id, text, _ in rows}
    id_to_norm = {clip_id: norm for clip_id, _, norm in rows}

    for label, mapping in (("tekst", id_to_text), ("tekst znormalizowany", id_to_norm)):
        lengths = [float(len(v)) for v in mapping.values()]
        print_num_stats(f"Dlugosc kolumny '{label}' w znakach", lengths)
        print_histogram(lengths, float(args.char_bin), args.max_bins,
                        f"Histogram dlugosci kolumny '{label}'", " znakow")

    print("\nPowtorzenia (dokladna zgodnosc tekstu):")
    report_duplicates("tekst", find_duplicates(id_to_text), args.dup_examples)
    report_duplicates("tekst znormalizowany", find_duplicates(id_to_norm), args.dup_examples)

    long_rows = [row for row in rows if len(row[2]) > args.max_norm_chars]
    print(f"\nWierszy z trzecia kolumna dluzsza niz {args.max_norm_chars} znakow: {len(long_rows)}")
    for clip_id, _, norm in long_rows:
        preview = norm.replace("\n", " ").strip()
        if len(preview) > 150:
            preview = preview[:150] + "..."
        print(f"   id={clip_id}, dlugosc={len(norm)}, tekst='{preview}'")


if __name__ == "__main__":
    main()
