#!/usr/bin/env python3
"""
Walidacja pliku metadata.csv w formacie LJSpeech (id|tekst|tekst_znormalizowany).

Uruchamiane przed treningiem, bo pojedyncza wadliwa linia potrafi przerwac
wczytywanie calego zbioru dopiero po kilku minutach pracy.
"""

import argparse
import csv
import sys
from pathlib import Path

MAX_REPORTED = 10


def validate(file_path: Path):
    errors = []
    warnings = []
    total = 0
    valid = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, row in enumerate(csv.reader(f, delimiter="|"), 1):
            total += 1

            if len(row) != 3:
                errors.append(f"Linia {line_num}: {len(row)} kolumn zamiast 3")
                continue

            file_id, text, normalized = row

            if not file_id.strip():
                errors.append(f"Linia {line_num}: puste ID")
                continue

            if not text.strip() or not normalized.strip():
                errors.append(f"Linia {line_num}: pusty tekst")
                continue

            # Rozjazd miedzy kolumnami nie blokuje treningu, ale zwykle oznacza
            # blad w skrypcie przygotowujacym dane.
            if text.strip() != normalized.strip():
                warnings.append(f"Linia {line_num}: kolumny 2 i 3 sie roznia")

            valid += 1

    return total, valid, errors, warnings


def report(section, items):
    if not items:
        return
    print(f"\n{section}:")
    for item in items[:MAX_REPORTED]:
        print(f"   {item}")
    if len(items) > MAX_REPORTED:
        print(f"   ... oraz {len(items) - MAX_REPORTED} wiecej")


def main():
    parser = argparse.ArgumentParser(description="Walidacja metadata.csv dla XTTS")
    parser.add_argument("path", nargs="?", default="metadata.csv", help="Sciezka do metadata.csv")
    args = parser.parse_args()

    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Plik {file_path} nie istnieje")
        sys.exit(1)

    total, valid, errors, warnings = validate(file_path)

    print("Statystyki")
    print(f"   Wszystkich linii: {total}")
    print(f"   Poprawnych:       {valid}")
    print(f"   Bledow:           {len(errors)}")
    print(f"   Ostrzezen:        {len(warnings)}")

    report("Bledy", errors)
    report("Ostrzezenia", warnings)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
