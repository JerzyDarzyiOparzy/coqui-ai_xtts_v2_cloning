#!/usr/bin/env python3
"""
Przycinanie ciszy na końcu nagrań.

Surowe nagrania cięte automatycznie mają zwykle ogon ciszy o zmiennej długości.
Model uczy się takiego ogona razem z mową i zaczyna go odtwarzać w syntezie,
dlatego przed treningiem cisza jest skracana do stałej wartości.

Skrypt szuka ostatniej ciszy w końcowym fragmencie pliku, tnie nagranie tuż za
końcem mowy, dokłada ustaloną ilość ciszy i wygasza końcówkę, żeby cięcie nie
dawało trzasku.

Wymaga ffmpeg i ffprobe dostępnych w PATH.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def run_capture(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def get_duration(wav: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav),
    ]
    return float(run_capture(cmd).stdout.strip())


def find_silences(wav: Path, start: float, length: float, sil_db: int, min_sil: float):
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(wav),
        "-af", f"silencedetect=noise={sil_db}dB:d={min_sil}",
        "-f", "null", "-",
    ]
    log = run_capture(cmd).stderr

    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)", log)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)", log)]
    durs = [float(m.group(1)) for m in re.finditer(r"silence_duration:\s*([0-9]+(?:\.[0-9]+)?)", log)]

    # Znaczniki czasu są liczone od początku analizowanego fragmentu, nie pliku.
    return [(start + starts[i], start + ends[i], durs[i]) for i in range(min(len(starts), len(ends), len(durs)))]


def cut_wav(in_wav: Path, out_wav: Path, new_end: float, sample_rate: int, fade_ms: int):
    new_end = max(0.05, new_end)
    fade_len = fade_ms / 1000.0
    fade_start = max(0.0, new_end - fade_len)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-to", f"{new_end:.3f}",
        "-af", f"afade=t=out:st={fade_start:.3f}:d={fade_len:.3f}",
        "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    run_capture(cmd)


def parse_args():
    ap = argparse.ArgumentParser(description="Przycinanie ciszy na koncu nagran")
    ap.add_argument("--wavs-dir", required=True, help="Katalog z plikami wejsciowymi")
    ap.add_argument("--out-dir", required=True, help="Katalog wynikowy")
    ap.add_argument("--sr", type=int, default=22050, help="Czestotliwosc probkowania na wyjsciu")
    ap.add_argument("--sil-db", type=int, default=-30, help="Prog ciszy w dB")
    ap.add_argument("--min-sil", type=float, default=0.40, help="Minimalna dlugosc ciszy do wykrycia")
    ap.add_argument("--lookback", type=float, default=3.0, help="Dlugosc analizowanej koncowki nagrania")
    ap.add_argument("--max-tail", type=float, default=0.25, help="Maksymalna resztka po ciszy, powyzej ktorej plik zostaje bez zmian")
    ap.add_argument("--keep", type=float, default=0.02, help="Margines przed poczatkiem ciszy")
    ap.add_argument("--tail-sil", type=float, default=0.45, help="Ile ciszy zostawic na koncu")
    ap.add_argument("--fade-ms", type=int, default=50, help="Dlugosc wygaszenia koncowki")
    return ap.parse_args()


def main():
    args = parse_args()

    wavs_dir = Path(args.wavs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wavs_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"Brak plikow .wav w {wavs_dir}")

    changed = 0
    for wav in wavs:
        duration = get_duration(wav)
        window_start = max(0.0, duration - args.lookback)
        window_len = min(args.lookback, duration)

        silences = find_silences(wav, window_start, window_len, args.sil_db, args.min_sil)

        # Brak wykrytej ciszy albo zbyt długa resztka po niej oznacza, że nagranie
        # kończy się mową i nie ma czego przycinać.
        if not silences:
            shutil.copy2(wav, out_dir / wav.name)
            continue

        sil_start, sil_end, _ = silences[-1]
        if duration - sil_end > args.max_tail:
            shutil.copy2(wav, out_dir / wav.name)
            continue

        new_end = min(duration, (sil_start - args.keep) + args.tail_sil)

        tmp = out_dir / (wav.stem + ".tmp.wav")
        cut_wav(wav, tmp, new_end=new_end, sample_rate=args.sr, fade_ms=args.fade_ms)
        tmp.replace(out_dir / wav.name)
        changed += 1

    print(f"Plikow:      {len(wavs)}")
    print(f"Przycietych: {changed}")
    print(f"Wynik:       {out_dir}")


if __name__ == "__main__":
    main()
