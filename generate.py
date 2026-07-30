#!/usr/bin/env python3
"""
Generowanie mowy dostrojonym modelem XTTS v2.

Katalog modelu musi zawierać trzy pliki obok siebie: model.pth, config.json
oraz vocab.json. Nazwa model.pth jest wymagana przez loader XTTS, checkpoint
zapisany przez trainer trzeba pod nią podmienić.

Nazwa pliku wyjściowego składa się z użytych parametrów, dzięki czemu kolejne
odsłuchy nie nadpisują się nawzajem i da się je porównywać.
"""

import argparse
from pathlib import Path

import torch
from TTS.api import TTS

DEFAULT_TEXT = "Zażółć gęślą jaźń, a potem sprawdź, jak brzmi wytrenowany model."


def parse_args():
    parser = argparse.ArgumentParser(description="Synteza mowy dostrojonym modelem XTTS v2")
    parser.add_argument("--model-dir", default="model", help="Katalog z model.pth, config.json i vocab.json")
    parser.add_argument("--speaker-wav", required=True, help="Nagranie referencyjne barwy glosu (.wav)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Tekst do syntezy")
    parser.add_argument("--language", default="pl", help="Kod jezyka")
    parser.add_argument("--output", default=None, help="Plik wyjsciowy (domyslnie nazwa z parametrow)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def resolve_model_dir(model_dir: Path):
    checkpoint = model_dir / "model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Brak {checkpoint}. Wagi nie sa czescia repozytorium, skopiuj wybrany "
            "checkpoint z katalogu treningu i zmien jego nazwe na model.pth."
        )

    for name in ("config.json", "vocab.json"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"Brak {model_dir / name}")

    return checkpoint


def build_output_name(args):
    return (
        f"temp-{args.temperature}_topk-{args.top_k}_topp-{args.top_p}"
        f"_reppen-{args.repetition_penalty}_speed-{args.speed}.wav"
    )


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    resolve_model_dir(model_dir)

    speaker_wav = Path(args.speaker_wav)
    if not speaker_wav.exists():
        raise FileNotFoundError(f"Brak nagrania referencyjnego: {speaker_wav}")

    output_file = Path(args.output) if args.output else Path(build_output_name(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {model_dir.resolve()} ({device})")

    tts = TTS(
        model_path=str(model_dir),
        config_path=str(model_dir / "config.json"),
        progress_bar=True,
    ).to(device)

    tts.tts_to_file(
        text=args.text,
        speaker_wav=str(speaker_wav),
        language=args.language,
        file_path=str(output_file),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        speed=args.speed,
    )

    print(f"Zapisano: {output_file.resolve()}")


if __name__ == "__main__":
    main()
