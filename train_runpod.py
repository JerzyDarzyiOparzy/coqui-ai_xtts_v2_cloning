#!/usr/bin/env python3
"""
Fine-tuning modelu XTTS v2 na własnym zbiorze nagrań, przygotowany pod RunPod.

Oczekiwana struktura katalogu z danymi:
    <dataset>/metadata.csv   format LJSpeech: id|tekst|tekst_znormalizowany
    <dataset>/wavs/*.wav
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import wget

try:
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer
    from TTS.tts.models.xtts import XttsAudioConfig
    from trainer import Trainer as GenericTrainer
    from trainer import TrainerArgs
except ImportError as e:
    print(f"Blad importu: {e}")
    print("Uruchom: pip install coqui-tts coqui-tts-trainer torch torchaudio wget coqpit")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

XTTS_BASE_URL = "https://huggingface.co/coqui/XTTS-v2/resolve/main/"
XTTS_FILES = ["dvae.pth", "mel_stats.pth", "vocab.json", "model.pth", "config.json"]


class RunPodVoiceCloner:
    def __init__(
        self,
        dataset_path=None,
        output_path=None,
        batch_size=16,
        grad_accumulation=16,
        language="pl",
        epochs=70,
        restore_path=None,
    ):
        self.workspace = Path("/workspace")
        self.restore_path = restore_path

        self.setup_runpod_environment()
        self.dataset_path = self.detect_dataset_path(dataset_path)

        if output_path is None:
            self.output_path = self.workspace / "voice_cloning_output"
        else:
            self.output_path = Path(output_path)
            if not self.output_path.is_absolute():
                self.output_path = self.workspace / output_path

        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.language = language
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_paths()

    def setup_runpod_environment(self):
        if not self.workspace.exists():
            logger.warning("Nie znaleziono /workspace, uzywam biezacego katalogu")
            self.workspace = Path.cwd()

        for name in ("voice_cloning_models", "voice_cloning_output", "voice_cloning_backup"):
            (self.workspace / name).mkdir(exist_ok=True)

    def detect_dataset_path(self, provided_path=None):
        if provided_path:
            candidate = Path(provided_path)
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            if self.is_valid_dataset_structure(candidate):
                return candidate

        candidates = [
            self.workspace / "audio",
            self.workspace / "dataset",
            self.workspace / "data" / "audio",
            Path.cwd() / "audio",
            self.workspace / "voice_data",
        ]

        for path in candidates:
            if self.is_valid_dataset_structure(path):
                logger.info(f"Wykryto dataset: {path}")
                return path

        raise FileNotFoundError(
            "Nie znaleziono poprawnej struktury datasetu (wymagane: metadata.csv oraz folder wavs/)"
        )

    def is_valid_dataset_structure(self, path: Path):
        if not path.exists():
            return False
        if not (path / "metadata.csv").exists():
            return False
        wavs = path / "wavs"
        return wavs.exists() and any(wavs.glob("*.wav"))

    def setup_paths(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = self.workspace / "voice_cloning_models" / "XTTS_v2.0_original_model_files"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

    def download_models(self):
        for filename in XTTS_FILES:
            file_path = self.checkpoints_path / filename
            if file_path.exists():
                continue

            logger.info(f"Pobieranie: {filename}")
            try:
                wget.download(XTTS_BASE_URL + filename, str(file_path))
                print()
            except Exception as e:
                logger.error(f"Blad pobierania {filename}: {e}")
                raise

        logger.info("Pliki modelu bazowego gotowe")

    def create_training_config(self, metadata_file: str):
        dataset_config = BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="custom_voice",
            path=str(self.dataset_path),
            meta_file_train=metadata_file,
            language=self.language,
        )

        audio_config = XttsAudioConfig(
            sample_rate=22050,
            dvae_sample_rate=22050,
            output_sample_rate=24000,
        )

        model_args = GPTArgs(
            max_conditioning_length=220500,  # 10 s przy 22050 Hz
            min_conditioning_length=66150,   # 3 s przy 22050 Hz
            debug_loading_failures=False,
            max_wav_length=264600,           # 12 s przy 22050 Hz
            max_text_length=200,
            mel_norm_file=str(self.checkpoints_path / "mel_stats.pth"),
            dvae_checkpoint=str(self.checkpoints_path / "dvae.pth"),
            xtts_checkpoint=str(self.checkpoints_path / "model.pth"),
            tokenizer_file=str(self.checkpoints_path / "vocab.json"),
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )

        wav_files = sorted((self.dataset_path / "wavs").glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError("Brak plikow .wav w katalogu wavs/")
        speaker_reference = [str(wav_files[0])]

        test_sentences = [
            {
                "text": "Żwirek kręci z Muchomorkiem.",
                "speaker_wav": speaker_reference,
                "language": self.language,
            },
            {
                "text": "Przyznaje się do związku z siedmioma krasnoludkami, ale podkreśla, że nie jest łatwa, Śnieżka!",
                "speaker_wav": speaker_reference,
                "language": self.language,
            },
        ]

        config = XttsConfig(
            run_name="GPT_XTTS_v2_RunPod",
            epochs=self.epochs,
            print_step=50,
            batch_size=self.batch_size,
            # Ewaluacja potrafi zjeść kilkanaście GB VRAM przy większych wartościach.
            eval_batch_size=4,
            num_loader_workers=4,
            eval_split_max_size=256,
            print_eval=True,
            output_path=str(self.output_path),
            model_args=model_args,
            audio=audio_config,
            optimizer="AdamW",
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 0.01},
            lr=1e-05,
            # Restarty kosinusowe powodują skoki loss widoczne w TensorBoard,
            # wykres wygląda źle mimo poprawnego przebiegu treningu.
            lr_scheduler="CosineAnnealingWarmRestarts",
            lr_scheduler_params={"T_0": 5, "T_mult": 2, "eta_min": 1e-6},
            # Gęste zapisy, bo best_model wybierany jest po wartości loss,
            # a najlepiej brzmiące checkpointy bywają późniejsze.
            save_step=100,
            save_n_checkpoints=5,
            save_checkpoints=True,
            test_sentences=test_sentences,
            datasets=[dataset_config],
        )

        # Wyłączone, inaczej trainer tworzy zduplikowane grupy parametrów optymalizatora.
        config.optimizer_wd_only_on_weights = False

        return config, [dataset_config]

    def train(self):
        metadata_path = self.dataset_path / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Brak metadata.csv w {self.dataset_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if "|" not in first_line:
            raise ValueError("metadata.csv nie ma formatu LJSpeech, brak separatora '|'")

        self.download_models()
        config, dataset_configs = self.create_training_config(str(metadata_path))

        train_samples, eval_samples = load_tts_samples(
            dataset_configs,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=0.01,
        )

        logger.info(f"Probki treningowe: {len(train_samples)}")
        logger.info(f"Probki walidacyjne: {len(eval_samples)}")
        logger.info(f"Efektywny batch size: {self.batch_size * self.grad_accumulation}")

        model = GPTTrainer.init_from_config(config)
        logger.info(f"Ladowanie wag bazowych z: {self.checkpoints_path}")
        model.xtts.load_checkpoint(config, checkpoint_dir=str(self.checkpoints_path), use_deepspeed=False)

        trainer_args = TrainerArgs(
            restore_path=self.restore_path,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=self.grad_accumulation,
        )

        trainer = GenericTrainer(
            trainer_args,
            config,
            output_path=str(self.output_path),
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )

        trainer.fit()

        logger.info(f"Trening zakonczony, wyniki w: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning XTTS v2 na RunPod")
    parser.add_argument("--dataset", type=str, help="Sciezka do katalogu z metadata.csv i wavs/")
    parser.add_argument("--output", type=str, help="Katalog wyjsciowy (domyslnie /workspace/voice_cloning_output)")
    parser.add_argument("--language", type=str, default="pl", help="Kod jezyka")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad-accumulation", type=int, default=16, help="Liczba krokow akumulacji gradientu")
    parser.add_argument("--epochs", type=int, default=70, help="Liczba epok")
    parser.add_argument("--restore-path", type=str, default=None, help="Checkpoint .pth do wznowienia treningu")
    args = parser.parse_args()

    try:
        cloner = RunPodVoiceCloner(
            dataset_path=args.dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            grad_accumulation=args.grad_accumulation,
            language=args.language,
            restore_path=args.restore_path,
            epochs=args.epochs,
        )
        cloner.train()
    except Exception as e:
        logger.error(f"Trening przerwany: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
