# Katalog modelu

Skrypt [generate.py](../generate.py) wczytuje model z tego katalogu. Potrzebne są trzy pliki obok siebie:

| Plik | Skąd pochodzi | W repozytorium |
| --- | --- | --- |
| `config.json` | zapisany przez trainer na początku przebiegu, zawiera pełną konfigurację treningu | tak |
| `vocab.json` | tokenizer XTTS v2, pobierany razem z modelem bazowym | tak |
| `model.pth` | checkpoint po fine-tuningu, około 5 GB | nie |

## Wagi modelu

Plik `model.pth` nie wchodzi w skład repozytorium ze względu na rozmiar. Aby uruchomić generowanie, wybierz jeden z checkpointów zapisanych przez trainer w katalogu wyników i skopiuj go tutaj pod nazwą `model.pth`:

```bash
cp /workspace/voice_cloning_output/GPT_XTTS_v2_RunPod-<data>/checkpoint_3541.pth model/model.pth
```

Nazwa `model.pth` jest wymagana, loader XTTS szuka dokładnie takiego pliku w katalogu przekazanym w `--model-dir`.

## config.json

To zapis konfiguracji z faktycznego przebiegu treningu opisanego w [README](../README.md), a nie plik przykładowy. Wartości `epochs`, `batch_size`, `lr`, `lr_scheduler`, `save_step` i `eval_batch_size` odpowiadają domyślnym ustawieniom w [train_runpod.py](../train_runpod.py).

Pola `temperature`, `top_k`, `top_p` i `repetition_penalty` w dolnej części pliku to wartości domyślne modelu bazowego, używane tylko wtedy, gdy nie podasz odpowiadających im argumentów w `generate.py`.

## vocab.json

Plik tokenizera skopiowany z modelu bazowego bez zmian. Nie należy go edytować. Identyfikatory tokenów są zapisane w wagach modelu, więc dopisanie lub usunięcie choćby jednej pozycji sprawia, że model zaczyna generować szum.
