# Klonowanie głosu w języku polskim (XTTS v2)

Fine-tuning modelu [XTTS v2](https://huggingface.co/coqui/XTTS-v2) na własnym zbiorze nagrań, przeprowadzony na wynajmowanym GPU w serwisie RunPod. Projekt poboczny, zrobiony po godzinach. Repozytorium zawiera cały kod oraz konfigurację z faktycznego przebiegu treningu: przygotowanie i walidację danych, dostrajanie modelu, a na końcu generowanie mowy gotowym modelem.

Projekt korzysta z aktywnie utrzymywanego forka [idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS), ponieważ oryginalne repozytorium Coqui nie jest już rozwijane.

Ten plik opisuje, co robi każdy element projektu. Jeżeli chcesz powtórzyć eksperyment od zera na własnych nagraniach, zacznij od [TUTORIAL.md](TUTORIAL.md), gdzie te same czynności ułożone są w kolejności wykonywania, razem z konfiguracją poda, przenoszeniem plików i listą typowych problemów.

## Cel

Sprawdzenie, na ile model XTTS v2 daje się dostroić do pojedynczego głosu w języku polskim przy ograniczonym budżecie obliczeniowym oraz jak dobór hiperparametrów, wybór checkpointu i wybór nagrania referencyjnego wpływają na jakość odsłuchową wyniku.

## Przebieg

```
1. przyciecie ciszy       tools/trim_silence.py
2. synchronizacja         tools/sync_wavs_with_metadata.py
3. scalenie zrodel        tools/merge_datasets.py
4. filtr dlugosci         tools/filter_by_duration.py
5. raport i walidacja     tools/dataset_stats.py, tools/validate_metadata.py
6. fine-tuning            train_runpod.py, 70 epok na GPU A40
7. wybor checkpointu      odsluch kilku ostatnich zapisow, nie tylko best_model
8. generowanie            generate.py, dobor parametrow i probki referencyjnej
```

## Struktura repozytorium

```
TUTORIAL.md                 przejscie krok po kroku od surowych nagran do gotowego modelu
train_runpod.py             fine-tuning XTTS v2: pobranie modelu bazowego, konfiguracja, trening
generate.py                 generowanie mowy dostrojonym modelem
tools/
  trim_silence.py           przyciecie ciszy na koncu nagran
  sync_wavs_with_metadata.py  usuniecie nagran bez wpisu w metadata.csv
  merge_datasets.py         scalenie kilku zbiorow z prefiksowaniem identyfikatorow
  filter_by_duration.py     odsianie nagran spoza przyjetego zakresu dlugosci
  dataset_stats.py          raport dlugosci, histogramy, wykrywanie powtorzen
  validate_metadata.py      walidacja formatu metadata.csv przed treningiem
model/
  config.json               konfiguracja z przeprowadzonego treningu
  vocab.json                tokenizer XTTS v2
  README.md                 jak podlozyc wagi model.pth
requirements.txt            zaleznosci Pythona
```

Wagi modelu (`model.pth`, około 5 GB) oraz zbiór treningowy nie wchodzą w skład repozytorium.

## Wymagania

* GPU z pamięcią co najmniej 48 GB VRAM przy ustawieniach domyślnych, trening prowadzono na NVIDIA A40
* Python 3.10 lub nowszy
* CUDA 12.4
* ffmpeg wraz z ffprobe w zmiennej PATH, wykorzystywane przez skrypty przygotowujące dane
* około 8 GB miejsca na dysku systemowym oraz 25 GB na wolumenie danych przy przechowywaniu dwóch checkpointów, każdy kolejny to około 5 GB

## Instalacja

Procedura odtwarzająca użyte środowisko, uruchamiana w konsoli pod RunPod:

```bash
cd /workspace
git clone https://github.com/idiap/coqui-ai-TTS
cd coqui-ai-TTS
pip install -e .
```

PyTorch zgodny z CUDA 12.4:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Pozostałe zależności:

```bash
pip install -r requirements.txt
```

Wersja `transformers` jest przypięta, ponieważ nowsze wydania zmieniają API generacji tekstu wykorzystywane przez warstwę GPT modelu XTTS.

Pliki z tego repozytorium skopiuj do katalogu `coqui-ai-TTS`, tak aby uruchamiać je obok zainstalowanej biblioteki.

## Dane

Zbiór treningowy nie wchodzi w skład repozytorium. Poniżej opis wymaganego formatu oraz charakterystyka użytego zbioru, co pozwala odtworzyć eksperyment na własnych nagraniach.

### Format

```
audio/
  metadata.csv
  wavs/
    nagranie_001.wav
    nagranie_002.wav
```

Każda linia `metadata.csv` ma trzy kolumny rozdzielone znakiem pionowej kreski:

```
nagranie_001|Treść wypowiedzi.|Treść wypowiedzi.
```

Pierwsza kolumna to nazwa pliku WAV bez rozszerzenia, druga to transkrypcja, trzecia to transkrypcja znormalizowana.

### Przygotowanie zbioru

Materiał wyjściowy to nagrania pocięte na fragmenty wraz z transkrypcjami, w kilku osobnych katalogach odpowiadających różnym źródłom. Transkrypcje powstały w Whisperze, a podział przebiegał po zdaniach. Fragmenty wykraczające poza przyjęty zakres długości zostały odrzucone w całości, bez łączenia zbyt krótkich i bez dzielenia zbyt długich. Kod z tego etapu nie zachował się i nie wchodzi w skład repozytorium, opis znajduje się w [TUTORIAL.md](TUTORIAL.md).

Katalog [tools/](tools/) zawiera skrypty, przez które ten materiał przechodzi po kolei. Wszystkie przyjmują `--dry-run` albo tryb zapisujący wynik obok oryginału, więc żaden krok nie jest nieodwracalny.

Pełne przejście krok po kroku, razem z konfiguracją poda i przenoszeniem plików, opisuje [TUTORIAL.md](TUTORIAL.md). Poniżej sam skrót.

Krok 1, przycięcie ciszy na końcu nagrań. Skrypt operuje wyłącznie na nagraniach, więc metadane trzeba przenieść obok:

```bash
python tools/trim_silence.py --wavs-dir surowe/wavs --out-dir przyciete/wavs
cp surowe/metadata.csv przyciete/metadata.csv
```

Krok 2, usunięcie nagrań, do których nie ma już wpisu w transkrypcjach:

```bash
python tools/sync_wavs_with_metadata.py --dataset-dir przyciete --dry-run
```

Krok 3, scalenie źródeł w jeden zbiór. Każde źródło dostaje prefiks identyfikatora, a kolejność wierszy jest losowana ze stałym ziarnem:

```bash
python tools/merge_datasets.py --src zrodlo1 zrodlo2 zrodlo3 --out scalone
```

Krok 4, odsianie nagrań spoza przyjętego zakresu długości:

```bash
python tools/filter_by_duration.py --dataset-dir scalone --mode new --out-dir audio --min-s 2 --max-s 12
```

Krok 5, raport końcowy i walidacja formatu:

```bash
python tools/dataset_stats.py --wavs-dir audio/wavs --metadata audio/metadata.csv
```

```bash
python tools/validate_metadata.py audio/metadata.csv
```

`dataset_stats.py` wypisuje histogramy długości nagrań i transkrypcji oraz wykrywa powtórzone zdania. `validate_metadata.py` sprawdza sam format pliku i kończy się kodem wyjścia różnym od zera, jeśli znajdzie błąd. Warto uruchomić go przed każdym treningiem, bo pojedyncza wadliwa linia potrafi przerwać wczytywanie zbioru dopiero po kilku minutach pracy.

### Charakterystyka użytego zbioru

| Parametr | Wartość |
| --- | --- |
| Liczba nagrań | 1173 |
| Łączny czas | 69,5 min (1,16 h) |
| Długość nagrania, zakres | od 2,004 s do 10,000 s |
| Długość nagrania, średnia | 3,56 s |
| Długość nagrania, mediana | 3,13 s |
| Format | WAV, 22050 Hz, mono, 16 bit |
| Długość transkrypcji | od 10 do 164 znaków, średnio 49 |
| Liczba słów w transkrypcji | średnio 7,1, maksymalnie 26 |

Rozkład długości nagrań:

| Przedział | Liczba nagrań | Udział |
| --- | --- | --- |
| 2 do 4 s | 881 | 75,1 % |
| 4 do 6 s | 206 | 17,6 % |
| 6 do 8 s | 53 | 4,5 % |
| 8 do 10 s | 21 | 1,8 % |
| 10 do 12 s | 12 | 1,0 % |

Przyjęty przedział to od 2 do 12 s, a fragmenty poza nim były odrzucane bez prób łączenia czy dzielenia. Taki zakres zalecają poradniki do XTTS v2 i pokrywa się on z ograniczeniem wynikającym z konfiguracji: `max_wav_length` ustawione jest na 264600 próbek, czyli 12 s przy 22050 Hz, a dłuższe pliki trainer odrzuca. Ograniczenie dolne wiąże się z `min_conditioning_length` równym 66150 próbek, czyli 3 s, poniżej którego nagranie nie nadaje się na próbkę warunkującą barwę głosu.

Faktyczny zakres w gotowym zbiorze jest węższy i wynosi od 2,004 s do 10,000 s. Dwanaście nagrań ma równo 10,000 s, a kolejne w kolejności nie przekracza 9,5 s. Skupienie na okrągłej wartości nie powstaje przy odrzucaniu, gdzie długości są przypadkowe, więc na etapie cięcia zadziałał dodatkowy limit czasu ustawiony na 10 s. Te dwanaście nagrań jest prawdopodobnie uciętych w trakcie wypowiedzi i ich transkrypcja opisuje więcej, niż słychać w pliku.

Transkrypcje mieszczą się w limicie `max_text_length` wynoszącym 200 znaków, najdłuższa ma 164 znaki.

## Trening

```bash
python train_runpod.py --dataset audio --epochs 70
```

Model bazowy XTTS v2 pobiera się automatycznie przy pierwszym uruchomieniu do katalogu `/workspace/voice_cloning_models`. Wyniki trafiają do `/workspace/voice_cloning_output`.

| Parametr | Domyślnie | Opis |
| --- | --- | --- |
| `--dataset` | wykrywany automatycznie | katalog z `metadata.csv` i `wavs/` |
| `--output` | `/workspace/voice_cloning_output` | katalog wyników |
| `--language` | `pl` | kod języka |
| `--batch-size` | `16` | rozmiar batcha |
| `--grad-accumulation` | `16` | liczba kroków akumulacji gradientu |
| `--epochs` | `70` | liczba epok |
| `--restore-path` | brak | checkpoint `.pth`, od którego wznawiany jest trening |

Wznowienie przerwanego treningu:

```bash
python train_runpod.py --restore-path /workspace/voice_cloning_output/GPT_XTTS_v2_RunPod-<data>/checkpoint_3541.pth
```

Pełna konfiguracja opisywanego tu przebiegu zapisana jest w [model/config.json](model/config.json).

### Monitorowanie

W terminalu na maszynie zdalnej:

```bash
tensorboard --logdir /workspace/voice_cloning_output --bind_all --port 6006
```

Tunel SSH z komputera lokalnego, z danymi połączenia z zakładki Connect w panelu RunPod, opcja SSH over exposed TCP:

```bash
ssh -L 6006:127.0.0.1:6006 root@<host> -p <port> -i ~/.ssh/id_ed25519
```

Panel dostępny jest pod adresem `http://127.0.0.1:6006`.

## Generowanie

Do katalogu [model/](model/) skopiuj wybrany checkpoint pod nazwą `model.pth`, obok znajdujących się już tam plików `config.json` i `vocab.json`. Szczegóły w [model/README.md](model/README.md).

```bash
python generate.py --model-dir model --speaker-wav wavs/nagranie_011.wav --text "Tekst do przeczytania przez model."
```

| Parametr | Domyślnie | Wpływ |
| --- | --- | --- |
| `--model-dir` | `model` | katalog z `model.pth`, `config.json` i `vocab.json` |
| `--speaker-wav` | wymagany | nagranie referencyjne barwy głosu |
| `--text` | zdanie testowe | tekst do syntezy |
| `--temperature` | `0.7` | niższa wartość daje spokojniejszą, bardziej powtarzalną wymowę |
| `--top-k` | `20` | odcina mało prawdopodobne tokeny audio |
| `--top-p` | `0.8` | ogranicza rozrzut generacji |
| `--repetition-penalty` | `2.0` | przeciwdziała zacinaniu się i powtarzaniu sylab |
| `--speed` | `1.0` | tempo mowy |

Nazwa pliku wyjściowego składa się z użytych parametrów, na przykład `temp-0.7_topk-20_topp-0.8_reppen-2.0_speed-1.0.wav`. Dzięki temu kolejne próby nie nadpisują się nawzajem i dają się porównywać odsłuchem.

## Wnioski z eksperymentów

### Dane

* Cisza na końcu nagrania jest częścią materiału uczącego. Fragmenty cięte automatycznie mają ogon ciszy o różnej długości, a model odtwarza go potem w syntezie jako niepotrzebną pauzę. Skrócenie ciszy do stałych 450 ms usunęło ten efekt.
* Kolejność wierszy w `metadata.csv` ma znaczenie, bo zbiór walidacyjny wycinany jest z końca. Przy materiale wczytywanym źródło po źródle cała walidacja pochodziła z jednego nagrania i jej wynik nie mówił nic o reszcie zbioru. Stąd losowanie kolejności ze stałym ziarnem przy scalaniu.
* Powtórzone zdania podbijają wagę pojedynczych fraz w treningu, dlatego raport ze statystyk sprawdza dokładne duplikaty w obu kolumnach tekstowych.
* Zbiór rozjeżdża się w obie strony po każdej ręcznej poprawce transkrypcji. Wpisy bez nagrania przewracają trainer, nagrania bez wpisu tylko zajmują miejsce, więc oba przypadki są raportowane osobno.
* Odrzucanie fragmentów spoza zakresu długości jest najprostszym podejściem i przy materiale liczonym w godzinach wystarcza. Kosztuje jednak część nagrań, których nie da się odzyskać inaczej niż przez sklejanie krótkich i dzielenie długich, a to wymaga osobnego narzędzia i sprawdzenia, czy tekst nadal odpowiada dźwiękowi.
* Warto obejrzeć rozkład długości pod kątem skupień na okrągłych wartościach. Przy odrzucaniu długości są przypadkowe, więc kilkanaście plików o identycznym czasie co do milisekundy oznacza twarde ucięcie na wcześniejszym etapie, a nie właściwość materiału. Takie nagrania kończą się w połowie wypowiedzi, choć ich transkrypcja opisuje całość.
* Nagrania krótsze niż 3 s nie nadają się na próbkę warunkującą barwę głosu, mimo że przechodzą trening bez błędu. Trzy czwarte zbioru mieści się w przedziale od 2 do 4 s, więc przy wyborze próbki referencyjnej pole manewru jest mniejsze, niż sugeruje wielkość zbioru.
* Niecała godzina i dwadzieścia minut nagrań okazała się wystarczająca do rozpoznawalnego odwzorowania barwy głosu przy dostrajaniu modelu wielojęzycznego. Nie jest to natomiast ilość pozwalająca na trening od zera.

### Trening

* Efektywny batch size, czyli iloczyn `batch_size` i `grad_accumulation`, opłaca się zwiększać przy treningu od zera. Przy fine-tuningu pojedynczego głosu bardzo duże wartości nie poprawiły wyniku. Ustawienie 16 na 16 mieściło się w 48 GB VRAM bez błędu braku pamięci.
* `eval_batch_size` trzeba trzymać nisko. Przy wartości 32 sama ewaluacja zajmowała około 10 GB VRAM i wywracała trening.
* Plik `best_model.pth` wybierany jest po wartości funkcji straty i nie zawsze brzmi najlepiej. W teście na 1800 krokach zawierał wyraźne artefakty, a lepiej wypadały późniejsze checkpointy. Stąd gęsty zapis co 100 kroków i przechowywanie pięciu ostatnich zapisów do odsłuchu.
* Harmonogram `CosineAnnealingWarmRestarts` powoduje regularne skoki na wykresie straty w TensorBoard. Wykres wygląda niepokojąco, ale trening przebiega poprawnie.
* Opcję `optimizer_wd_only_on_weights` trzeba pozostawić wyłączoną, w przeciwnym razie trainer tworzy zduplikowane grupy parametrów optymalizatora i przerywa pracę.
* Ocena wyniku wymaga odsłuchu. Wartość funkcji straty pozwala odrzucić przebiegi wyraźnie nieudane, ale nie porządkuje sensownie checkpointów zbliżonych do siebie.

### Generowanie

* Wybór nagrania referencyjnego wpływa na wynik silniej niż różnice w parametrach generacji. Próbki pochodzące z jednego materiału źródłowego dawały zauważalnie czystszy głos niż pozostałe, przy identycznych ustawieniach i tym samym modelu. Warto przesłuchać kilkanaście próbek i wybrać na stałe tę, która brzmi najlepiej.
* Przetestowano dwa zestawy parametrów: `temperature 0.6`, `top_k 35`, `repetition_penalty 4.0` oraz `temperature 0.7`, `top_k 20`, `repetition_penalty 2.0`. Drugi brzmi naturalniej, pierwszy stabilniej przy dłuższych zdaniach. Wartości domyślne w `generate.py` odpowiadają drugiemu zestawowi.
* Wysoki `repetition_penalty` skutecznie usuwa zacinanie się modelu, ale zawyżony spłaszcza intonację.
* Loader XTTS wymaga, aby plik z wagami nazywał się dokładnie `model.pth`. Checkpoint zapisany przez trainer pod nazwą `checkpoint_3541.pth` trzeba przemianować, inaczej biblioteka go nie znajdzie.

## Licencja

Kod w tym repozytorium udostępniony jest na licencji MIT.

Model bazowy XTTS v2 oraz pochodzący z niego plik `model/vocab.json` objęte są licencją [Coqui Public Model License](https://coqui.ai/cpml), która ogranicza wykorzystanie do zastosowań niekomercyjnych. Fine-tuning nie zmienia warunków tej licencji.
