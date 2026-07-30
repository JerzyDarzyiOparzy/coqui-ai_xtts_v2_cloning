# Odtworzenie eksperymentu krok po kroku

Przejście od surowych nagrań do gotowego modelu mówiącego wybranym głosem. Instrukcja zakłada, że pracujesz lokalnie na Windowsie lub Linuksie, a trening odpalasz na wynajętym GPU w serwisie RunPod. [README.md](README.md) opisuje, co robi każdy skrypt, tutaj jest sama kolejność działań.

Spis treści:

1. [Czego potrzebujesz](#1-czego-potrzebujesz)
2. [Materiał wejściowy](#2-materiał-wejściowy)
3. [Przygotowanie zbioru lokalnie](#3-przygotowanie-zbioru-lokalnie)
4. [Uruchomienie poda](#4-uruchomienie-poda)
5. [Wgranie danych i instalacja](#5-wgranie-danych-i-instalacja)
6. [Trening](#6-trening)
7. [Pobranie modelu](#7-pobranie-modelu)
8. [Generowanie](#8-generowanie)
9. [Typowe problemy](#9-typowe-problemy)

## 1. Czego potrzebujesz

Lokalnie:

* Python 3.10 lub nowszy
* ffmpeg razem z ffprobe w zmiennej PATH, sprawdź poleceniem `ffprobe -version`
* klucz SSH, na przykład `~/.ssh/id_ed25519`
* około 1 GB wolnego miejsca na zbiór i około 5 GB na pobrany checkpoint

Zdalnie:

* konto RunPod z doładowanym saldem
* GPU z co najmniej 48 GB VRAM przy ustawieniach domyślnych

Nagrania: około godziny czystej mowy jednego mówcy. Poniżej tej wartości barwa głosu bywa rozpoznawalna, ale wyraźnie spada stabilność wymowy.

## 2. Materiał wejściowy

Punktem wyjścia są długie nagrania, z których trzeba wydobyć krótkie fragmenty z dopasowanym tekstem.

W tym projekcie zrobił to Whisper. Cały materiał został przepuszczony przez transkrypcję, a wynik podzielony na zdania. Fragmenty wykraczające poza przedział od 2 do 12 sekund zostały odrzucone w całości, bez łączenia zbyt krótkich i bez dzielenia zbyt długich. Taki zakres zalecają poradniki do XTTS v2 i pokrywa się on z ograniczeniem `max_wav_length` w konfiguracji treningu. Przycinaniem ciszy i odsiewaniem fragmentów poza zakresem zajmowały się już skrypty z katalogu [tools/](tools/), opisane w kroku 3.

Odrzucanie jest najprostszym z możliwych podejść i kosztuje część materiału. Alternatywa, czyli sklejanie sąsiadujących krótkich fragmentów i dzielenie długich na pauzach, zachowuje więcej nagrań, ale wymaga osobnego narzędzia i kontroli, czy tekst po sklejeniu nadal odpowiada dźwiękowi. Przy materiale liczonym w godzinach prostsze podejście wystarczyło.

Kod użyty do samego cięcia i transkrypcji nie zachował się, dlatego nie ma go w repozytorium. Poniżej opis tego, co musi z tego etapu wyjść, oraz jeden ze sposobów, żeby to powtórzyć.

### Wymagany wynik

Na wejściu do kroku 3 potrzebujesz jednego lub kilku katalogów w takim układzie:

```
zrodlo1/
  metadata.csv
  wavs/
    klip_0001.wav
    klip_0002.wav
```

Wymagania wobec plików:

* WAV, mono, 22050 Hz, 16 bit
* jeden fragment to jedno zdanie lub jego naturalna część, nie dłużej niż kilkanaście sekund
* bez muzyki w tle, bez nakładających się głosów, bez pogłosu

Wymagania wobec `metadata.csv`:

* trzy kolumny rozdzielone znakiem `|`, czyli identyfikator, transkrypcja, transkrypcja znormalizowana
* identyfikator to nazwa pliku WAV bez rozszerzenia
* kodowanie UTF-8
* transkrypcja musi zgadzać się z nagraniem co do słowa, literówki i pominięte wyrazy przekładają się wprost na wymowę modelu

Przykład linii:

```
klip_0001|Wieniawa zdał relację z przygotowań.|Wieniawa zdał relację z przygotowań.
```

Kolumna trzecia służy do zapisania tekstu po normalizacji, czyli z liczbami i skrótami rozwiniętymi na słowa. Jeżeli nie prowadzisz osobnej normalizacji, powtórz w niej kolumnę drugą. Użyty tu zbiór miał obie kolumny identyczne.

### Jak to powtórzyć

Poniżej jeden ze sposobów, a nie odtworzenie oryginalnego kodu.

Transkrypcja z zapisem znaczników czasu:

```bash
pip install openai-whisper
whisper nagranie.mp3 --language pl --model large-v3 --output_format srt
```

Mniejsze modele też zadziałają, ale przy polskim wyraźnie rosną wtedy błędy w odmianie i interpunkcji, a interpunkcja decyduje o podziale na zdania.

Plik SRT daje segmenty z czasem początku i końca. Każdy segment tnij z oryginału do osobnego pliku, od razu w docelowym formacie:

```bash
ffmpeg -i nagranie.wav -ss 00:01:23.400 -to 00:01:29.100 -ac 1 -ar 22050 -acodec pcm_s16le wavs/klip_0001.wav
```

Segmenty Whispera nie pokrywają się jeden do jednego ze zdaniami, więc na tym etapie trzeba jeszcze przejrzeć transkrypcje i poprawić błędy. Whisper myli się przy nazwach własnych i gubi interpunkcję, a każda literówka przekłada się wprost na wymowę modelu. Jest to najbardziej pracochłonna część całego przygotowania danych i nie da się jej pominąć.

Fragmentami spoza zakresu długości nie musisz zajmować się ręcznie, odsieje je krok 4. Uważaj natomiast, żeby przy cięciu nie nałożyć własnego limitu czasu, na przykład flagą `-t` w ffmpeg. Nagranie ucięte w połowie słowa przejdzie przez filtr długości, bo mieści się w zakresie, ale jego transkrypcja opisuje już coś innego niż dźwięk.

Konwersja gotowego pliku do wymaganego formatu, gdyby materiał był w innym:

```bash
ffmpeg -i wejscie.wav -ac 1 -ar 22050 -acodec pcm_s16le wyjscie.wav
```

## 3. Przygotowanie zbioru lokalnie

Sklonuj repozytorium i wejdź do katalogu:

```bash
git clone https://github.com/<login>/coqui-ai_xtts_v2_cloning.git
cd coqui-ai_xtts_v2_cloning
```

### Krok 1: przycięcie ciszy

Fragmenty cięte automatycznie mają ogon ciszy o zmiennej długości. Model uczy się go razem z mową i odtwarza potem w syntezie jako niepotrzebną pauzę, dlatego cisza zostaje skrócona do stałej wartości.

```bash
python tools/trim_silence.py --wavs-dir zrodlo1/wavs --out-dir przyciete1/wavs
cp zrodlo1/metadata.csv przyciete1/metadata.csv
```

Skrypt operuje wyłącznie na nagraniach, kopia metadanych jest konieczna, żeby kolejne kroki widziały komplet. Powtórz dla każdego źródła.

### Krok 2: synchronizacja

Po ręcznych poprawkach transkrypcji zbiór rozjeżdża się w obie strony. Najpierw obejrzyj raport, dopiero potem usuwaj:

```bash
python tools/sync_wavs_with_metadata.py --dataset-dir przyciete1 --dry-run
python tools/sync_wavs_with_metadata.py --dataset-dir przyciete1
```

Skrypt kasuje wyłącznie nagrania bez wpisu. Wpisy bez nagrania tylko raportuje, bo ich usunięcie oznaczałoby utratę transkrypcji, i to Ty decydujesz, czy dopisać brakujący plik, czy skasować linię.

### Krok 3: scalenie źródeł

```bash
python tools/merge_datasets.py --src przyciete1 przyciete2 przyciete3 --out scalone
```

Każde źródło dostaje prefiks `ds1_`, `ds2_` i tak dalej, więc identyfikatory nie kolidują. Kolejność wierszy jest losowana ze stałym ziarnem, co jest istotne, bo zbiór walidacyjny wycinany jest z końca pliku. Bez losowania walidacja pochodzi w całości z jednego źródła i jej wynik nie mówi nic o reszcie zbioru.

Przy jednym źródle ten krok możesz pominąć.

### Krok 4: filtr długości

```bash
python tools/filter_by_duration.py --dataset-dir scalone --mode new --out-dir audio --min-s 2 --max-s 12
```

Górny próg wynika z `max_wav_length` w konfiguracji treningu, czyli 264600 próbek, co przy 22050 Hz daje 12 s. Dłuższe pliki trainer i tak odrzuca, lepiej wiedzieć o tym wcześniej. Dolny próg bierze się stąd, że bardzo krótkie fragmenty nie nadają się na próbkę warunkującą barwę głosu.

Tryb `new` zostawia katalog wejściowy nietknięty. Tryb `delete` czyści zbiór w miejscu i wtedy warto dodać `--backup`.

### Krok 5: raport i walidacja

```bash
python tools/dataset_stats.py --wavs-dir audio/wavs --metadata audio/metadata.csv
python tools/validate_metadata.py audio/metadata.csv
```

Na co patrzeć w raporcie:

* łączny czas, bo poniżej mniej więcej godziny wyniki wyraźnie się psują
* histogram długości, bo skupienie wszystkiego w jednym wąskim przedziale ogranicza wybór próbki referencyjnej
* liczba powtórzeń, bo te same zdania podbijają wagę pojedynczych fraz
* wiersze z bardzo długą trzecią kolumną, bo limit tokenizera to 200 znaków

`validate_metadata.py` kończy się kodem wyjścia różnym od zera, jeśli znajdzie błąd formatu. Uruchom go przed każdym treningiem, bo jedna wadliwa linia potrafi przerwać wczytywanie zbioru dopiero po kilku minutach pracy.

Po tym etapie masz katalog `audio/` gotowy do wysłania.

## 4. Uruchomienie poda

W panelu RunPod wybierz GPU z co najmniej 48 GB VRAM, na przykład A40. Ustaw wolumen na co najmniej 50 GB, bo mieści się w tym model bazowy, zbiór i kilka checkpointów po około 5 GB każdy. Wybierz szablon z PyTorch i CUDA 12.

Po starcie poda wejdź w zakładkę Connect i wybierz SSH over exposed TCP. Dostaniesz adres i port, których użyjesz niżej zamiast `<host>` i `<port>`.

Sprawdź połączenie:

```bash
ssh root@<host> -p <port> -i ~/.ssh/id_ed25519
```

## 5. Wgranie danych i instalacja

Wyślij zbiór z komputera lokalnego:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r audio root@<host>:/workspace/
```

Alternatywnie użyj `runpodctl send` lokalnie i `runpodctl receive` na podzie, co bywa szybsze przy dużych katalogach.

Dalej pracujesz już na podzie. Zainstaluj bibliotekę:

```bash
cd /workspace
git clone https://github.com/idiap/coqui-ai-TTS
cd coqui-ai-TTS
pip install -e .
```

Podmień PyTorcha na wersję zgodną z CUDA 12.4:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Wgraj kod z tego repozytorium do katalogu `coqui-ai-TTS` i doinstaluj resztę zależności:

```bash
cd /workspace/coqui-ai-TTS
git clone https://github.com/<login>/coqui-ai_xtts_v2_cloning.git repo
cp -r repo/train_runpod.py repo/generate.py repo/tools repo/model .
pip install -r repo/requirements.txt
```

Sprawdź, czy karta jest widoczna:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 6. Trening

```bash
cd /workspace/coqui-ai-TTS
python train_runpod.py --dataset /workspace/audio --epochs 70
```

Przy pierwszym uruchomieniu skrypt pobiera model bazowy XTTS v2 do `/workspace/voice_cloning_models`, czyli około 2 GB. Wyniki trafiają do `/workspace/voice_cloning_output/GPT_XTTS_v2_RunPod-<data>`.

Trening odpalaj w `tmux` albo `screen`, inaczej zerwane połączenie SSH ubije proces:

```bash
tmux new -s trening
```

Skala: przy 1173 próbkach i `batch_size` równym 16 wychodzi około 73 kroków na epokę, więc 70 epok to mniej więcej 5100 kroków. Checkpoint zapisywany jest co 100 kroków, przechowywanych jest pięć ostatnich.

### Monitorowanie

Na podzie:

```bash
tensorboard --logdir /workspace/voice_cloning_output --bind_all --port 6006
```

Lokalnie tunel:

```bash
ssh -L 6006:127.0.0.1:6006 root@<host> -p <port> -i ~/.ssh/id_ed25519
```

Panel otworzysz pod adresem `http://127.0.0.1:6006`.

Wykres straty będzie regularnie skakał w górę i to jest oczekiwane. Harmonogram `CosineAnnealingWarmRestarts` co jakiś czas podbija tempo uczenia, co widać jako pik. Poza tymi pikami trend powinien opadać.

Ważniejsza od wykresu jest zakładka z próbkami audio. To one mówią, czy model idzie w dobrą stronę.

### Wznowienie po przerwaniu

```bash
python train_runpod.py --restore-path /workspace/voice_cloning_output/GPT_XTTS_v2_RunPod-<data>/checkpoint_3541.pth
```

## 7. Pobranie modelu

Nie bierz automatycznie `best_model.pth`. Jest wybierany po wartości funkcji straty, a ta nie porządkuje sensownie checkpointów zbliżonych do siebie. W trakcie testów zdarzało się, że plik oznaczony jako najlepszy miał wyraźne artefakty, a lepiej brzmiały zapisy późniejsze.

Zobacz, co masz do wyboru:

```bash
ls -la /workspace/voice_cloning_output/GPT_XTTS_v2_RunPod-<data>/*.pth
```

Pobierz kilka ostatnich na dysk lokalny i porównaj odsłuchem:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 root@<host>:/workspace/voice_cloning_output/GPT_XTTS_v2_RunPod-<data>/checkpoint_3541.pth ./
```

Pobierz też kilka nagrań ze zbioru, przydadzą się jako próbki referencyjne:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 root@<host>:/workspace/audio/wavs/ds1_klip_0011.wav ./
```

## 8. Generowanie

Wybrany checkpoint umieść w katalogu `model/` pod nazwą `model.pth`. Nazwa jest wymagana, loader XTTS szuka dokładnie takiego pliku:

```bash
cp checkpoint_3541.pth model/model.pth
```

W katalogu `model/` leżą już `config.json` i `vocab.json` z repozytorium, więc komplet jest gotowy.

```bash
python generate.py --model-dir model --speaker-wav ds1_klip_0011.wav --text "Tekst do przeczytania przez model."
```

Plik wyjściowy dostanie nazwę złożoną z użytych parametrów, więc kolejne próby nie nadpisują się nawzajem.

### Dobór ustawień

Zacznij od próbki referencyjnej, bo to ona decyduje o wyniku najsilniej. Przesłuchaj kilkanaście nagrań ze zbioru, wygeneruj to samo zdanie z każdym z nich przy identycznych parametrach i wybierz jedno na stałe. Różnice bywają większe niż między skrajnymi ustawieniami generacji.

Dopiero potem ruszaj parametry:

```bash
python generate.py --model-dir model --speaker-wav proba.wav --temperature 0.6 --top-k 35 --repetition-penalty 4.0 --text "Zdanie testowe."
```

Punkt odniesienia to dwa przetestowane zestawy. Pierwszy, `temperature 0.6` z `top_k 35` i `repetition_penalty 4.0`, jest stabilniejszy przy długich zdaniach. Drugi, `temperature 0.7` z `top_k 20` i `repetition_penalty 2.0`, brzmi naturalniej i to on jest domyślny.

Wysoki `repetition_penalty` usuwa zacinanie się modelu, ale przesadzony spłaszcza intonację.

## 9. Typowe problemy

**`CUDA out of memory` w trakcie ewaluacji, sam trening szedł normalnie**

Ewaluacja potrafi zająć kilkanaście GB ponad to, co zajmuje trening. Zmniejsz `eval_batch_size` w `train_runpod.py`, wartość 4 mieści się bez problemu. Przy 32 trening się przewracał.

**`CUDA out of memory` od razu na starcie**

Zmniejsz `--batch-size` i podnieś `--grad-accumulation`, tak aby iloczyn został ten sam. Przy fine-tuningu jednego głosu bardzo duży efektywny batch i tak nie poprawia wyniku.

**`Nie znaleziono poprawnej struktury datasetu`**

Skrypt szuka katalogu zawierającego jednocześnie `metadata.csv` i podkatalog `wavs/` z co najmniej jednym plikiem. Sprawdź, czy metadane nie zostały przypadkiem w środku `wavs/` i czy podałeś `--dataset` ze ścieżką bezwzględną.

**`metadata.csv nie ma formatu LJSpeech, brak separatora '|'`**

Plik został zapisany z przecinkiem albo średnikiem zamiast pionowej kreski. Częsty efekt otwarcia i zapisania pliku w Excelu, który przy okazji potrafi zmienić kodowanie na inne niż UTF-8.

**Trainer przerywa pracę z błędem o zduplikowanych grupach parametrów**

`optimizer_wd_only_on_weights` musi zostać wyłączone. W `train_runpod.py` jest ustawione na `False` i nie należy tego zmieniać.

**`ffprobe` nie jest rozpoznawane jako polecenie**

Skrypty przygotowujące dane wywołują ffmpeg i ffprobe jako zewnętrzne programy. Zainstaluj ffmpeg i dopisz katalog `bin` do zmiennej PATH.

**Model generuje szum albo urywane sylaby**

Najpierw sprawdź, czy nie edytowałeś `vocab.json`. Identyfikatory tokenów są zapisane w wagach modelu i usunięcie choćby jednej pozycji rozsypuje generację. Jeżeli plik jest nietknięty, sprawdź inny checkpoint, bo pojedyncze zapisy potrafią być gorsze od sąsiednich.

**Biblioteka nie znajduje modelu mimo poprawnej ścieżki**

Plik z wagami musi nazywać się dokładnie `model.pth` i leżeć w katalogu podanym w `--model-dir`, razem z `config.json` i `vocab.json`.

**Błąd importu z `transformers` przy starcie treningu**

Wersja jest przypięta na `4.57.0`, ponieważ nowsze wydania zmieniają API generacji tekstu używane przez warstwę GPT modelu XTTS. Zainstaluj dokładnie tę wersję.

**Połączenie SSH padło i trening umarł razem z nim**

Uruchamiaj trening w `tmux` albo `screen`. Po ponownym zalogowaniu wróć do sesji przez `tmux attach -t trening`. Jeżeli proces już przepadł, wznów od ostatniego checkpointu przez `--restore-path`.
