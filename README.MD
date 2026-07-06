# FanPulse World Cup MLOps

FanPulse World Cup is an end-to-end MLOps project for detecting **player-level fan impression** from football tweets.

The goal is not just to classify a tweet as positive or negative. The project detects which football player is mentioned in a tweet, validates whether the tweet is actually about that player, assigns a fan impression label, and prepares the data for a future player hype/criticism leaderboard.

For the full machine learning development cycle, see [README_ML.md](README_ML.md).
---

## Problem

During World Cup matches, football fans generate thousands of reactions on Twitter/X and other social platforms.

Sports media, fan communities, and fantasy football users often need a quick way to answer questions such as:

- Which players impressed fans the most?
- Which players received the most criticism?
- Which players generated the most discussion?
- Are fans reacting positively or negatively to a specific player?

This project builds the data and MLOps foundation for a player-level fan perception analytics system.

---

## Project Objective

The project implements the first stage of a full MLOps lifecycle:

```text
raw football tweets
→ unified English tweet dataset
→ player mention matching
→ Label Studio manual validation
→ versioned labeled dataset
→ future model training
→ future inference API
→ future monitoring and CI/CD
```

The current repository focuses on **Data Management / Labeling / Dataset Versioning**, which is the foundation for the next MLOps stages.

---

## Data Sources

The project uses public football and FIFA World Cup 2022 tweet datasets, plus an elite players list.

Raw tweet datasets:

- [Tweets on Football World Cup 2022](https://www.kaggle.com/datasets/deepeshnigamdata/tweets-on-football-world-cup-2022)
- [FIFA World Cup 2022 Tweets](https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets)
- [FIFA WorldCup 2022 Tweets Dataset](https://www.kaggle.com/datasets/twtdata/fifa-worldcup-2022-tweets-dataset)

Player reference dataset:

- [World Cup 2022 Elite Players Image Dataset / List Of All Players Names](https://www.kaggle.com/datasets/peterkibuchi/world-cup-2022-elite-players-image-dataset?select=List+Of+All+Players+Names.csv)

The raw files are not committed directly to Git because they can be large. They are versioned with DVC.

---

## Current Data Files

Raw tweet files are stored under:

```text
data/raw/tweets/
```

Expected raw files:

```text
data/raw/tweets/fifa_world_cup_2022_tweets.csv
data/raw/tweets/tweets_football.csv
data/raw/tweets/tweets1.xlsx
data/raw/tweets/tweets2.xlsx
data/raw/tweets/twtdata.com_tweets_by_hashtag_Fifa_Xd8b0u2i9n.csv
```

Reference files are stored under:

```text
data/reference/
```

Important reference files:

```text
data/reference/players.csv
data/reference/players_curated_clean.csv
data/reference/player_safe_aliases.csv
data/reference/player_contextual_aliases.csv
```

---

## Data Lineage

The project follows this data lineage:

```text
Raw football tweet files
        ↓
Unified English deduplicated tweets
        ↓
Player alias matching
        ↓
Candidate tweet-player pairs
        ↓
Balanced Label Studio task file
        ↓
Manual validation and impression labeling
        ↓
Label Studio export v1/v2
        ↓
Processed labeled dataset v1/v2
        ↓
Future model training dataset
```

Main generated files:

```text
data/interim/tweets_unified_english_dedup.csv
data/interim/matched_tweets_long_for_manual_labeling.csv
data/interim/matched_tweets_wide_for_manual_labeling.csv
data/interim/player_mentions_long_only_keys.csv

data/labeling/tasks/fanpulse_labeling_tasks_v1.json

data/labeled/exports/label_studio_export_v1.json
data/labeled/exports/label_studio_export_v2.json

data/processed/fanpulse_dataset_v1.csv
data/processed/fanpulse_dataset_v2.csv
```

---

## What is `fanpulse_labeling_tasks_v1.json`?

`data/labeling/tasks/fanpulse_labeling_tasks_v1.json` is the file imported into Label Studio.

It is **not** the final labeled dataset.

It contains candidate examples to annotate. Each task contains:

- tweet text;
- candidate player;
- player id;
- position;
- matched aliases;
- match method;
- engagement metadata;
- source file.

Example task structure:

```json
{
  "text": "Bellingham is carrying England again. Unreal player.",
  "candidate_player": "Jude Bellingham",
  "player_id": "20",
  "position": "Midfielder",
  "matched_aliases": "bellingham",
  "match_method": "safe_alias",
  "likes": 120,
  "retweets": 5,
  "tweet_key": "id_123456789"
}
```

The actual manual labels are created inside Label Studio and exported to:

```text
data/labeled/exports/label_studio_export_v1.json
data/labeled/exports/label_studio_export_v2.json
```

The final training-ready datasets are:

```text
data/processed/fanpulse_dataset_v1.csv
data/processed/fanpulse_dataset_v2.csv
```

---

## Labeling Task

Tool used: **Label Studio**

Each annotation task shows:

- the tweet text;
- a candidate player detected by alias matching;
- metadata about how the candidate player was matched.

The annotator must answer:

1. Is this tweet actually about the candidate player?
2. What is the fan impression toward this player?
3. Optional: what is the reason for the impression?

---

## Label Schema

Main impression labels:

```text
strong_positive
positive
neutral
negative
strong_negative
not_about_player
```

Additional validation label:

```text
is_about_player:
- yes
- no
- unclear
```

Optional reason labels:

```text
performance
goal_or_assist
mistake
hype
comparison
meme_or_joke
news_context
other
```

---

## Why Player-Level Labeling?

Global sentiment analysis is not enough for this task.

Example:

```text
"Mbappe was electric, but Dembele killed every attack."
```

A normal sentiment model may classify the whole tweet as mixed or neutral.

But this project needs player-level labels:

```text
Kylian Mbappe  → positive
Ousmane Dembele → negative
```

This is why the dataset is built as tweet-player pairs, not just tweet-level sentiment rows.

---

## Player Matching Strategy

The project uses deterministic player alias matching to generate candidate tweet-player pairs.

There are two alias types:

### Safe aliases

Safe aliases are matched directly.

Examples:

```text
messi → Lionel Messi
ronaldo → Cristiano Ronaldo
mbappe → Kylian Mbappe
bellingham → Jude Bellingham
```

### Contextual aliases

Contextual aliases are matched only if supporting context terms are present.

Example:

```text
walker → Kyle Walker
```

This alias is accepted only if the tweet contains context such as:

```text
england
man city
city
mbappe
france
```

This reduces false positives.

### Fuzzy matching

Broad fuzzy matching is intentionally disabled by default because it can introduce noisy matches.

Example of a risky false positive:

```text
cancel → Cancelo
```

For the current MVP, the project uses high-precision deterministic alias matching and manual validation in Label Studio.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
├── data/
│   ├── raw/
│   │   └── tweets/
│   ├── interim/
│   ├── reference/
│   ├── labeling/
│   │   └── tasks/
│   ├── labeled/
│   │   └── exports/
│   └── processed/
├── docker/
├── labeling/
│   └── label_studio_config.xml
├── monitoring/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   │   ├── unify_raw_tweets.py
│   │   ├── match_player_mentions.py
│   │   ├── prepare_labeling_tasks.py
│   │   └── build_processed_dataset.py
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.MD
```

---

## Setup

Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Minimal expected dependencies:

```text
pandas
numpy
openpyxl
tqdm
rapidfuzz
lingua-language-detector
scikit-learn
dvc
dvc-azure
```

Label Studio is run through Docker Compose, so it does not need to be installed into the Python environment.

---

## DVC Dataset Versioning

DVC is used to version large data files and generated datasets.

Git stores:

- source code;
- DVC metadata;
- lightweight reference files;
- Label Studio config;
- small reports;
- README.

Git does **not** store large raw/interim/processed datasets directly.

Large files are restored through DVC.

---

## DVC Remote Storage

The project can use Azure Blob Storage as DVC remote storage.

Check configured DVC remotes:

```bash
dvc remote list
```

Example Azure remote setup:

```bash
dvc remote add -d azure_remote azure://<container-name>/data
dvc remote modify azure_remote account_name <storage-account-name>
```

Login to Azure:

```bash
az login
```

Pull data from remote:

```bash
dvc pull
```

Push updated DVC data:

```bash
dvc push
```

Do not commit Azure keys, connection strings, or secrets to Git.

---

## DVC Tracking Strategy

Manual/input data is tracked with `.dvc` files:

```text
data/raw/tweets.dvc
data/labeled/exports/label_studio_export_v1.json.dvc
data/labeled/exports/label_studio_export_v2.json.dvc
```

Generated data is produced through `dvc.yaml` stages and tracked in `dvc.lock`:

```text
data/interim/*
data/processed/*
```

This separation keeps the pipeline reproducible:

```text
manual inputs → DVC .dvc pointers
generated outputs → dvc.yaml / dvc.lock
```

---

## Reproduce Data Pipeline

Restore DVC-tracked inputs:

```bash
dvc pull
```

Run the full DVC pipeline:

```bash
dvc repro
```

Show pipeline DAG:

```bash
dvc dag
```

Check DVC status:

```bash
dvc status
```

---

## Run Data Pipeline Manually

### 1. Unify raw tweet datasets

```bash
python src/data/unify_raw_tweets.py \
  --raw-dir data/raw/tweets \
  --output data/interim/tweets_unified_english_dedup.csv \
  --detect-language-for-unknown
```

Output:

```text
data/interim/tweets_unified_english_dedup.csv
```

---

### 2. Match player mentions

```bash
python src/data/match_player_mentions.py \
  --tweets data/interim/tweets_unified_english_dedup.csv \
  --players data/reference/players.csv \
  --interim-dir data/interim \
  --reports-dir reports
```

Outputs:

```text
data/interim/matched_tweets_long_for_manual_labeling.csv
data/interim/matched_tweets_wide_for_manual_labeling.csv
data/interim/player_mentions_long_only_keys.csv
reports/player_match_summary.csv
reports/alias_match_summary.csv
```

---

### 3. Prepare Label Studio tasks

```bash
python src/data/prepare_labeling_tasks.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --output data/labeling/tasks/fanpulse_labeling_tasks_v1.json
```

Output:

```text
data/labeling/tasks/fanpulse_labeling_tasks_v1.json
```

This file is imported into Label Studio.

---

### 4. Build processed dataset from Label Studio export

Dataset v1:

```bash
python src/data/build_processed_dataset.py \
  --input data/labeled/exports/label_studio_export_v1.json \
  --output data/processed/fanpulse_dataset_v1.csv
```

Dataset v2:

```bash
python src/data/build_processed_dataset.py \
  --input data/labeled/exports/label_studio_export_v2.json \
  --output data/processed/fanpulse_dataset_v2.csv
```

Outputs:

```text
data/processed/fanpulse_dataset_v1.csv
data/processed/fanpulse_dataset_v2.csv
```

---

## Run Label Studio

Start Label Studio:

```bash
docker compose up -d label-studio
```

Open in browser:

```text
http://localhost:8080
```

Create a new Label Studio project.

Use labeling config:

```text
labeling/label_studio_config.xml
```

Import task file:

```text
data/labeling/tasks/fanpulse_labeling_tasks_v1.json
```

After manual labeling, export annotations as JSON and save them to:

```text
data/labeled/exports/label_studio_export_v1.json
data/labeled/exports/label_studio_export_v2.json
```

Then rebuild processed datasets:

```bash
python src/data/build_processed_dataset.py \
  --input data/labeled/exports/label_studio_export_v1.json \
  --output data/processed/fanpulse_dataset_v1.csv
```

---

## Dataset Versions

The repository contains two dataset versions:

```text
dataset-v1 — initial manually labeled dataset
dataset-v2 — extended manually labeled dataset
```

List tags:

```bash
git tag
```

Checkout dataset v1:

```bash
git checkout dataset-v1
dvc checkout
```

Checkout dataset v2:

```bash
git checkout dataset-v2
dvc checkout
```

---

## Current Status

Completed:

- Raw football tweet files collected.
- Raw tweet datasets unified into one English deduplicated dataset.
- Curated player list prepared.
- Safe and contextual player aliases generated.
- Candidate tweet-player pairs generated.
- Balanced Label Studio task file prepared.
- Label Studio labeling config created.
- Manual Label Studio exports saved as v1/v2.
- Processed labeled datasets created as v1/v2.
- Dataset versions are tracked with Git tags and DVC.

---

## Reports

The project generates lightweight reports for data quality checks:

```text
reports/player_match_summary.csv
reports/alias_match_summary.csv
```

These reports help inspect:

- most frequently matched players;
- aliases that produced matches;
- example tweets;
- potential matching noise;
- distribution of matched players.

---

## Demo Checklist for Data Management Homework

A short demo should show:

1. Repository structure.
2. Raw files tracked by DVC.
3. Player reference files and alias tables.
4. Data pipeline scripts in `src/data`.
5. Label Studio project.
6. One example of manual annotation.
7. Label Studio export file.
8. Processed dataset v1/v2.
9. Git tags `dataset-v1` and `dataset-v2`.
10. DVC commands:

```bash
dvc status
dvc dag
dvc remote list
git tag
```

---

## Future MLOps Steps

The next stages of the project will extend this data layer into a full MLOps system.

Planned stages:

- Train a player-level fan impression classifier.
- Track experiments with W&B or MLflow.
- Register the best model in a model registry.
- Serve the model through FastAPI.
- Add `/predict`, `/leaderboard`, `/health`, and `/metrics` endpoints.
- Monitor latency and prediction volume with Prometheus and Grafana.
- Add data drift checks with Evidently.
- Add GitHub Actions for CI/CD.
- Deploy the inference service to Azure Container Apps.

---

## Planned API Example

Future `/predict` request:

```json
{
  "text": "Bellingham is carrying England again. Unreal player.",
  "likes": 120
}
```

Expected response:

```json
{
  "mentions": [
    {
      "player": "Jude Bellingham",
      "impression": "strong_positive",
      "confidence": 0.89,
      "fan_impression_score": 10.43
    }
  ],
  "model_version": "fanpulse-impression-classifier:production"
}
```

Future `/leaderboard` response:

```json
{
  "top_impressed": [
    {
      "player": "Jude Bellingham",
      "score": 87.4,
      "mentions": 42
    }
  ],
  "most_criticized": [
    {
      "player": "Kylian Mbappe",
      "score": -35.2,
      "mentions": 18
    }
  ]
}
```

---

## License and Data Notes

This repository contains code and metadata for educational use.

Raw datasets are sourced from public Kaggle datasets. Users should follow the terms and licenses of the original datasets when downloading or reusing the data.

Large raw files are not committed directly to Git and should be restored through DVC or downloaded from the original sources.