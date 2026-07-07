# FanPulse World Cup MLOps

FanPulse World Cup is an end-to-end MLOps project for building **player-level fan perception analytics** from football tweets.

The project detects football player mentions in World Cup tweets, classifies the fan impression toward each specific player, and aggregates predictions into **FanPulse Player Power Rankings**.

This is not simple tweet-level sentiment analysis. The system works with tweet-player pairs:

```text
tweet text + candidate player → player-level fan impression
```

For the full machine learning development report, see:

```text
README_ML.md
```

---

## Project Goal

During World Cup matches, fans generate thousands of reactions on Twitter/X.

The goal of this project is to answer questions such as:

- Which players impressed fans the most?
- Which players received the most criticism?
- Which players generated the most discussion?
- Which players had the strongest positive or negative fan signal?
- How can we turn noisy football tweets into a player-level ranking?

The final product is a **FanPulse Power Ranking**:

```text
World Cup players ranked by fan perception, not objective football performance.
```

---

## Why Player-Level Classification?

Global tweet sentiment is not enough.

Example:

```text
Mbappe was electric, but Dembele killed every attack.
```

A normal sentiment model may classify the tweet as mixed or neutral.

FanPulse needs player-level labels:

```text
Kylian Mbappé   → positive
Ousmane Dembélé → negative
```

Therefore, the dataset is built as tweet-player pairs rather than tweet-level rows.

---

## Current Project Status

Completed:

- Raw World Cup tweet datasets collected.
- Raw tweets unified into one English deduplicated dataset.
- Player reference data prepared.
- Safe and contextual player aliases created.
- Tweet-player candidate pairs generated.
- Label Studio annotation workflow implemented.
- Manual datasets v1/v2 created and versioned.
- LLM-assisted, human-reviewed dataset v3 created.
- Model training pipeline implemented.
- Experiments tracked with W&B.
- Best model registered in W&B Model Registry.
- FanPulse Player Power Rankings generated from model predictions.
- Data and artifacts versioned with DVC/Git/W&B.

Current production model:

```text
fanpulse-impression-classifier:production
```

Current ranking output:

```text
data/rankings/player_power_rankings_v1.csv
```

---

## Project Lifecycle

```text
raw football tweets
→ tweet cleaning and unification
→ player alias matching
→ tweet-player candidate pairs
→ manual Label Studio annotation
→ dataset v1/v2
→ first supervised models
→ active sampling
→ LLM-assisted pre-labeling
→ human correction
→ dataset v3
→ augmented model training
→ W&B experiment tracking
→ W&B model registry
→ full-pool inference
→ FanPulse Player Power Rankings
```

---

## Data Sources

The project uses public football and FIFA World Cup 2022 tweet datasets, plus a player reference dataset.

Raw tweet datasets:

- [Tweets on Football World Cup 2022](https://www.kaggle.com/datasets/deepeshnigamdata/tweets-on-football-world-cup-2022)
- [FIFA World Cup 2022 Tweets](https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets)
- [FIFA WorldCup 2022 Tweets Dataset](https://www.kaggle.com/datasets/twtdata/fifa-worldcup-2022-tweets-dataset)

Player reference dataset:

- [World Cup 2022 Elite Players Image Dataset / List Of All Players Names](https://www.kaggle.com/datasets/peterkibuchi/world-cup-2022-elite-players-image-dataset?select=List+Of+All+Players+Names.csv)

Large data files are not committed directly to Git. They are versioned with DVC.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
├── configs/
│   └── train_*.yaml
├── data/
│   ├── raw/
│   │   └── tweets/
│   ├── interim/
│   ├── reference/
│   ├── labeling/
│   │   ├── review/
│   │   └── tasks/
│   ├── labeled/
│   │   └── exports/
│   ├── processed/
│   └── rankings/
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
│   ├── labeling/
│   │   ├── sample_v3_candidates.py
│   │   ├── llm_prelabel_v3.py
│   │   └── build_dataset_v3.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_augmented.py
│   ├── ranking/
│   │   └── build_power_rankings.py
│   ├── inference/
│   └── monitoring/
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.MD
└── README_ML.md
```

---

## Main Components

### 1. Data Pipeline

The data pipeline prepares tweet-player pairs from raw football tweets.

Main scripts:

```text
src/data/unify_raw_tweets.py
src/data/match_player_mentions.py
src/data/prepare_labeling_tasks.py
src/data/build_processed_dataset.py
```

Main outputs:

```text
data/interim/tweets_unified_english_dedup.csv
data/interim/matched_tweets_long_for_manual_labeling.csv
data/interim/matched_tweets_wide_for_manual_labeling.csv
data/interim/player_mentions_long_only_keys.csv
data/labeling/tasks/fanpulse_labeling_tasks_v1.json
data/processed/fanpulse_dataset_v1.csv
data/processed/fanpulse_dataset_v2.csv
```

---

### 2. Labeling Workflow

Manual labeling was done with Label Studio.

Label Studio task file:

```text
data/labeling/tasks/fanpulse_labeling_tasks_v1.json
```

Label Studio config:

```text
labeling/label_studio_config.xml
```

Manual exports:

```text
data/labeled/exports/label_studio_export_v1.json
data/labeled/exports/label_studio_export_v2.json
```

The labeling task asks:

```text
1. Is this tweet actually about the candidate player?
2. What is the fan impression toward this player?
3. What is the reason for the impression?
```

Main labels:

```text
strong_positive
positive
neutral
negative
strong_negative
not_about_player
```

---

### 3. Dataset v3 Expansion

Dataset v3 was created using an LLM-assisted, human-reviewed workflow.

The process:

```text
active sampling from matched tweet-player pool
→ LLM pre-labeling
→ human correction
→ clean corrected dataset v3
```

Important files:

```text
src/labeling/sample_v3_candidates.py
src/labeling/llm_prelabel_v3.py
src/labeling/build_dataset_v3.py
data/processed/fanpulse_dataset_v3.csv
```

Dataset v3 summary:

```text
Rows: 997
manual_v2 rows: 240
llm_assisted_human_reviewed rows: 757
```

Full labeling audit and ML details are documented in:

```text
README_ML.md
```

---

### 4. Model Training

The production model is a 4-class player-level fan impression classifier.

Target classes:

```text
positive
neutral
negative
not_about_player
```

Training scripts:

```text
src/training/train.py
src/training/train_augmented.py
```

Main production config:

```text
configs/train_tfidf_lr_4class_word_char_c40_v3_cw_strong.yaml
```

Production model:

```text
TF-IDF word + character n-grams
LogisticRegression
custom class weights
dataset v3
```

Current best validation metrics:

```text
accuracy:          0.6000
balanced_accuracy: 0.5310
macro-F1:          0.5358
weighted-F1:       0.5889
```

More details are in:

```text
README_ML.md
```

---

### 5. Experiment Tracking and Model Registry

Experiments are tracked with Weights & Biases.

W&B project:

```text
fanpulse-worldcup-mlops
```

Each run logs:

- hyperparameters;
- dataset version;
- model type;
- target mode;
- train/validation split metadata;
- accuracy;
- balanced accuracy;
- macro-F1;
- weighted-F1;
- confusion matrix;
- prediction distribution;
- model artifacts.

Registered model artifact:

```text
fanpulse-impression-classifier
```

Production aliases:

```text
latest
best
production
```

The production model can be loaded from W&B as:

```text
shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production
```

---

### 6. Player Power Rankings

The final product of the ML pipeline is a player-level ranking.

Ranking script:

```text
src/ranking/build_power_rankings.py
```

Ranking outputs:

```text
data/rankings/fanpulse_pair_predictions_v1.csv
data/rankings/player_power_rankings_v1.csv
data/rankings/player_leaderboards_v1.json
reports/player_power_rankings_top20.csv
```

The ranking combines:

```text
model sentiment probabilities
about-player confidence
mention volume
tweet engagement
Bayesian shrinkage
```

Current top players:

```text
1. Kylian Mbappé
2. Lionel Messi
3. Neymar
4. Jude Bellingham
5. Olivier Giroud
```

This ranking represents fan perception in the collected tweet data, not objective player performance.

Full ranking methodology is documented in:

```text
README_ML.md
```

---

## Setup

Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

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
joblib
pyyaml
wandb
openai
dvc
dvc-azure
```

Label Studio is run through Docker Compose, so it does not need to be installed into the Python environment.

---

## DVC Dataset Versioning

DVC is used to version large files and generated datasets.

Git stores:

- source code;
- configs;
- DVC metadata;
- lightweight reference files;
- Label Studio config;
- small reports;
- README files.

Git does not store large raw/interim/processed datasets directly.

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

Do not commit Azure keys, connection strings, API keys, or secrets to Git.

---

## DVC Tracking Strategy

Manual/input data is tracked with `.dvc` files:

```text
data/raw/tweets.dvc
data/labeled/exports/label_studio_export_v1.json.dvc
data/labeled/exports/label_studio_export_v2.json.dvc
data/processed/fanpulse_dataset_v3.csv.dvc
data/rankings/*.dvc
```

Generated data is also reproducible through pipeline scripts and DVC metadata:

```text
dvc.yaml
dvc.lock
```

---

## Reproduce Data Pipeline

Restore DVC-tracked data:

```bash
dvc pull
```

Run the DVC pipeline:

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

### 1. Unify raw tweets

```bash
PYTHONPATH=. python src/data/unify_raw_tweets.py \
  --raw-dir data/raw/tweets \
  --output data/interim/tweets_unified_english_dedup.csv \
  --detect-language-for-unknown
```

---

### 2. Match player mentions

```bash
PYTHONPATH=. python src/data/match_player_mentions.py \
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
PYTHONPATH=. python src/data/prepare_labeling_tasks.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --output data/labeling/tasks/fanpulse_labeling_tasks_v1.json
```

---

### 4. Build processed dataset from Label Studio export

Dataset v1:

```bash
PYTHONPATH=. python src/data/build_processed_dataset.py \
  --input data/labeled/exports/label_studio_export_v1.json \
  --output data/processed/fanpulse_dataset_v1.csv
```

Dataset v2:

```bash
PYTHONPATH=. python src/data/build_processed_dataset.py \
  --input data/labeled/exports/label_studio_export_v2.json \
  --output data/processed/fanpulse_dataset_v2.csv
```

---

## Run Label Studio

Start Label Studio:

```bash
docker compose up -d label-studio
```

Open:

```text
http://localhost:8080
```

Use labeling config:

```text
labeling/label_studio_config.xml
```

Import task file:

```text
data/labeling/tasks/fanpulse_labeling_tasks_v1.json
```

After annotation, export JSON and save it under:

```text
data/labeled/exports/
```

---

## Train Model

Login to W&B:

```bash
wandb login
```

Train the current production model:

```bash
PYTHONPATH=. python src/training/train_augmented.py \
  --config configs/train_tfidf_lr_4class_word_char_c40_v3_cw_strong.yaml
```

Train the earlier v2 baseline:

```bash
PYTHONPATH=. python src/training/train.py \
  --config configs/train_tfidf_lr_4class_word_char_c20.yaml
```

Model artifacts are saved locally under:

```text
artifacts/models/
```

and logged to W&B.

Local artifacts are ignored by Git.

---

## Build Player Power Rankings

Using local model artifact:

```bash
PYTHONPATH=. python src/ranking/build_power_rankings.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --model-path artifacts/models/tfidf_lr_4class_word_char_c40_v3_cw_strong/model.joblib \
  --min-effective-mentions 20 \
  --prior-strength 30 \
  --top-n 20
```

Using W&B production artifact:

```bash
PYTHONPATH=. python src/ranking/build_power_rankings.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --wandb-artifact shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production \
  --min-effective-mentions 20 \
  --prior-strength 30 \
  --top-n 20
```
---

## Inference API

The project includes a FastAPI inference service for serving the production FanPulse model.

The service loads the model from W&B Model Registry using the production alias:

```text
shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production
```

The model is not committed to the repository. It is downloaded from W&B at service startup and cached locally.

### Inference Endpoints

```text
GET  /health
GET  /model-info
POST /predict
POST /predict-batch
```

### Environment Variables

Create a local `.env` file:

```bash
cp .env.example .env
```

Set your W&B API key and model artifact reference:

```env
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=shinkaryovae-set-university
WANDB_PROJECT=fanpulse-worldcup-mlops
WANDB_MODEL_ARTIFACT=shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production
```

For local non-Docker execution, use:

```env
MODEL_CACHE_DIR=./model_cache
```

For Docker execution, `docker-compose.yml` overrides this value with:

```env
MODEL_CACHE_DIR=/app/model_cache
```

The model is downloaded from W&B Model Registry at service startup and cached in this directory.

Do not commit `.env`.

### Run Locally Without Docker

```bash
export $(cat .env | xargs)

PYTHONPATH=. uvicorn src.inference.app:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

### Run With Docker Compose

Build and start the inference service:

```bash
docker compose up --build inference-api
```

Health check:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production"
}
```

### Single Prediction Example

```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Mbappe scored two goals in the World Cup final. Unreal player.",
    "candidate_player": "Kylian Mbappé",
    "position": "Forward",
    "likes": 120,
    "retweets": 10
  }' | python -m json.tool
```

Example response:

```json
{
  "candidate_player": "Kylian Mbappé",
  "prediction": "positive",
  "confidence": 0.74,
  "probabilities": {
    "negative": 0.08,
    "neutral": 0.10,
    "not_about_player": 0.08,
    "positive": 0.74
  },
  "aboutness": 0.92,
  "sentiment_signal": 0.66,
  "model_version": "shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production",
  "latency_ms": 3.21
}
```

Actual probabilities may differ depending on the registered model version.

### Batch Prediction Example

```bash
curl -s -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {
        "text": "Bellingham is carrying England. Unreal player.",
        "candidate_player": "Jude Bellingham",
        "position": "Midfielder"
      },
      {
        "text": "Kane missed the penalty and England are out.",
        "candidate_player": "Harry Kane",
        "position": "Forward"
      }
    ]
  }' | python -m json.tool
```

### API Documentation

When the service is running, FastAPI documentation is available at:

```text
http://localhost:8000/docs
```


---

## Dataset and Model Versions

Git/DVC tags:

```text
dataset-v1
dataset-v2
dataset-v3
hw1-final
```

Check tags:

```bash
git tag
```

Restore a dataset version:

```bash
git checkout dataset-v2
dvc checkout
```

Restore latest data:

```bash
git checkout <branch-name>
dvc pull
dvc checkout
```

---

## Reports

Generated reports:

```text
reports/player_match_summary.csv
reports/alias_match_summary.csv
reports/player_power_rankings_top20.csv
```

These reports help inspect:

- matched player distribution;
- alias quality;
- player examples;
- possible matching noise;
- top FanPulse ranking results.

---

## Demo Checklist

### Data / Labeling

- Repository structure.
- Raw files tracked by DVC.
- Player reference files and alias tables.
- Data pipeline scripts in `src/data`.
- Label Studio config.
- Example Label Studio annotation.
- Processed dataset v1/v2/v3.
- DVC commands:

```bash
dvc status
dvc dag
dvc remote list
git tag
```

### Model Training

- Training configs in `configs/`.
- Training scripts in `src/training/`.
- W&B project with multiple runs.
- Metrics comparison.
- Confusion matrix.
- Prediction distribution.
- Registered W&B model artifact.

### Ranking Product

- Ranking script in `src/ranking/`.
- `player_power_rankings_v1.csv`.
- `player_leaderboards_v1.json`.
- Top 20 player ranking report.

---

---

## Monitoring and Observability

The project includes local monitoring for the FastAPI inference service using Prometheus and Grafana.

The inference API exposes Prometheus metrics at:

```text
GET /metrics
```

Prometheus scrapes the inference service every 5 seconds, and Grafana visualizes the collected metrics.

### Monitored Metrics

The service exposes the following custom metrics:

```text
fanpulse_predictions_total
fanpulse_prediction_latency_seconds
fanpulse_prediction_errors_total
fanpulse_http_requests_total
fanpulse_http_request_latency_seconds
fanpulse_model_loaded
fanpulse_model_info
```

The Grafana dashboard shows:

- predictions per minute;
- average prediction latency;
- p95 prediction latency;
- predictions by class;
- total predictions;
- model loaded status;
- prediction errors.

### Run Monitoring Stack

Start the inference API, Prometheus, and Grafana:

```bash
docker compose up --build inference-api prometheus grafana
```

Services:

```text
Inference API: http://localhost:8000
Prometheus:    http://localhost:9090
Grafana:       http://localhost:3000
```

Grafana credentials:

```text
username: admin
password: admin
```

### Check Metrics Endpoint

```bash
curl -s http://localhost:8000/metrics | grep fanpulse
```

### Generate Demo Traffic

```bash
N=50 ./scripts/send_demo_requests.sh
```

### Prometheus

Open:

```text
http://localhost:9090
```

Check:

```text
Status → Targets
```

The `fanpulse-inference-api` target should be `UP`.

Useful Prometheus queries:

```promql
sum(increase(fanpulse_predictions_total[1m]))
```

```promql
1000 * sum(rate(fanpulse_prediction_latency_seconds_sum[1m])) / sum(rate(fanpulse_prediction_latency_seconds_count[1m]))
```

```promql
1000 * histogram_quantile(0.95, sum(rate(fanpulse_prediction_latency_seconds_bucket[5m])) by (le))
```

```promql
sum(increase(fanpulse_predictions_total[5m])) by (prediction)
```

### Grafana Dashboard

Open:

```text
http://localhost:3000
```

Go to:

```text
Dashboards → FanPulse → FanPulse Inference Monitoring
```

The dashboard is provisioned from:

```text
monitoring/grafana/dashboards/fanpulse-inference-dashboard.json
```

Datasource provisioning:

```text
monitoring/grafana/provisioning/datasources/prometheus.yml
```

Dashboard provisioning:

```text
monitoring/grafana/provisioning/dashboards/dashboard.yml
```

---

## License and Data Notes

This repository contains code and metadata for educational use.

Raw datasets are sourced from public Kaggle datasets. Users should follow the terms and licenses of the original datasets when downloading or reusing the data.

Large raw files are not committed directly to Git and should be restored through DVC or downloaded from the original sources.