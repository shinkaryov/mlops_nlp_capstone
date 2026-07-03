# FanPulse World Cup MLOps

## Problem
Football fans generate thousands of reactions during World Cup matches.
This project detects player-level fan impression from football tweets and builds a player hype/criticism leaderboard.

## Dataset
Raw football tweets are collected from public football/World Cup tweet datasets.
A curated player dictionary and alias tables are used to generate candidate tweet-player pairs.

## Data Lineage
raw tweets
→ English deduped tweets
→ player alias matching
→ candidate tweet-player pairs
→ Label Studio manual validation
→ labeled dataset v1/v2
→ model training dataset

## Labeling
Tool: Label Studio.
Each task contains a tweet and a candidate player.
The annotator validates whether the tweet is about the player and labels fan impression.

Labels:
- strong_positive
- positive
- neutral
- negative
- strong_negative
- not_about_player

## Dataset Versioning
DVC is used to version raw, interim, labeled and processed datasets.

## How to run Label Studio
docker compose up -d label-studio

Open:
http://localhost:8080

Import:
data/labeling/tasks/fanpulse_labeling_tasks_v1.json

Use labeling config:
labeling/label_studio_config.xml

## Reproduce Data
dvc pull
python src/data/prepare_labeling_tasks.py
python src/data/build_processed_dataset.py

## Future MLOps steps
- Train impression classifier
- Track experiments in W&B/MLflow
- Register model
- Serve with FastAPI
- Monitor with Prometheus/Grafana/Evidently
- Deploy with GitHub Actions to Azure