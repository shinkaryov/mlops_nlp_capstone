import json
import pandas as pd
from pathlib import Path

INPUT = Path("data/interim/matched_tweets_long_for_manual_labeling.csv")
OUTPUT = Path("data/labeling/tasks/fanpulse_labeling_tasks_v1.json")

N_PER_TOP_PLAYER = 8
N_RANDOM_OTHER = 80

df = pd.read_csv(INPUT)

print(df.columns.tolist())

TEXT_COL = "tweet_text"
PLAYER_COL = "player_name"

df = df.dropna(subset=[TEXT_COL, PLAYER_COL]).copy()

df["text_len"] = df[TEXT_COL].astype(str).str.len()
df = df[df["text_len"] >= 20]

top_players = df[PLAYER_COL].value_counts().head(20).index.tolist()

samples = []

for player in top_players:
    part = df[df[PLAYER_COL] == player].sample(
        n=min(N_PER_TOP_PLAYER, len(df[df[PLAYER_COL] == player])),
        random_state=42
    )
    samples.append(part)

sample_df = pd.concat(samples, ignore_index=True)

other_df = df[~df[PLAYER_COL].isin(top_players)]
if len(other_df) > 0:
    sample_df = pd.concat([
        sample_df,
        other_df.sample(n=min(N_RANDOM_OTHER, len(other_df)), random_state=42)
    ], ignore_index=True)

sample_df = sample_df.drop_duplicates(subset=[TEXT_COL, PLAYER_COL])

tasks = []

for _, row in sample_df.iterrows():
    task = {
        "text": str(row[TEXT_COL]),
        "candidate_player": str(row[PLAYER_COL]),
        "source_file": str(row.get("source_file", "")),
        "matched_aliases": str(row.get("matched_aliases", row.get("alias", ""))),
        "match_method": str(row.get("match_method", "")),
        "likes": int(row.get("Number of Likes", row.get("likes", 0)) or 0)
    }
    tasks.append(task)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"Saved {len(tasks)} tasks to {OUTPUT}")