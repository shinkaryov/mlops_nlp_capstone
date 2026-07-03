import json
import pandas as pd
from pathlib import Path

INPUT = Path("data/labeled/exports/label_studio_export_v2.json")
OUTPUT = Path("data/processed/fanpulse_dataset_v2.csv")

def get_choice(annotation, name):
    for result in annotation.get("result", []):
        if result.get("from_name") == name:
            choices = result.get("value", {}).get("choices", [])
            if choices:
                return "|".join(choices)
    return None

with open(INPUT, "r", encoding="utf-8") as f:
    tasks = json.load(f)

rows = []

for task in tasks:
    data = task.get("data", {})
    annotations = task.get("annotations", [])

    if not annotations:
        continue

    ann = annotations[0]

    rows.append({
        "text": data.get("text"),
        "candidate_player": data.get("candidate_player"),
        "source_file": data.get("source_file"),
        "matched_aliases": data.get("matched_aliases"),
        "match_method": data.get("match_method"),
        "likes": data.get("likes"),
        "is_about_player": get_choice(ann, "is_about_player"),
        "impression": get_choice(ann, "impression"),
        "reason": get_choice(ann, "reason"),
    })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT, index=False)

print(df.shape)
print(df["impression"].value_counts(dropna=False))
print(f"Saved to {OUTPUT}")