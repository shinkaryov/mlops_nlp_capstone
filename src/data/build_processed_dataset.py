import argparse
import json
from pathlib import Path
import pandas as pd


def get_choice(annotation: dict, name: str) -> str | None:
    for result in annotation.get("result", []):
        if result.get("from_name") == name:
            choices = result.get("value", {}).get("choices", [])
            if choices:
                return "|".join(choices)
    return None


def build_processed_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    rows = []

    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations", [])

        if not annotations:
            continue

        # Use the first completed annotation.
        ann = annotations[0]

        row = {
            "text": data.get("text"),
            "candidate_player": data.get("candidate_player"),
            "player_id": data.get("player_id"),
            "position": data.get("position"),
            "source_file": data.get("source_file"),
            "tweet_key": data.get("tweet_key"),
            "matched_aliases": data.get("matched_aliases"),
            "match_method": data.get("match_method"),
            "best_match_score": data.get("best_match_score"),
            "likes": data.get("likes"),
            "retweets": data.get("retweets"),
            "is_about_player": get_choice(ann, "is_about_player"),
            "impression": get_choice(ann, "impression"),
            "reason": get_choice(ann, "reason"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No annotated rows found in {input_path}")

    df["label_is_valid"] = (
        df["is_about_player"].eq("yes")
        & df["impression"].notna()
        & ~df["impression"].eq("not_about_player")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Input tasks: {len(tasks)}")
    print(f"Annotated rows: {len(df)}")
    print(f"Valid training rows: {df['label_is_valid'].sum()}")
    print("\nImpression distribution:")
    print(df["impression"].value_counts(dropna=False))
    print(f"\nSaved: {output_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_processed_dataset(input_path=args.input, output_path=args.output)