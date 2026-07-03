import argparse
import json
from pathlib import Path
import pandas as pd


def pick_text_col(df: pd.DataFrame) -> str:
    for col in ["tweet_text", "Tweet", "text"]:
        if col in df.columns:
            return col
    raise ValueError("Cannot find tweet text column. Expected one of: tweet_text, Tweet, text")


def pick_player_col(df: pd.DataFrame) -> str:
    for col in ["player_name", "player", "candidate_player"]:
        if col in df.columns:
            return col
    raise ValueError("Cannot find player column. Expected one of: player_name, player, candidate_player")


def prepare_labeling_tasks(
    input_path: Path,
    output_path: Path,
    n_per_top_player: int,
    n_random_other: int,
    top_players_count: int,
    min_text_len: int,
    random_state: int,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)

    text_col = pick_text_col(df)
    player_col = pick_player_col(df)

    df = df.dropna(subset=[text_col, player_col]).copy()
    df["text_len"] = df[text_col].astype(str).str.len()
    df = df[df["text_len"] >= min_text_len].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check input data.")

    top_players = df[player_col].value_counts().head(top_players_count).index.tolist()

    samples = []

    for player in top_players:
        part = df[df[player_col] == player]
        part_sample = part.sample(
            n=min(n_per_top_player, len(part)),
            random_state=random_state,
        )
        samples.append(part_sample)

    sample_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    other_df = df[~df[player_col].isin(top_players)]

    if len(other_df) > 0 and n_random_other > 0:
        other_sample = other_df.sample(
            n=min(n_random_other, len(other_df)),
            random_state=random_state,
        )
        sample_df = pd.concat([sample_df, other_sample], ignore_index=True)

    sample_df = sample_df.drop_duplicates(subset=[text_col, player_col]).reset_index(drop=True)

    tasks = []

    for _, row in sample_df.iterrows():
        likes = row.get("like_count", 0)
        retweets = row.get("retweet_count", 0)

        try:
            likes = int(0 if pd.isna(likes) else likes)
        except Exception:
            likes = 0

        try:
            retweets = int(0 if pd.isna(retweets) else retweets)
        except Exception:
            retweets = 0

        task = {
            "text": str(row[text_col]),
            "candidate_player": str(row[player_col]),
            "player_id": str(row.get("player_id", "")),
            "position": str(row.get("position", "")),
            "source_file": str(row.get("source_file", "")),
            "matched_aliases": str(row.get("matched_aliases", row.get("matched_alias", ""))),
            "match_method": str(row.get("match_methods", row.get("match_method", ""))),
            "best_match_score": float(row.get("best_match_score", row.get("match_score", 0)) or 0),
            "likes": likes,
            "retweets": retweets,
            "tweet_key": str(row.get("tweet_key", "")),
        }
        tasks.append(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Input rows: {len(df)}")
    print(f"Saved tasks: {len(tasks)}")
    print(f"Output: {output_path}")

    print("\nTasks per player:")
    print(sample_df[player_col].value_counts().head(30))

    return sample_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/interim/matched_tweets_long_for_manual_labeling.csv",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="data/labeling/tasks/fanpulse_labeling_tasks_v1.json",
        type=Path,
    )
    parser.add_argument("--n-per-top-player", default=8, type=int)
    parser.add_argument("--n-random-other", default=80, type=int)
    parser.add_argument("--top-players-count", default=20, type=int)
    parser.add_argument("--min-text-len", default=20, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prepare_labeling_tasks(
        input_path=args.input,
        output_path=args.output,
        n_per_top_player=args.n_per_top_player,
        n_random_other=args.n_random_other,
        top_players_count=args.top_players_count,
        min_text_len=args.min_text_len,
        random_state=args.random_state,
    )