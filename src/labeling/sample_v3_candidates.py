import argparse
import hashlib
from pathlib import Path

import joblib
import pandas as pd


def stable_hash(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:20]


def make_model_input(df: pd.DataFrame) -> pd.Series:
    candidate_player = df["candidate_player"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    position = df["position"].fillna("").astype(str) if "position" in df.columns else ""

    return (
        "candidate_player: " + candidate_player
        + " position: " + position
        + " tweet: " + text
    )


def normalize_unlabeled_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["text"] = df["tweet_text"].fillna("").astype(str)
    out["candidate_player"] = df["player_name"].fillna("").astype(str)
    out["player_id"] = df.get("player_id", "")
    out["position"] = df.get("position", "")
    out["source_file"] = df.get("source_file", "")
    out["tweet_key"] = df.get("tweet_key", "")
    out["matched_aliases"] = df.get("matched_aliases", "")
    out["match_method"] = df.get("match_methods", "")
    out["best_match_score"] = df.get("best_match_score", "")
    out["likes"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0)
    out["retweets"] = pd.to_numeric(df.get("retweet_count", 0), errors="coerce").fillna(0)

    out["row_hash"] = out.apply(
        lambda r: stable_hash(str(r["text"]) + "|" + str(r["candidate_player"])),
        axis=1,
    )

    return out


def add_bucket(selected, pool, bucket_name, n, random_state):
    if n <= 0 or pool.empty:
        return selected

    already = set(selected["row_hash"]) if not selected.empty else set()
    candidates = pool[~pool["row_hash"].isin(already)].copy()

    if candidates.empty:
        return selected

    take_n = min(n, len(candidates))

    sampled = candidates.sample(n=take_n, random_state=random_state).copy()
    sampled["sample_bucket"] = bucket_name

    return pd.concat([selected, sampled], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-path", default="data/processed/fanpulse_dataset_v2.csv")
    parser.add_argument("--unlabeled-path", default="data/interim/matched_tweets_long_for_manual_labeling.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="data/labeling/review/fanpulse_v3_candidates.csv")
    parser.add_argument("--target-size", type=int, default=760)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    manual = pd.read_csv(args.manual_path)
    manual["row_hash"] = manual.apply(
        lambda r: stable_hash(str(r["text"]) + "|" + str(r["candidate_player"])),
        axis=1,
    )
    manual_hashes = set(manual["row_hash"])

    raw = pd.read_csv(args.unlabeled_path, low_memory=False)
    pool = normalize_unlabeled_pool(raw)

    pool = pool[~pool["row_hash"].isin(manual_hashes)].copy()
    pool = pool.drop_duplicates("row_hash").reset_index(drop=True)

    # Mark multi-player tweets if tweet_key exists.
    if "tweet_key" in pool.columns:
        tweet_player_counts = pool.groupby("tweet_key")["candidate_player"].transform("nunique")
        pool["is_multi_player_tweet"] = tweet_player_counts.gt(1)
    else:
        pool["is_multi_player_tweet"] = False

    model = joblib.load(args.model_path)
    X = make_model_input(pool)

    proba = model.predict_proba(X)
    classes = list(model.classes_)

    pred_ids = proba.argmax(axis=1)
    pool["model_pred_label"] = [classes[i] for i in pred_ids]
    pool["model_confidence"] = proba.max(axis=1)

    pool["engagement_score"] = pool["likes"].fillna(0) + 2 * pool["retweets"].fillna(0)

    selected = pd.DataFrame()

    # Main idea: select informative examples, not random examples.
    selected = add_bucket(
        selected,
        pool[(pool["model_confidence"] >= 0.35) & (pool["model_confidence"] <= 0.55)],
        "uncertain_035_055",
        250,
        args.random_state,
    )

    selected = add_bucket(
        selected,
        pool[pool["model_pred_label"].eq("negative")].sort_values("model_confidence", ascending=False),
        "predicted_negative",
        170,
        args.random_state + 1,
    )

    selected = add_bucket(
        selected,
        pool[pool["model_pred_label"].eq("not_about_player")].sort_values("model_confidence", ascending=False),
        "predicted_not_about_player",
        140,
        args.random_state + 2,
    )

    selected = add_bucket(
        selected,
        pool[pool["model_pred_label"].eq("neutral")].sort_values("model_confidence", ascending=False),
        "predicted_neutral",
        120,
        args.random_state + 3,
    )

    selected = add_bucket(
        selected,
        pool[pool["is_multi_player_tweet"].eq(True)],
        "multi_player_tweet",
        50,
        args.random_state + 4,
    )

    selected = add_bucket(
        selected,
        pool.sort_values("engagement_score", ascending=False).head(5000),
        "high_engagement",
        30,
        args.random_state + 5,
    )

    # Fill remaining rows randomly from the pool.
    remaining = args.target_size - len(selected)
    if remaining > 0:
        selected = add_bucket(
            selected,
            pool,
            "random_fill",
            remaining,
            args.random_state + 6,
        )

    selected = selected.head(args.target_size).reset_index(drop=True)
    selected.insert(0, "llm_input_id", range(1, len(selected) + 1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)

    print(f"Pool rows after removing manual labels: {len(pool)}")
    print(f"Selected rows: {len(selected)}")
    print("\nSample buckets:")
    print(selected["sample_bucket"].value_counts())
    print("\nModel predicted labels:")
    print(selected["model_pred_label"].value_counts())
    print("\nModel confidence:")
    print(selected["model_confidence"].describe())
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()