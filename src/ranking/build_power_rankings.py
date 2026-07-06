import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REQUIRED_CLASSES = ["positive", "negative", "neutral", "not_about_player"]


def stable_hash(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:20]


def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def first_existing_column(df: pd.DataFrame, candidates: list[str], default="") -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def normalize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports both:
    - matched_tweets_long_for_manual_labeling.csv schema
    - processed dataset schema
    """
    out = pd.DataFrame()

    out["text"] = first_existing_column(df, ["tweet_text", "text"]).fillna("").astype(str)
    out["candidate_player"] = first_existing_column(df, ["player_name", "candidate_player"]).fillna("").astype(str)
    out["player_id"] = first_existing_column(df, ["player_id"]).fillna("").astype(str)
    out["position"] = first_existing_column(df, ["position"]).fillna("").astype(str)

    out["source_file"] = first_existing_column(df, ["source_file"]).fillna("").astype(str)
    out["tweet_key"] = first_existing_column(df, ["tweet_key"]).fillna("").astype(str)
    out["tweet_id"] = first_existing_column(df, ["tweet_id"]).fillna("").astype(str)
    out["created_at_utc"] = first_existing_column(df, ["created_at_utc"]).fillna("").astype(str)

    out["matched_aliases"] = first_existing_column(df, ["matched_aliases"]).fillna("").astype(str)
    out["match_method"] = first_existing_column(df, ["match_methods", "match_method"]).fillna("").astype(str)
    out["best_match_score"] = safe_numeric(first_existing_column(df, ["best_match_score"], default=100), 100)

    out["likes"] = safe_numeric(first_existing_column(df, ["like_count", "likes"], default=0), 0)
    out["retweets"] = safe_numeric(first_existing_column(df, ["retweet_count", "retweets"], default=0), 0)

    out["row_hash"] = out.apply(
        lambda r: stable_hash(str(r["text"]) + "|" + str(r["candidate_player"])),
        axis=1,
    )

    # One row per tweet-player pair.
    out = out.drop_duplicates("row_hash").reset_index(drop=True)

    return out


def make_model_input(df: pd.DataFrame) -> pd.Series:
    return (
        "candidate_player: " + df["candidate_player"].fillna("").astype(str)
        + " position: " + df["position"].fillna("").astype(str)
        + " tweet: " + df["text"].fillna("").astype(str)
    )


def load_model(model_path: str | None, wandb_artifact: str | None):
    if model_path:
        return joblib.load(model_path)

    if wandb_artifact:
        import wandb

        run = wandb.init(
            project="fanpulse-worldcup-mlops",
            job_type="load-ranking-model",
        )
        artifact = run.use_artifact(wandb_artifact, type="model")
        artifact_dir = artifact.download()
        run.finish()

        return joblib.load(Path(artifact_dir) / "model.joblib")

    raise ValueError("Provide either --model-path or --wandb-artifact")


def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    X = make_model_input(df)

    proba = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]

    proba_df = pd.DataFrame(
        proba,
        columns=[f"p_{c}" for c in classes],
        index=df.index,
    )

    for cls in REQUIRED_CLASSES:
        col = f"p_{cls}"
        if col not in proba_df.columns:
            proba_df[col] = 0.0

    pred_idx = proba.argmax(axis=1)
    df["pred_label"] = [classes[i] for i in pred_idx]
    df["pred_confidence"] = proba.max(axis=1)

    df = pd.concat([df, proba_df], axis=1)

    df["aboutness"] = 1.0 - df["p_not_about_player"]
    df["sentiment_signal"] = df["p_positive"] - df["p_negative"]

    raw_engagement = df["likes"].clip(lower=0) + 2.0 * df["retweets"].clip(lower=0)
    df["raw_engagement"] = raw_engagement

    df["engagement_weight"] = 1.0 + np.log1p(raw_engagement)
    df["engagement_weight"] = df["engagement_weight"].clip(lower=1.0, upper=6.0)

    df["row_weight"] = df["aboutness"] * df["engagement_weight"]
    df["weighted_sentiment"] = df["sentiment_signal"] * df["row_weight"]

    return df


def aggregate_player_rankings(
    predictions: pd.DataFrame,
    prior_strength: float,
    min_effective_mentions: float,
) -> pd.DataFrame:
    rows = []

    for player, g in predictions.groupby("candidate_player", dropna=False):
        if not player:
            continue

        weighted_mentions = float(g["row_weight"].sum())
        effective_mentions = float(g["aboutness"].sum())
        raw_mentions = int(len(g))
        total_engagement = float(g["raw_engagement"].sum())

        if weighted_mentions > 0:
            avg_sentiment = float(g["weighted_sentiment"].sum() / weighted_mentions)
            pos_share = float((g["p_positive"] * g["row_weight"]).sum() / weighted_mentions)
            neg_share = float((g["p_negative"] * g["row_weight"]).sum() / weighted_mentions)
            neutral_share = float((g["p_neutral"] * g["row_weight"]).sum() / weighted_mentions)
            not_about_share = float((g["p_not_about_player"] * g["engagement_weight"]).sum() / g["engagement_weight"].sum())
        else:
            avg_sentiment = 0.0
            pos_share = 0.0
            neg_share = 0.0
            neutral_share = 0.0
            not_about_share = 1.0

        # Neutral prior = 0. This prevents low-volume players from ranking too high.
        sentiment_sum = float(g["weighted_sentiment"].sum())
        shrunk_sentiment = sentiment_sum / (weighted_mentions + prior_strength)

        rows.append(
            {
                "player_name": player,
                "player_id": g["player_id"].replace("", np.nan).dropna().iloc[0] if g["player_id"].replace("", np.nan).dropna().shape[0] else "",
                "position": g["position"].replace("", np.nan).dropna().iloc[0] if g["position"].replace("", np.nan).dropna().shape[0] else "",
                "raw_mentions": raw_mentions,
                "effective_mentions": effective_mentions,
                "weighted_mentions": weighted_mentions,
                "total_engagement": total_engagement,
                "avg_sentiment": avg_sentiment,
                "shrunk_sentiment": shrunk_sentiment,
                "positive_share": pos_share,
                "negative_share": neg_share,
                "neutral_share": neutral_share,
                "not_about_share": not_about_share,
                "avg_model_confidence": float(g["pred_confidence"].mean()),
                "pred_positive_count": int(g["pred_label"].eq("positive").sum()),
                "pred_negative_count": int(g["pred_label"].eq("negative").sum()),
                "pred_neutral_count": int(g["pred_label"].eq("neutral").sum()),
                "pred_not_about_player_count": int(g["pred_label"].eq("not_about_player").sum()),
            }
        )

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    max_effective_mentions = max(ranking["effective_mentions"].max(), 1.0)
    max_total_engagement = max(ranking["total_engagement"].max(), 1.0)

    ranking["sentiment_component"] = 50.0 + 50.0 * ranking["shrunk_sentiment"].clip(-1, 1)

    ranking["volume_component"] = (
        100.0
        * np.log1p(ranking["effective_mentions"])
        / np.log1p(max_effective_mentions)
    )

    ranking["engagement_component"] = (
        100.0
        * np.log1p(ranking["total_engagement"])
        / np.log1p(max_total_engagement)
    )

    ranking["power_score"] = (
        0.70 * ranking["sentiment_component"]
        + 0.20 * ranking["volume_component"]
        + 0.10 * ranking["engagement_component"]
    )

    ranking["criticism_score"] = (
        100.0
        * ranking["negative_share"]
        * (ranking["effective_mentions"] / (ranking["effective_mentions"] + prior_strength))
    )

    ranking["controversy_score"] = (
        100.0
        * (ranking["positive_share"] + ranking["negative_share"])
        * (1.0 - abs(ranking["positive_share"] - ranking["negative_share"]))
        * (ranking["effective_mentions"] / (ranking["effective_mentions"] + prior_strength))
    )

    ranking["ranking_eligible"] = ranking["effective_mentions"].ge(min_effective_mentions)

    ranking = ranking.sort_values(
        ["ranking_eligible", "power_score", "effective_mentions"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    ranking["power_rank"] = range(1, len(ranking) + 1)

    return ranking


def write_leaderboards(ranking: pd.DataFrame, output_path: Path, top_n: int):
    eligible = ranking[ranking["ranking_eligible"]].copy()

    def records(df):
        cols = [
            "power_rank",
            "player_name",
            "position",
            "power_score",
            "effective_mentions",
            "raw_mentions",
            "avg_sentiment",
            "positive_share",
            "negative_share",
            "neutral_share",
            "not_about_share",
            "total_engagement",
        ]
        cols = [c for c in cols if c in df.columns]
        return df[cols].head(top_n).to_dict(orient="records")

    leaderboards = {
        "metadata": {
            "description": "FanPulse player power rankings from World Cup tweet-player predictions.",
            "ranking_note": "This is a fan perception ranking, not an objective football performance ranking.",
            "top_n": top_n,
        },
        "top_power": records(eligible.sort_values("power_score", ascending=False)),
        "most_positive": records(eligible.sort_values("shrunk_sentiment", ascending=False)),
        "most_criticized": records(eligible.sort_values("criticism_score", ascending=False)),
        "most_discussed": records(eligible.sort_values("effective_mentions", ascending=False)),
        "most_controversial": records(eligible.sort_values("controversy_score", ascending=False)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(leaderboards, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/matched_tweets_long_for_manual_labeling.csv")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--wandb-artifact", default=None)
    parser.add_argument("--predictions-output", default="data/rankings/fanpulse_pair_predictions_v1.csv")
    parser.add_argument("--rankings-output", default="data/rankings/player_power_rankings_v1.csv")
    parser.add_argument("--leaderboards-output", default="data/rankings/player_leaderboards_v1.json")
    parser.add_argument("--top-report-output", default="reports/player_power_rankings_top20.csv")
    parser.add_argument("--prior-strength", type=float, default=30.0)
    parser.add_argument("--min-effective-mentions", type=float, default=20.0)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    raw = pd.read_csv(args.input, low_memory=False)
    candidates = normalize_candidates(raw)

    print(f"Loaded raw rows: {len(raw)}")
    print(f"Candidate tweet-player pairs after dedup: {len(candidates)}")

    model = load_model(args.model_path, args.wandb_artifact)

    predictions = add_predictions(candidates, model)

    ranking = aggregate_player_rankings(
        predictions=predictions,
        prior_strength=args.prior_strength,
        min_effective_mentions=args.min_effective_mentions,
    )

    predictions_output = Path(args.predictions_output)
    rankings_output = Path(args.rankings_output)
    leaderboards_output = Path(args.leaderboards_output)
    top_report_output = Path(args.top_report_output)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    rankings_output.parent.mkdir(parents=True, exist_ok=True)
    top_report_output.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_output, index=False)
    ranking.to_csv(rankings_output, index=False)

    write_leaderboards(ranking, leaderboards_output, args.top_n)

    top_report = ranking[ranking["ranking_eligible"]].head(args.top_n).copy()
    top_report.to_csv(top_report_output, index=False)

    print(f"\nSaved predictions: {predictions_output}")
    print(f"Saved rankings: {rankings_output}")
    print(f"Saved leaderboards: {leaderboards_output}")
    print(f"Saved top report: {top_report_output}")

    print("\nTop FanPulse Power Ranking:")
    display_cols = [
        "power_rank",
        "player_name",
        "position",
        "power_score",
        "effective_mentions",
        "avg_sentiment",
        "positive_share",
        "negative_share",
        "neutral_share",
        "not_about_share",
        "total_engagement",
    ]
    display_cols = [c for c in display_cols if c in ranking.columns]

    print(
        ranking[ranking["ranking_eligible"]]
        .head(args.top_n)[display_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()