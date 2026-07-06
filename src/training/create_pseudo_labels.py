import argparse
from pathlib import Path

import joblib
import pandas as pd


def make_model_input(df: pd.DataFrame) -> pd.Series:
    candidate_player = df["candidate_player"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)

    position = (
        df["position"].fillna("").astype(str)
        if "position" in df.columns
        else ""
    )

    return (
        "candidate_player: " + candidate_player
        + " position: " + position
        + " tweet: " + text
    )


def normalize_unlabeled_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["text"] = df["tweet_text"].astype(str)
    out["candidate_player"] = df["player_name"].astype(str)
    out["player_id"] = df.get("player_id", "")
    out["position"] = df.get("position", "")
    out["source_file"] = df.get("source_file", "")
    out["tweet_key"] = df.get("tweet_key", "")
    out["matched_aliases"] = df.get("matched_aliases", "")
    out["match_method"] = df.get("match_methods", "")
    out["best_match_score"] = df.get("best_match_score", "")
    out["likes"] = df.get("like_count", 0)
    out["retweets"] = df.get("retweet_count", 0)

    return out


def label_to_annotation_fields(pred_label: str) -> tuple[str, str]:
    if pred_label == "not_about_player":
        return "no", "not_about_player"

    return "yes", pred_label


def create_pseudo_labels(
    manual_path: Path,
    unlabeled_path: Path,
    model_path: Path,
    output_path: Path,
    confidence_threshold: float,
    max_per_class: int,
    pseudo_weight: float,
) -> pd.DataFrame:
    manual = pd.read_csv(manual_path)
    manual = manual.copy()
    manual["label_source"] = "manual"
    manual["sample_weight"] = 1.0
    manual["pseudo_confidence"] = 1.0

    unlabeled_raw = pd.read_csv(unlabeled_path, low_memory=False)
    unlabeled = normalize_unlabeled_pool(unlabeled_raw)

    # remove rows that are already manually labeled if keys exist
    if "tweet_key" in manual.columns and "tweet_key" in unlabeled.columns:
        manual_keys = set(manual["tweet_key"].dropna().astype(str))
        if manual_keys:
            unlabeled = unlabeled[~unlabeled["tweet_key"].astype(str).isin(manual_keys)].copy()

    model = joblib.load(model_path)

    X_unlabeled = make_model_input(unlabeled)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba. Use LogisticRegression pipeline.")

    proba = model.predict_proba(X_unlabeled)
    classes = list(model.classes_)

    pred_ids = proba.argmax(axis=1)
    pred_labels = [classes[i] for i in pred_ids]
    pred_conf = proba.max(axis=1)

    unlabeled["pseudo_label"] = pred_labels
    unlabeled["pseudo_confidence"] = pred_conf

    pseudo = unlabeled[unlabeled["pseudo_confidence"] >= confidence_threshold].copy()

    if pseudo.empty:
        raise ValueError("No pseudo-labels selected. Lower confidence threshold or check model.")

    # cap per class to avoid positive class dominating everything
    pseudo = (
        pseudo.sort_values("pseudo_confidence", ascending=False)
        .groupby("pseudo_label", group_keys=False)
        .head(max_per_class)
        .reset_index(drop=True)
    )

    annotation_fields = pseudo["pseudo_label"].map(label_to_annotation_fields)
    pseudo["is_about_player"] = [x[0] for x in annotation_fields]
    pseudo["impression"] = [x[1] for x in annotation_fields]
    pseudo["reason"] = "pseudo_label"
    pseudo["label_is_valid"] = pseudo["is_about_player"].eq("yes")
    pseudo["label_source"] = "pseudo"
    pseudo["sample_weight"] = pseudo_weight

    pseudo = pseudo[
        [
            "text",
            "candidate_player",
            "player_id",
            "position",
            "source_file",
            "tweet_key",
            "matched_aliases",
            "match_method",
            "best_match_score",
            "likes",
            "retweets",
            "is_about_player",
            "impression",
            "reason",
            "label_is_valid",
            "label_source",
            "sample_weight",
            "pseudo_confidence",
        ]
    ]

    manual_cols = list(pseudo.columns)
    for col in manual_cols:
        if col not in manual.columns:
            manual[col] = None

    manual = manual[manual_cols]

    combined = pd.concat([manual, pseudo], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print("Manual rows:", len(manual))
    print("Pseudo rows:", len(pseudo))
    print("Combined rows:", len(combined))
    print("\nPseudo distribution:")
    print(pseudo["impression"].value_counts(dropna=False))
    print("\nPseudo confidence:")
    print(pseudo["pseudo_confidence"].describe())
    print(f"\nSaved: {output_path}")

    return combined


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-path", default="data/processed/fanpulse_dataset_v2.csv", type=Path)
    parser.add_argument("--unlabeled-path", default="data/interim/matched_tweets_long_for_manual_labeling.csv", type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", default="data/processed/fanpulse_dataset_v2_pseudo.csv", type=Path)
    parser.add_argument("--confidence-threshold", default=0.90, type=float)
    parser.add_argument("--max-per-class", default=300, type=int)
    parser.add_argument("--pseudo-weight", default=0.25, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    create_pseudo_labels(
        manual_path=args.manual_path,
        unlabeled_path=args.unlabeled_path,
        model_path=args.model_path,
        output_path=args.output,
        confidence_threshold=args.confidence_threshold,
        max_per_class=args.max_per_class,
        pseudo_weight=args.pseudo_weight,
    )