import argparse
from pathlib import Path

import pandas as pd


BASE_COLUMNS = [
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
]


ALLOWED_IS_ABOUT = {"yes", "no", "unclear"}

ALLOWED_IMPRESSIONS = {
    "strong_positive",
    "positive",
    "neutral",
    "negative",
    "strong_negative",
    "not_about_player",
}

ALLOWED_REASONS = {
    "performance",
    "goal_or_assist",
    "mistake",
    "hype",
    "comparison",
    "meme_or_joke",
    "news_context",
    "not_about_player",
    "other",
}


def clean_text_value(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def read_review_csv(path: str | Path) -> pd.DataFrame:
    """
    Numbers exported this file with semicolon separator.
    We read everything as string to preserve multiline tweets, tweet ids,
    decimal commas, booleans, and labels.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # Drop Numbers trailing columns like "Unnamed: 37" and empty column names.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.loc[:, df.columns != ""]

    return df


def normalize_status(x) -> str:
    """
    User decision:
    - explicit 'skip' rows stay skipped;
    - everything else is considered verified.
    """
    x = clean_text_value(x).lower()

    if x == "skip":
        return "skip"

    return "verified"


def normalize_is_about(x) -> str:
    x = clean_text_value(x).lower()

    if x in ALLOWED_IS_ABOUT:
        return x

    return "unclear"


def normalize_impression(x) -> str:
    x = clean_text_value(x).lower()

    if x in ALLOWED_IMPRESSIONS:
        return x

    return "neutral"


def normalize_reason(x) -> str:
    """
    Reason may be a single label or a pipe-separated list:
    performance|goal_or_assist|comparison
    """
    x = clean_text_value(x).lower()

    if not x:
        return "other"

    parts = [p.strip() for p in x.split("|") if p.strip()]
    parts = [p for p in parts if p in ALLOWED_REASONS]

    if not parts:
        return "other"

    # Preserve order while removing duplicates.
    seen = set()
    cleaned = []
    for p in parts:
        if p not in seen:
            cleaned.append(p)
            seen.add(p)

    return "|".join(cleaned)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    return df


def to_bool_string(value: bool) -> bool:
    return bool(value)


def build_review_corrected(review: pd.DataFrame) -> pd.DataFrame:
    required_review_cols = [
        "text",
        "candidate_player",
        "final_is_about_player",
        "final_impression",
        "final_reason",
        "review_status",
    ]

    missing = [c for c in required_review_cols if c not in review.columns]
    if missing:
        raise ValueError(f"Missing required columns in review file: {missing}")

    review = review.copy()

    review["review_status"] = review["review_status"].map(normalize_status)
    review["final_is_about_player"] = review["final_is_about_player"].map(normalize_is_about)
    review["final_impression"] = review["final_impression"].map(normalize_impression)
    review["final_reason"] = review["final_reason"].map(normalize_reason)

    review["label_source"] = "llm_assisted_human_reviewed"
    review["label_is_verified"] = review["review_status"].eq("verified")

    return review


def build_new_dataset_rows(review_corrected: pd.DataFrame) -> pd.DataFrame:
    verified = review_corrected[review_corrected["review_status"].eq("verified")].copy()

    verified = ensure_columns(verified, BASE_COLUMNS)

    new_rows = pd.DataFrame()

    # Preserve metadata columns from review file.
    metadata_cols = [
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
    ]

    for col in metadata_cols:
        new_rows[col] = verified[col].map(clean_text_value)

    # Use reviewed final labels as canonical labels.
    new_rows["is_about_player"] = verified["final_is_about_player"]
    new_rows["impression"] = verified["final_impression"]
    new_rows["reason"] = verified["final_reason"]

    # Means annotation row is valid, not that the tweet is about the player.
    new_rows["label_is_valid"] = True

    new_rows["label_source"] = "llm_assisted_human_reviewed"
    new_rows["label_is_verified"] = True

    return new_rows


def normalize_base_dataset(base: pd.DataFrame) -> pd.DataFrame:
    base = ensure_columns(base, BASE_COLUMNS)

    base = base[BASE_COLUMNS].copy()
    base["label_source"] = "manual_v2"
    base["label_is_verified"] = True

    return base


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        default="data/processed/fanpulse_dataset_v2.csv",
        help="Base manually labeled v2 dataset.",
    )
    parser.add_argument(
        "--review",
        default="data/labeling/review/fanpulse_v3_review_prevlabels_prelabeled_exported.csv",
        help="CSV exported from Numbers with final_* columns.",
    )
    parser.add_argument(
        "--out-review",
        default="data/labeling/review/fanpulse_v3_review_corrected.csv",
        help="Clean normalized review CSV.",
    )
    parser.add_argument(
        "--out-dataset",
        default="data/processed/fanpulse_dataset_v3.csv",
        help="Final v3 training dataset.",
    )

    args = parser.parse_args()

    base = pd.read_csv(
        args.base,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    review_raw = read_review_csv(args.review)
    review_corrected = build_review_corrected(review_raw)

    out_review = Path(args.out_review)
    out_review.parent.mkdir(parents=True, exist_ok=True)
    review_corrected.to_csv(out_review, index=False)

    base_norm = normalize_base_dataset(base)
    new_rows = build_new_dataset_rows(review_corrected)

    combined = pd.concat(
        [
            base_norm[BASE_COLUMNS + ["label_source", "label_is_verified"]],
            new_rows[BASE_COLUMNS + ["label_source", "label_is_verified"]],
        ],
        ignore_index=True,
    )

    # Keep manual v2 if the exact same tweet-player pair appears in v3 review.
    before_dedup = len(combined)
    combined = (
        combined
        .drop_duplicates(subset=["text", "candidate_player"], keep="first")
        .reset_index(drop=True)
    )
    after_dedup = len(combined)

    out_dataset = Path(args.out_dataset)
    out_dataset.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dataset, index=False)

    verified_count = int(review_corrected["review_status"].eq("verified").sum())
    skip_count = int(review_corrected["review_status"].eq("skip").sum())

    print("=== Review build summary ===")
    print(f"Raw review rows:        {len(review_raw)}")
    print(f"Verified review rows:   {verified_count}")
    print(f"Skipped review rows:    {skip_count}")
    print(f"Base v2 rows:           {len(base_norm)}")
    print(f"New v3 rows:            {len(new_rows)}")
    print(f"Combined before dedup:  {before_dedup}")
    print(f"Combined after dedup:   {after_dedup}")
    print(f"Duplicates removed:     {before_dedup - after_dedup}")

    print("\n=== Raw review_status distribution ===")
    print(review_raw["review_status"].value_counts(dropna=False))

    print("\n=== Normalized review_status distribution ===")
    print(review_corrected["review_status"].value_counts(dropna=False))

    print("\n=== Dataset v3 label_source ===")
    print(combined["label_source"].value_counts(dropna=False))

    print("\n=== Dataset v3 is_about_player ===")
    print(combined["is_about_player"].value_counts(dropna=False))

    print("\n=== Dataset v3 impression ===")
    print(combined["impression"].value_counts(dropna=False))

    print("\nSaved:")
    print(f"- {out_review}")
    print(f"- {out_dataset}")


if __name__ == "__main__":
    main()