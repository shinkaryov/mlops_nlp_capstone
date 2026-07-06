import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.training.train import build_targets, make_model_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/fanpulse_dataset_v3.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="data/labeling/review/fanpulse_v3_validation_error_review.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "candidate_player", "is_about_player", "impression"]).copy()

    df["target"] = build_targets(df, "4class")
    df["model_input"] = make_model_input(df)

    manual = df[df["label_source"].eq("manual_v2")].copy()

    _, val_idx = train_test_split(
        manual.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=manual["target"],
    )

    val = df.loc[val_idx].copy()

    model = joblib.load(args.model_path)

    pred = model.predict(val["model_input"])
    proba = model.predict_proba(val["model_input"])
    classes = list(model.classes_)

    val["pred"] = pred
    val["is_correct"] = val["target"].astype(str).eq(val["pred"].astype(str))

    for i, cls in enumerate(classes):
        val[f"p_{cls}"] = proba[:, i]

    val["max_proba"] = proba.max(axis=1)

    val["review_correct_label"] = ""
    val["review_comment"] = ""

    cols = [
        "is_correct",
        "target",
        "pred",
        "candidate_player",
        "is_about_player",
        "impression",
        "reason",
        "p_positive",
        "p_negative",
        "p_neutral",
        "p_not_about_player",
        "max_proba",
        "review_correct_label",
        "review_comment",
        "text",
    ]

    cols = [c for c in cols if c in val.columns]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    val[cols].sort_values(["is_correct", "target"]).to_csv(output, index=False)

    print(f"Validation rows: {len(val)}")
    print(f"Errors: {(~val['is_correct']).sum()}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()