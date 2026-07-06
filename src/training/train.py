import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
import wandb

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def stable_hash(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:20]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_targets(df: pd.DataFrame, target_mode: str) -> pd.Series:
    def target_6(row):
        if row["is_about_player"] != "yes":
            return "not_about_player"
        return row["impression"]

    y6 = df.apply(target_6, axis=1)

    if target_mode == "6class":
        return y6

    if target_mode == "4class":
        mapping = {
            "strong_positive": "positive",
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative",
            "strong_negative": "negative",
            "not_about_player": "not_about_player",
        }
        return y6.map(mapping).fillna("not_about_player")

    raise ValueError(f"Unsupported target_mode: {target_mode}")


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


def make_group_key(df: pd.DataFrame) -> pd.Series:
    if "tweet_key" in df.columns and df["tweet_key"].notna().any():
        return df["tweet_key"].fillna(
            df.apply(lambda r: stable_hash(str(r["text"]) + "|" + str(r["candidate_player"])), axis=1)
        )

    return df.apply(
        lambda r: stable_hash(str(r["text"]) + "|" + str(r["candidate_player"])),
        axis=1,
    )


def build_model(config: dict):
    model_type = config["model_type"]

    if model_type == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if model_type != "tfidf_logreg":
        raise ValueError(f"Unsupported model_type: {model_type}")

    feature_mode = config.get("feature_mode", "word")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(config.get("word_ngram_min", 1), config.get("word_ngram_max", 2)),
        max_features=config.get("max_features", 10000),
        lowercase=True,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(config.get("char_ngram_min", 3), config.get("char_ngram_max", 5)),
        max_features=config.get("max_features", 10000),
        lowercase=True,
        sublinear_tf=True,
    )

    if feature_mode == "word":
        vectorizer = word_vectorizer
    elif feature_mode == "char":
        vectorizer = char_vectorizer
    elif feature_mode == "word_char":
        vectorizer = FeatureUnion(
            [
                ("word_tfidf", word_vectorizer),
                ("char_tfidf", char_vectorizer),
            ]
        )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    classifier = LogisticRegression(
        C=config.get("C", 1.0),
        class_weight=config.get("class_weight", "balanced"),
        max_iter=config.get("max_iter", 2000),
        random_state=config.get("random_state", 42),
        n_jobs=None,
    )

    return Pipeline(
        [
            ("features", vectorizer),
            ("classifier", classifier),
        ]
    )


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main(config_path: Path):
    config = load_config(config_path)

    df = pd.read_csv(config["data_path"])
    df = df.dropna(subset=["text", "candidate_player", "is_about_player", "impression"]).copy()

    df["target"] = build_targets(df, config["target_mode"])
    df["model_input"] = make_model_input(df)
    df["group_key"] = make_group_key(df)

    # In the current dataset every text is unique, but this keeps future multi-player tweets safe.
    X = df["model_input"]
    y = df["target"]

    # If every group is unique, stratified split is safe and gives better class balance.
    if df["group_key"].nunique() == len(df):
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            X,
            y,
            df.index,
            test_size=config.get("test_size", 0.25),
            random_state=config.get("random_state", 42),
            stratify=y,
        )
        split_strategy = "stratified_row_split_unique_groups"
    else:
        # Fallback: still split by group-like key to avoid leakage.
        unique_groups = df[["group_key", "target"]].drop_duplicates("group_key")
        train_groups, val_groups = train_test_split(
            unique_groups["group_key"],
            test_size=config.get("test_size", 0.25),
            random_state=config.get("random_state", 42),
            stratify=unique_groups["target"],
        )

        train_mask = df["group_key"].isin(train_groups)
        val_mask = df["group_key"].isin(val_groups)

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        train_idx, val_idx = df.index[train_mask], df.index[val_mask]
        split_strategy = "group_split"

    sample_weight = None
    if "sample_weight" in df.columns:
        sample_weight = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0)

    model = build_model(config)

    run = wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
    )

    wandb.config.update(
        {
            "n_rows": len(df),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "target_distribution": y.value_counts().to_dict(),
            "train_distribution": y_train.value_counts().to_dict(),
            "val_distribution": y_val.value_counts().to_dict(),
            "split_strategy": split_strategy,
        },
        allow_val_change=True,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    y_val_str = y_val.astype(str)
    y_pred_str = pd.Series(y_pred).astype(str)

    labels = sorted([str(label) for label in y.unique().tolist()])

    metrics = {
        "accuracy": accuracy_score(y_val_str, y_pred_str),
        "balanced_accuracy": balanced_accuracy_score(y_val_str, y_pred_str),
        "f1_macro": f1_score(y_val_str, y_pred_str, average="macro"),
        "f1_weighted": f1_score(y_val_str, y_pred_str, average="weighted"),
    }

    report = classification_report(
        y_val_str,
        y_pred_str,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_val_str, y_pred_str, labels=labels)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nClassification report:")
    print(classification_report(y_val_str, y_pred_str, labels=labels, zero_division=0))

    print("\nConfusion matrix labels:")
    print(labels)
    print(cm)

    wandb.log(metrics)

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y_val_ids = [label_to_id[str(label)] for label in y_val_str.tolist()]
    y_pred_ids = [label_to_id[str(label)] for label in y_pred_str.tolist()]

    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_val_ids,
                preds=y_pred_ids,
                class_names=labels,
            )
        }
    )

    output_dir = Path("artifacts/models") / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    labels_path = output_dir / "labels.json"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.json"
    split_path = output_dir / "split_indices.json"

    joblib.dump(model, model_path)
    save_json(labels, labels_path)
    save_json(metrics, metrics_path)
    save_json(report, report_path)
    save_json(
        {
            "train_indices": [int(x) for x in train_idx],
            "val_indices": [int(x) for x in val_idx],
            "split_strategy": split_strategy,
        },
        split_path,
    )

    artifact = wandb.Artifact(
        name=config["artifact_name"],
        type="model",
        metadata={
            "run_name": config["run_name"],
            "target_mode": config["target_mode"],
            "model_type": config["model_type"],
            "feature_mode": config.get("feature_mode"),
            **metrics,
        },
    )

    artifact.add_file(str(model_path))
    artifact.add_file(str(labels_path))
    artifact.add_file(str(metrics_path))
    artifact.add_file(str(report_path))

    aliases = config.get("register_aliases", ["latest"])
    run.log_artifact(artifact, aliases=aliases)

    run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)