import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import wandb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.training.train import (
    build_model,
    build_targets,
    load_config,
    make_model_input,
    save_json,
)


def main(config_path: Path):
    config = load_config(config_path)

    df = pd.read_csv(config["data_path"])
    df = df.dropna(subset=["text", "candidate_player", "is_about_player", "impression"]).copy()

    if "label_source" not in df.columns:
        raise ValueError("Augmented training requires label_source column.")

    df["target"] = build_targets(df, config["target_mode"])
    df["model_input"] = make_model_input(df)

    holdout_source = config.get("holdout_source", "manual_v2")
    augmentation_sources = set(config.get("augmentation_sources", ["llm_assisted_reviewed"]))

    manual = df[df["label_source"].eq(holdout_source)].copy()
    augment = df[df["label_source"].isin(augmentation_sources)].copy()

    if manual.empty:
        raise ValueError(f"No rows found for holdout_source={holdout_source}")

    manual_train_idx, manual_val_idx = train_test_split(
        manual.index,
        test_size=config.get("test_size", 0.25),
        random_state=config.get("random_state", 42),
        stratify=manual["target"],
    )

    train_idx = list(manual_train_idx) + list(augment.index)
    val_idx = list(manual_val_idx)

    train_df = df.loc[train_idx].copy()
    val_df = df.loc[val_idx].copy()

    X_train = train_df["model_input"]
    y_train = train_df["target"].astype(str)

    X_val = val_df["model_input"]
    y_val = val_df["target"].astype(str)

    model = build_model(config)

    run = wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
    )

    wandb.config.update(
        {
            "n_rows": len(df),
            "n_manual": len(manual),
            "n_augmentation": len(augment),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "target_distribution_all": df["target"].value_counts().to_dict(),
            "target_distribution_train": y_train.value_counts().to_dict(),
            "target_distribution_val": y_val.value_counts().to_dict(),
            "label_source_distribution_train": train_df["label_source"].value_counts().to_dict(),
            "validation_source": holdout_source,
            "split_strategy": "manual_holdout_plus_augmented_train",
        },
        allow_val_change=True,
    )

    if config.get("use_sample_weight", False) and "sample_weight" in train_df.columns:
        sample_weight = pd.to_numeric(train_df["sample_weight"], errors="coerce").fillna(1.0)
        model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_val)).astype(str)
    pred_distribution = y_pred.value_counts().to_dict()
    pred_distribution_ratio = y_pred.value_counts(normalize=True).to_dict()

    print("\nPrediction distribution:")
    print(y_pred.value_counts())

    print("\nPrediction distribution ratio:")
    print(y_pred.value_counts(normalize=True))
    y_val_str = y_val.astype(str)

    labels = sorted([str(label) for label in df["target"].unique().tolist()])

    metrics = {
        "accuracy": accuracy_score(y_val_str, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val_str, y_pred),
        "f1_macro": f1_score(y_val_str, y_pred, average="macro"),
        "f1_weighted": f1_score(y_val_str, y_pred, average="weighted"),
    }

    report = classification_report(
        y_val_str,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_val_str, y_pred, labels=labels)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nClassification report:")
    print(classification_report(y_val_str, y_pred, labels=labels, zero_division=0))

    print("\nConfusion matrix labels:")
    print(labels)
    print(cm)

    metrics.update({
        "pred_positive_count": int((y_pred == "positive").sum()),
        "pred_negative_count": int((y_pred == "negative").sum()),
        "pred_neutral_count": int((y_pred == "neutral").sum()),
        "pred_not_about_player_count": int((y_pred == "not_about_player").sum()),
        "pred_positive_rate": float((y_pred == "positive").mean()),
        "pred_negative_rate": float((y_pred == "negative").mean()),
        "pred_neutral_rate": float((y_pred == "neutral").mean()),
        "pred_not_about_player_rate": float((y_pred == "not_about_player").mean()),
    })

    wandb.log(metrics)

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y_val_ids = [label_to_id[str(label)] for label in y_val_str.tolist()]
    y_pred_ids = [label_to_id[str(label)] for label in y_pred.tolist()]

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
            "split_strategy": "manual_holdout_plus_augmented_train",
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
            "training_dataset": config["data_path"],
            "validation_source": holdout_source,
            **metrics,
        },
    )

    artifact.add_file(str(model_path))
    artifact.add_file(str(labels_path))
    artifact.add_file(str(metrics_path))
    artifact.add_file(str(report_path))

    run.log_artifact(
        artifact,
        aliases=config.get("register_aliases", ["latest"]),
    )

    run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)