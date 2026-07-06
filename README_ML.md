# FanPulse World Cup — Machine Learning Development

## 1. Project Goal

FanPulse World Cup is a machine learning project for **player-level fan impression classification** from football tweets.

The goal is to classify the fan impression toward a specific candidate player mentioned in a tweet.

This is different from standard tweet-level sentiment analysis. A single tweet can praise one player and criticize another.

Example:

```text
Mbappe was electric, but Dembele killed every attack.
```

A tweet-level sentiment model may classify this as mixed or neutral.  
The FanPulse task requires player-level labels:

```text
Kylian Mbappé  → positive
Ousmane Dembélé → negative
```

The final ML task is:

```text
Input:
tweet text + candidate player

Output:
fan impression toward the candidate player
```

---

## 2. Capstone Scope

This project implements a full machine learning development cycle:

```text
data collection
→ data cleaning
→ player mention matching
→ manual labeling
→ experiment tracking
→ model training
→ model comparison
→ model registry
→ production candidate selection
```

The project is designed as both:

- a capstone ML project;
- an MLOps course project;
- a portfolio project for technical interviews.

---

## 3. Data Sources

The raw data comes from public football and FIFA World Cup 2022 tweet datasets, plus a player reference dataset.

Raw tweet datasets:

- Tweets on Football World Cup 2022
- FIFA World Cup 2022 Tweets
- FIFA WorldCup 2022 Tweets Dataset

Player reference dataset:

- World Cup 2022 Elite Players / player names dataset

The raw files are versioned with DVC and are not committed directly to Git.

---

## 4. Data Lineage

The data pipeline follows this lineage:

```text
raw football tweet files
→ unified English deduplicated tweets
→ player alias matching
→ candidate tweet-player pairs
→ manual Label Studio annotation
→ processed dataset v2
→ first supervised models
→ active sampling from unlabeled matched pool
→ LLM-assisted pre-labeling
→ human review and correction
→ processed dataset v3
→ augmented training
→ best production model
```

---

## 5. Dataset Versions

### Dataset v2

Dataset v2 was built from manual Label Studio annotations.

```text
Rows: 240
```

4-class target distribution:

```text
positive            123
negative             51
not_about_player     47
neutral              19
```

This dataset was small and positive-heavy, but it was fully manually labeled and used as the initial gold dataset.

### Dataset v3

Dataset v3 was created using an LLM-assisted, human-reviewed workflow.

The remaining matched tweet-player pool contained more than 50k candidate pairs. A subset was selected using active sampling:

- uncertain model predictions;
- predicted minority classes;
- multi-player tweets;
- high-engagement tweets;
- random control examples.

The selected examples were pre-labeled using an assistant and then manually corrected.

Final dataset v3:

```text
Rows: 997
manual_v2 rows: 240
llm_assisted_human_reviewed rows: 757
```

Raw final label distribution:

```text
positive            370
negative            228
neutral             200
not_about_player    113
strong_positive      62
strong_negative      24
```

For model training, labels are collapsed into a 4-class target:

```text
strong_positive → positive
positive        → positive
neutral         → neutral
negative        → negative
strong_negative → negative
not_about_player → not_about_player
```

This makes the task more stable with the available dataset size.

---

## 6. LLM-Assisted Labeling Audit

LLM pre-labeling was tested as a faster alternative to manual annotation.

However, the audit showed that LLM labels were not reliable enough to use directly as gold labels.

Comparing assistant pre-labels to the final human-reviewed labels over 760 rows:

| Error type | Count | Share |
|---|---:|---:|
| Any change in is_about / impression / reason | 520 / 760 | 68.4% |
| Core label changed: is_about or impression | 399 / 760 | 52.5% |
| is_about_player changed | 91 / 760 | 12.0% |
| impression changed | 391 / 760 | 51.4% |
| reason changed | 384 / 760 | 50.5% |
| 6-class target changed | 396 / 760 | 52.1% |
| 4-class target changed | 368 / 760 | 48.4% |
| reason-only changes | 121 / 760 | 15.9% |
| major target errors | 371 / 760 | 48.8% |

The main observed issue was over-prediction of positive sentiment.

Assistant pre-label distribution:

```text
positive            346
neutral             208
strong_positive     104
negative             54
not_about_player     32
strong_negative      16
```

Final human-reviewed distribution:

```text
positive            259
negative            183
neutral             173
not_about_player     80
strong_positive      48
strong_negative      17
```

The assistant under-detected:

- negative examples;
- not_about_player cases;
- neutral factual tweets.

Therefore, LLM output was used only as a **pre-labeling draft**, not as training labels directly.

The final dataset v3 is treated as:

```text
LLM-assisted + human-reviewed
```

not as:

```text
automatically LLM-labeled
```

This was an important data quality lesson in the project.

---

## 7. ML Task Formulation

Two task formulations were tested.

### 6-class formulation

```text
strong_positive
positive
neutral
negative
strong_negative
not_about_player
```

This formulation is closer to the original labeling schema but unstable with limited data.

The rare classes were too small:

```text
strong_positive: 14 in v2
strong_negative: 7 in v2
```

### 4-class formulation

```text
positive
neutral
negative
not_about_player
```

This formulation was selected as the main production task because it is more stable and better supported by the data.

---

## 8. Train / Validation Strategy

The initial validation set is based on the manually labeled v2 data.

For augmented v3 training, the strategy is:

```text
validation = manual_v2 holdout only
training = manual_v2 train split + llm_assisted_human_reviewed rows
```

This avoids evaluating the model only on LLM-assisted examples and keeps the validation set grounded in the original manual labels.

The validation set is small:

```text
60 rows
```

For that reason, model selection is based not only on accuracy but also on:

- macro-F1;
- balanced accuracy;
- per-class precision/recall;
- confusion matrix;
- prediction distribution;
- positive prediction rate.

Accuracy alone is not reliable because the validation set is positive-heavy.

---

## 9. Experiment Tracking

Experiments are tracked with Weights & Biases.

Each training run logs:

- model type;
- dataset version;
- target mode;
- hyperparameters;
- train/validation split metadata;
- accuracy;
- balanced accuracy;
- macro-F1;
- weighted-F1;
- confusion matrix;
- predicted label distribution;
- model artifact.

W&B project:

```text
fanpulse-worldcup-mlops
```

Model artifact:

```text
fanpulse-impression-classifier
```

The production candidate is registered with aliases:

```text
latest
best
production
```

---

## 10. Models Tested

### Baseline

```text
DummyClassifier(strategy="most_frequent")
```

Purpose:

- establish a weak baseline;
- show why accuracy is misleading.

### TF-IDF + Logistic Regression

Several TF-IDF Logistic Regression models were tested:

- word n-grams only;
- character n-grams only;
- word + character n-grams;
- 4-class target;
- 6-class target;
- different regularization values;
- balanced class weights;
- manual anti-positive class weights.

The best feature setup was:

```text
TF-IDF word n-grams + character n-grams
LogisticRegression
4-class target
```

---

## 11. Main Experiment Results

| Run | Dataset | Target | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| dummy_4class | v2 | 4-class | 0.5167 | 0.2500 | 0.1703 | 0.3520 | predicts majority class |
| tfidf_lr_4class_word | v2 | 4-class | 0.5167 | 0.3558 | 0.3686 | 0.4700 | word TF-IDF baseline |
| tfidf_lr_4class_word_char | v2 | 4-class | 0.5667 | 0.3927 | 0.4128 | 0.5193 | word + char improves |
| tfidf_lr_4class_word_char_c20 | v2 | 4-class | 0.5833 | 0.4008 | 0.4210 | 0.5312 | best v2 model |
| tfidf_lr_6class_word_char | v2 | 6-class | 0.3833 | 0.2010 | 0.1917 | 0.3605 | unstable due to rare classes |
| tfidf_lr_4class_word_char_c20_v3_human_reviewed | v3 | 4-class | 0.5833 | 0.4300 | 0.4548 | 0.5300 | v3 improves minority quality |
| tfidf_lr_4class_word_char_c40_v3_human_reviewed | v3 | 4-class | 0.6167 | 0.4461 | 0.4706 | 0.5527 | better accuracy but still positive-biased |
| tfidf_lr_4class_word_char_c40_v3_cw_soft | v3 | 4-class | 0.6000 | 0.4636 | 0.4865 | 0.5589 | reduced positive bias |
| tfidf_lr_4class_word_char_c40_v3_cw_strong | v3 | 4-class | 0.6000 | 0.5310 | 0.5358 | 0.5889 | best production candidate |

---

## 12. Best Model

The selected production candidate is:

```text
tfidf_lr_4class_word_char_c40_v3_cw_strong
```

Configuration:

```text
features: TF-IDF word n-grams + character n-grams
classifier: LogisticRegression
target: 4-class
dataset: fanpulse_dataset_v3
C: 4.0
manual class weights:
  positive: 0.30
  negative: 1.80
  neutral: 2.50
  not_about_player: 3.00
```

Validation results:

```text
accuracy:          0.6000
balanced_accuracy: 0.5310
macro-F1:          0.5358
weighted-F1:       0.5889
```

Prediction distribution on validation:

```text
positive            35 / 60
negative            11 / 60
not_about_player     9 / 60
neutral              5 / 60
```

This model was selected because it achieved the best macro-F1 and balanced accuracy while reducing the previous positive-class prediction bias.

---

## 13. Final Confusion Matrix

Labels:

```text
negative
neutral
not_about_player
positive
```

Confusion matrix:

```text
[[ 5  1  0  6]
 [ 1  3  0  1]
 [ 3  1  4  4]
 [ 2  0  5 24]]
```

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.45 | 0.42 | 0.43 |
| neutral | 0.60 | 0.60 | 0.60 |
| not_about_player | 0.44 | 0.33 | 0.38 |
| positive | 0.69 | 0.77 | 0.73 |

The model still performs best on positive examples, but it no longer collapses into positive-only predictions.

---

## 14. Key Problems Encountered

### Problem 1: Small initial labeled dataset

The first manually labeled dataset contained only 240 rows.

This was enough for a baseline model but not enough for a high-quality semantic classifier.

### Problem 2: Positive-class bias

The first models over-predicted positive sentiment.

This happened because:

- the initial dataset was positive-heavy;
- football tweets often praise star players;
- TF-IDF models rely on surface-level lexical patterns;
- minority classes had limited examples.

This was partially fixed by:

- dataset v3 expansion;
- active sampling;
- human correction of LLM pre-labels;
- manual anti-positive class weights.

### Problem 3: LLM pre-labeling quality

LLM pre-labeling was much noisier than expected.

Almost half of the 4-class targets changed after human review.

This showed that automatic LLM labeling was not reliable enough for this task without human verification.

### Problem 4: `not_about_player` is not sentiment

The `not_about_player` class is structurally different from positive, neutral, and negative.

It represents candidate matching failure or weak player relevance, not sentiment.

A future improvement is to split the model into two stages:

```text
Stage 1: about_player vs not_about_player
Stage 2: positive vs neutral vs negative
```

### Problem 5: Small validation set

The manual validation set contains only 60 rows.

One changed prediction affects accuracy by 1.67 percentage points.

Therefore, model quality must be interpreted using several metrics, not accuracy alone.

---

## 15. Why Not Fine-Tune BERT?

Transformer fine-tuning was intentionally avoided at this stage.

Reasons:

- only 240 original manual labels were available at the start;
- even v3 has fewer than 1,000 verified examples;
- training a transformer would increase complexity and overfitting risk;
- the MLOps goal of HW2 was experiment tracking and model registry, not maximizing leaderboard score;
- TF-IDF + Logistic Regression is fast, interpretable, easy to package, and suitable for the next FastAPI inference stage.

A transformer or embedding-based model is a possible future improvement after collecting more labels.

---

## 16. Future Improvements

Planned improvements:

1. 5-fold cross-validation on manual holdout splits.
2. Two-stage classifier:
   - about-player detection;
   - impression classification.
3. More manual labels for `not_about_player`, `neutral`, and hard negative examples.
4. Better active learning:
   - uncertainty sampling;
   - model disagreement;
   - high-impact player examples.
5. Embedding-based classifier:
   - sentence embeddings;
   - small transformer;
   - linear classifier on top.
6. Better player context features:
   - alias type;
   - number of candidate players in tweet;
   - player position;
   - engagement features.
7. Production monitoring:
   - prediction distribution drift;
   - positive-rate drift;
   - player mention drift;
   - latency and error rate.

---

## 17. How to Train

Install dependencies:

```bash
pip install -r requirements.txt
```

Login to W&B:

```bash
wandb login
```

Train the current production candidate:

```bash
PYTHONPATH=. python src/training/train_augmented.py \
  --config configs/train_tfidf_lr_4class_word_char_c40_v3_cw_strong.yaml
```

Train the earlier supervised v2 baseline:

```bash
PYTHONPATH=. python src/training/train.py \
  --config configs/train_tfidf_lr_4class_word_char_c20.yaml
```

---

## 18. Model Registry

The best model is logged to W&B as:

```text
fanpulse-impression-classifier
```

Production aliases:

```text
latest
best
production
```

This artifact will be loaded by the inference service in the next MLOps stage.

---

## 19. Summary

The project started with a small manually labeled dataset and a simple baseline.

Through data-centric iteration, the dataset was expanded from 240 to 997 verified examples, LLM-assisted labels were audited and corrected, and model performance improved substantially.

The final selected model improves over the initial dummy baseline and the first supervised model:

```text
Dummy baseline macro-F1:      0.1703
Best v2 model macro-F1:       0.4210
Best v3 model macro-F1:       0.5358
```

The final model is not perfect, but it is a solid production candidate for the next MLOps stage: FastAPI inference and monitoring.