# FanPulse World Cup — Machine Learning Development Report

## 1. Project Overview

**FanPulse World Cup** is a machine learning project for analyzing football Twitter/X reactions at the player level.

The final product is a **World Cup player power ranking** based on fan perception in tweets.

The project does not attempt to rank players by objective football performance. Instead, it estimates how football fans reacted to specific players in public World Cup tweets.

```text
Final product:
World Cup player FanPulse Power Rankings

Core ML task:
tweet text + candidate player → fan impression toward that player
```

This is different from standard tweet-level sentiment analysis. A single tweet may praise one player and criticize another.

Example:

```text
Mbappe was electric, but Dembele killed every attack.
```

A tweet-level sentiment model may classify this as mixed or neutral.

FanPulse requires player-level labels:

```text
Kylian Mbappé   → positive
Ousmane Dembélé → negative
```

---

## 2. Capstone Scope

This project implements a full machine learning development cycle:

```text
data collection
→ data cleaning
→ player mention matching
→ manual labeling
→ first supervised dataset
→ baseline training
→ experiment tracking
→ model comparison
→ model registry
→ LLM-assisted data expansion
→ human label correction
→ augmented training
→ class-bias mitigation
→ production model selection
→ player-level power ranking
```

The project is designed as:

- a capstone ML project;
- an MLOps course project;
- a portfolio project for technical interviews.

The main focus is not only the final model score, but the full practical ML workflow: data quality, labeling, iteration, experiment tracking, model registry, and product-level output.

---

## 3. Problem Statement

Football fans generate thousands of reactions during World Cup matches.

Sports media, fan communities, and football analysts may want to answer:

- Which players generated the strongest positive fan reaction?
- Which players were most criticized?
- Which players were most discussed?
- Which players were controversial?
- How does fan perception differ from raw mention volume?

A simple keyword count is not enough, because popular players such as Messi, Ronaldo, or Mbappé receive many mentions regardless of whether the reaction is positive or negative.

A simple sentiment classifier is also not enough, because the sentiment must be linked to a specific candidate player.

Therefore, the project uses a **tweet-player pair classification task**.

---

## 4. Data Sources

The project uses public football and FIFA World Cup 2022 tweet datasets, plus a player reference dataset.

Raw tweet datasets:

- [Tweets on Football World Cup 2022](https://www.kaggle.com/datasets/deepeshnigamdata/tweets-on-football-world-cup-2022)
- [FIFA World Cup 2022 Tweets](https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets)
- [FIFA WorldCup 2022 Tweets Dataset](https://www.kaggle.com/datasets/twtdata/fifa-worldcup-2022-tweets-dataset)

Player reference dataset:

- [World Cup 2022 Elite Players Image Dataset / List Of All Players Names](https://www.kaggle.com/datasets/peterkibuchi/world-cup-2022-elite-players-image-dataset?select=List+Of+All+Players+Names.csv)

Large raw files are versioned with DVC and are not committed directly to Git.

---

## 5. Data Lineage

The full data lineage:

```text
raw tweet files
→ unified English deduplicated tweets
→ curated player list
→ safe/contextual player aliases
→ candidate tweet-player pairs
→ Label Studio manual annotation
→ processed dataset v2
→ initial supervised models
→ active sampling from 57k matched tweet-player pairs
→ LLM-assisted pre-labeling
→ human review and correction
→ processed dataset v3
→ augmented model training
→ W&B model registry
→ prediction over full matched pool
→ FanPulse player power rankings
```

Main pipeline files:

```text
src/data/unify_raw_tweets.py
src/data/match_player_mentions.py
src/data/prepare_labeling_tasks.py
src/data/build_processed_dataset.py

src/labeling/sample_v3_candidates.py
src/labeling/llm_prelabel_v3.py
src/labeling/build_dataset_v3.py

src/training/train.py
src/training/train_augmented.py

src/ranking/build_power_rankings.py
```

---

## 6. Player Matching

The raw tweets are converted into candidate tweet-player pairs using deterministic player alias matching.

The matching system uses:

```text
safe aliases
contextual aliases
```

### Safe aliases

Safe aliases can be matched directly.

Examples:

```text
messi      → Lionel Messi
mbappe     → Kylian Mbappé
bellingham → Jude Bellingham
ronaldo    → Cristiano Ronaldo
```

### Contextual aliases

Contextual aliases are only accepted when supporting football context exists.

Example:

```text
walker → Kyle Walker
```

This is only reliable if the tweet contains context such as:

```text
England
Man City
France
Mbappé
World Cup
```

Broad fuzzy matching was intentionally avoided as a default strategy because it can create false positives, for example:

```text
cancel → Cancelo
```

The output of this stage is a large pool of matched tweet-player pairs.

```text
matched tweet-player pairs: ~57k
```

---

## 7. Labeling Schema

Each labeled row represents one tweet-player pair.

Input fields:

```text
text
candidate_player
player_id
position
matched_aliases
match_method
likes
retweets
```

Annotation fields:

```text
is_about_player
impression
reason
```

Allowed `is_about_player` values:

```text
yes
no
unclear
```

Allowed impression labels:

```text
strong_positive
positive
neutral
negative
strong_negative
not_about_player
```

Optional reason labels:

```text
performance
goal_or_assist
mistake
hype
comparison
meme_or_joke
news_context
not_about_player
other
```

---

## 8. Dataset Versions

### Dataset v2

Dataset v2 was created from manual Label Studio annotations.

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

Dataset v2 was small and positive-heavy, but it was fully manually labeled and used as the initial gold dataset.

---

### Dataset v3

Dataset v3 was created using an LLM-assisted, human-reviewed workflow.

The remaining matched tweet-player pool contained more than 57k candidate pairs. A subset was selected using active sampling:

- uncertain model predictions;
- predicted minority classes;
- multi-player tweets;
- high-engagement tweets;
- random control examples.

The selected examples were first pre-labeled using an assistant and then manually reviewed and corrected.

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

For model training, the labels are collapsed into a 4-class target.

```text
strong_positive  → positive
positive         → positive
neutral          → neutral
negative         → negative
strong_negative  → negative
not_about_player → not_about_player
```

Approximate 4-class distribution:

```text
positive            432
negative            252
neutral             200
not_about_player    113
```

This made the training set much healthier than the original 240-row dataset.

---

## 9. LLM-Assisted Labeling Audit

LLM pre-labeling was tested as a faster alternative to manual annotation.

However, the audit showed that LLM labels were not reliable enough to use directly as gold labels.

Comparison:

```text
assistant_* = assistant pre-labels
final_*     = human-corrected labels
```

Audit over 760 pre-labeled rows:

| Error type | Count | Share |
|---|---:|---:|
| Any change in is_about / impression / reason | 520 / 760 | 68.4% |
| Core label changed: is_about or impression | 399 / 760 | 52.5% |
| is_about_player changed | 91 / 760 | 12.0% |
| impression changed | 391 / 760 | 51.4% |
| reason changed | 384 / 760 | 50.5% |
| 6-class target changed | 396 / 760 | 52.1% |
| 4-class target changed | 368 / 760 | 48.4% |
| Reason-only changes | 121 / 760 | 15.9% |
| Major target errors | 371 / 760 | 48.8% |

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

This was one of the most important lessons in the project:

```text
LLM labels were useful as a draft, but not reliable as final labels.
```

Therefore, dataset v3 is treated as:

```text
LLM-assisted + human-reviewed
```

not as:

```text
automatically LLM-labeled
```

---

## 10. ML Task Formulation

Two formulations were tested.

### 6-class formulation

```text
strong_positive
positive
neutral
negative
strong_negative
not_about_player
```

This is closer to the original labeling schema, but it was unstable with the available data.

In dataset v2, rare classes were too small:

```text
strong_positive: 14
strong_negative: 7
```

The 6-class model performed poorly.

---

### 4-class formulation

```text
positive
neutral
negative
not_about_player
```

This became the main production formulation because it is more stable and better supported by the dataset size.

The selected production model uses this 4-class target.

---

## 11. Train / Validation Strategy

The project uses a conservative validation strategy.

For augmented v3 training:

```text
validation = manual_v2 holdout only
training   = manual_v2 train split + llm_assisted_human_reviewed rows
```

This avoids evaluating only on LLM-assisted examples and keeps the validation set grounded in the original manual labels.

Validation size:

```text
60 rows
```

Because the validation set is small and positive-heavy, model selection does not rely on accuracy alone.

Tracked metrics:

- accuracy;
- balanced accuracy;
- macro-F1;
- weighted-F1;
- per-class precision;
- per-class recall;
- confusion matrix;
- predicted label distribution;
- positive prediction rate.

This was necessary because early models achieved acceptable accuracy while still over-predicting the positive class.

---

## 12. Experiment Tracking

Experiments are tracked with Weights & Biases.

W&B project:

```text
fanpulse-worldcup-mlops
```

Each run logs:

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
- prediction distribution;
- trained model artifact.

Model artifact:

```text
fanpulse-impression-classifier
```

Production aliases:

```text
latest
best
production
```

This model artifact is later used by the ranking pipeline and will be used by the FastAPI inference service in the next MLOps stage.

---

## 13. Models Tested

### Dummy baseline

```text
DummyClassifier(strategy="most_frequent")
```

Purpose:

- establish a weak baseline;
- show why accuracy is misleading on imbalanced data.

---

### TF-IDF + Logistic Regression

Main model family:

```text
TF-IDF features + LogisticRegression
```

Tested variants:

- word n-grams only;
- character n-grams only;
- word + character n-grams;
- 4-class target;
- 6-class target;
- different regularization values;
- `class_weight=balanced`;
- custom anti-positive class weights.

The final selected model uses:

```text
word n-grams + character n-grams
LogisticRegression
4-class target
custom class weights
```

---

## 14. Main Experiment Results

| Run | Dataset | Target | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| dummy_4class | v2 | 4-class | 0.5167 | 0.2500 | 0.1703 | 0.3520 | majority-class baseline |
| tfidf_lr_4class_word | v2 | 4-class | 0.5167 | 0.3558 | 0.3686 | 0.4700 | word TF-IDF baseline |
| tfidf_lr_4class_word_char | v2 | 4-class | 0.5667 | 0.3927 | 0.4128 | 0.5193 | word + char improves |
| tfidf_lr_4class_word_char_c20 | v2 | 4-class | 0.5833 | 0.4008 | 0.4210 | 0.5312 | best v2 model |
| tfidf_lr_6class_word_char | v2 | 6-class | 0.3833 | 0.2010 | 0.1917 | 0.3605 | unstable due to rare classes |
| tfidf_lr_4class_word_char_c20_v3_human_reviewed | v3 | 4-class | 0.5833 | 0.4300 | 0.4548 | 0.5300 | v3 improves minority quality |
| tfidf_lr_4class_word_char_c40_v3_human_reviewed | v3 | 4-class | 0.6167 | 0.4461 | 0.4706 | 0.5527 | better accuracy but positive-biased |
| tfidf_lr_4class_word_char_c40_v3_cw_soft | v3 | 4-class | 0.6000 | 0.4636 | 0.4865 | 0.5589 | reduced positive bias |
| tfidf_lr_4class_word_char_c40_v3_cw_strong | v3 | 4-class | 0.6000 | 0.5310 | 0.5358 | 0.5889 | best production candidate |

---

## 15. Best Model

The selected production candidate is:

```text
tfidf_lr_4class_word_char_c40_v3_cw_strong
```

Configuration:

```text
features:
  TF-IDF word n-grams
  TF-IDF character n-grams

classifier:
  LogisticRegression

target:
  4-class fan impression

dataset:
  fanpulse_dataset_v3

C:
  4.0

manual class weights:
  positive:         0.30
  negative:         1.80
  neutral:          2.50
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

This model was selected because it achieved the best macro-F1 and balanced accuracy while reducing the earlier positive-class prediction bias.

---

## 16. Final Confusion Matrix

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

## 17. Improvement Over Baselines

The final model improved substantially over both the dummy baseline and the best v2 supervised model.

```text
Dummy baseline macro-F1:      0.1703
Best v2 model macro-F1:       0.4210
Best v3 model macro-F1:       0.5358
```

Balanced accuracy:

```text
Dummy baseline:               0.2500
Best v2 model:                0.4008
Best v3 model:                0.5310
```

This improvement came mainly from:

- expanding the dataset from 240 to 997 verified rows;
- correcting LLM pre-labeling errors;
- improving minority-class coverage;
- adding custom class weights to reduce positive-class bias.

---

## 18. FanPulse Player Power Rankings

The trained classifier is not the final product by itself.

The final product is a **FanPulse World Cup Player Power Ranking**.

The ranking is computed by applying the production classifier to the full matched tweet-player pool.

```text
input:
~57k matched tweet-player pairs

model:
fanpulse-impression-classifier:production

output:
player-level power rankings
```

This ranking measures fan perception, not objective football performance.

---

## 19. Ranking Methodology

For every tweet-player pair, the model predicts:

```text
p_positive
p_negative
p_neutral
p_not_about_player
```

Instead of using only hard predicted labels, the ranking uses probabilities.

### Row-level features

```text
aboutness = 1 - p_not_about_player

sentiment_signal = p_positive - p_negative

raw_engagement = likes + 2 * retweets

engagement_weight = 1 + log1p(raw_engagement)
engagement_weight is clipped to [1, 6]

row_weight = aboutness * engagement_weight

weighted_sentiment = sentiment_signal * row_weight
```

### Player-level aggregation

For each player:

```text
effective_mentions = sum(aboutness)

weighted_mentions = sum(row_weight)

avg_sentiment = sum(weighted_sentiment) / sum(row_weight)

positive_share = weighted average of p_positive

negative_share = weighted average of p_negative

neutral_share = weighted average of p_neutral

not_about_share = weighted average of p_not_about_player
```

### Bayesian shrinkage

A shrinkage term is used to avoid over-ranking low-volume players.

```text
shrunk_sentiment = sentiment_sum / (weighted_mentions + prior_strength)
```

Default:

```text
prior_strength = 30
```

This prevents a player with only a few positive tweets from ranking above players with many consistent positive mentions.

---

## 20. Final Power Score

The final power score combines:

```text
70% sentiment component
20% mention volume component
10% engagement component
```

Formula:

```text
power_score =
    0.70 * sentiment_component
  + 0.20 * volume_component
  + 0.10 * engagement_component
```

Where:

```text
sentiment_component = 50 + 50 * shrunk_sentiment

volume_component = normalized log effective_mentions

engagement_component = normalized log total_engagement
```

The ranking also computes additional scores:

```text
criticism_score
controversy_score
```

These are used for separate leaderboards.

---

## 21. Ranking Outputs

The ranking pipeline produces:

```text
data/rankings/fanpulse_pair_predictions_v1.csv
data/rankings/player_power_rankings_v1.csv
data/rankings/player_leaderboards_v1.json
reports/player_power_rankings_top20.csv
```

### `fanpulse_pair_predictions_v1.csv`

Contains row-level predictions for every tweet-player pair:

```text
candidate_player
text
p_positive
p_negative
p_neutral
p_not_about_player
pred_label
pred_confidence
aboutness
sentiment_signal
row_weight
weighted_sentiment
```

### `player_power_rankings_v1.csv`

Contains aggregated player-level ranking metrics:

```text
player_name
position
raw_mentions
effective_mentions
weighted_mentions
avg_sentiment
positive_share
negative_share
neutral_share
not_about_share
total_engagement
power_score
criticism_score
controversy_score
power_rank
```

### `player_leaderboards_v1.json`

Contains several leaderboard views:

```text
top_power
most_positive
most_criticized
most_discussed
most_controversial
```

---

## 22. Current FanPulse Power Ranking

Top 30 players from the current ranking:

| Rank | Player | Position | Power Score | Effective Mentions | Avg Sentiment |
|---:|---|---|---:|---:|---:|
| 1 | Kylian Mbappé | Forward | 78.31 | 10006.71 | 0.468 |
| 2 | Lionel Messi | Forward | 73.04 | 20775.27 | 0.230 |
| 3 | Neymar | Forward | 66.93 | 1437.97 | 0.305 |
| 4 | Jude Bellingham | Midfielder | 63.54 | 136.24 | 0.488 |
| 5 | Olivier Giroud | Forward | 62.70 | 857.00 | 0.245 |
| 6 | Marcus Rashford | Forward | 61.23 | 352.05 | 0.286 |
| 7 | Julián Álvarez | Forward | 59.91 | 1267.08 | 0.208 |
| 8 | Vinícius Júnior | Forward | 58.59 | 262.59 | 0.292 |
| 9 | Bruno Fernandes | Midfielder | 57.45 | 207.08 | 0.215 |
| 10 | Bukayo Saka | Forward | 57.25 | 339.23 | 0.112 |
| 11 | Karim Benzema | Forward | 54.47 | 382.20 | 0.053 |
| 12 | Enzo Fernández | Midfielder | 54.13 | 52.07 | 0.335 |
| 13 | Rafael Leão | Forward | 54.11 | 108.40 | 0.204 |
| 14 | Eden Hazard | Forward | 52.47 | 44.44 | 0.438 |
| 15 | Jack Grealish | Forward | 52.40 | 48.27 | 0.235 |
| 16 | Rodrigo De Paul | Midfielder | 52.37 | 48.80 | 0.262 |
| 17 | Luka Modrić | Midfielder | 52.04 | 465.14 | -0.052 |
| 18 | Harry Kane | Forward | 49.79 | 1121.31 | -0.213 |
| 19 | Sadio Mané | Forward | 47.40 | 32.81 | 0.062 |
| 20 | Casemiro | Midfielder | 46.25 | 38.82 | 0.194 |
| 21 | João Cancelo | Defender | 45.72 | 22.25 | 0.019 |
| 22 | Bernardo Silva | Midfielder | 45.11 | 22.46 | 0.251 |
| 23 | Son Heung-min | Forward | 44.44 | 31.52 | -0.263 |
| 24 | Cristiano Ronaldo | Forward | 44.05 | 10268.28 | -0.557 |
| 25 | Emiliano Martinez | Goalkeeper | 42.47 | 583.73 | -0.332 |
| 26 | Trent Alexander-Arnold | Defender | 42.08 | 29.40 | -0.041 |
| 27 | Kyle Walker | Defender | 41.74 | 34.15 | -0.174 |
| 28 | Alisson | Goalkeeper | 39.93 | 35.78 | -0.122 |
| 29 | Pedri | Midfielder | 38.98 | 41.67 | -0.327 |
| 30 | Robert Lewandowski | Forward | 38.62 | 68.91 | -0.304 |

The ranking is logically consistent with the data:

- Mbappé ranks first due to high positive sentiment and large mention volume.
- Messi ranks second due to extremely high discussion volume and positive fan perception.
- Ronaldo is heavily discussed but ranks lower because the model detects a strong negative fan signal.
- Kane, Son, Pedri, Lewandowski, and Emiliano Martinez receive lower scores due to stronger criticism signals.
- Bellingham ranks high despite lower volume because his sentiment signal is very strong, but shrinkage prevents him from overtaking high-volume stars.

---

## 23. How to Build Rankings

Using local model artifact:

```bash
PYTHONPATH=. python src/ranking/build_power_rankings.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --model-path artifacts/models/tfidf_lr_4class_word_char_c40_v3_cw_strong/model.joblib \
  --min-effective-mentions 20 \
  --prior-strength 30 \
  --top-n 20
```

Using W&B production artifact:

```bash
PYTHONPATH=. python src/ranking/build_power_rankings.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --wandb-artifact shinkaryovae-set-university/fanpulse-worldcup-mlops/fanpulse-impression-classifier:production \
  --min-effective-mentions 20 \
  --prior-strength 30 \
  --top-n 20
```

---

## 24. How to Train

Install dependencies:

```bash
pip install -r requirements.txt
```

Login to W&B:

```bash
wandb login
```

Train the production model:

```bash
PYTHONPATH=. python src/training/train_augmented.py \
  --config configs/train_tfidf_lr_4class_word_char_c40_v3_cw_strong.yaml
```

Train the earlier v2 baseline:

```bash
PYTHONPATH=. python src/training/train.py \
  --config configs/train_tfidf_lr_4class_word_char_c20.yaml
```

Run ranking generation:

```bash
PYTHONPATH=. python src/ranking/build_power_rankings.py \
  --input data/interim/matched_tweets_long_for_manual_labeling.csv \
  --model-path artifacts/models/tfidf_lr_4class_word_char_c40_v3_cw_strong/model.joblib
```

---

## 25. Model Registry

The best model is logged to W&B as:

```text
fanpulse-impression-classifier
```

Aliases:

```text
latest
best
production
```

The model registry is important because the next MLOps stage can load the production model without relying on a local file path.

Planned usage in HW3:

```text
FastAPI service
→ load W&B artifact fanpulse-impression-classifier:production
→ expose /predict endpoint
→ expose /rankings endpoint
```

---

## 26. Key Problems Encountered

### Problem 1: Small initial labeled dataset

Dataset v2 contained only 240 manually labeled rows.

This was enough for baseline experiments but not enough for a strong player-level classifier.

Solution:

```text
LLM-assisted active sampling + human review → dataset v3 with 997 verified rows
```

---

### Problem 2: Positive-class bias

Early models over-predicted positive sentiment.

This happened because:

- the initial dataset was positive-heavy;
- football Twitter often praises star players;
- high-profile players receive many positive mentions;
- minority classes had too few examples;
- TF-IDF models rely on lexical patterns.

Solution:

```text
dataset v3 expansion
+ human correction
+ prediction distribution tracking
+ custom anti-positive class weights
```

---

### Problem 3: LLM pre-labeling quality

LLM pre-labeling was much noisier than expected.

Almost half of the 4-class targets changed after human review.

Solution:

```text
LLM output used only as draft labels
final labels always human-reviewed
label_source stored in dataset
```

---

### Problem 4: `not_about_player` is structurally different

`not_about_player` is not a sentiment class. It means the candidate player is not actually the subject of the tweet.

Solution used now:

```text
include not_about_player as a 4th class
use custom class weights
track not_about_player recall
```

Future solution:

```text
two-stage classifier:
Stage 1: about_player vs not_about_player
Stage 2: positive vs neutral vs negative
```

---

### Problem 5: Small validation set

The validation set contains only 60 manual rows.

One changed prediction affects accuracy by 1.67 percentage points.

Solution:

```text
use macro-F1
use balanced accuracy
inspect confusion matrix
track prediction distribution
avoid accuracy-only model selection
```

---

### Problem 6: Ranking must not be pure volume

A ranking based only on positive tweet count would be dominated by the most famous players.

Solution:

```text
combine sentiment, volume, engagement, and Bayesian shrinkage
```

This gives a more stable player power ranking.

---

## 27. Why Not Fine-Tune BERT?

Transformer fine-tuning was intentionally avoided at this stage.

Reasons:

- only 240 manual labels were available at the start;
- even v3 has fewer than 1,000 verified rows;
- transformer fine-tuning would increase overfitting risk;
- TF-IDF + Logistic Regression is fast, interpretable, and easy to deploy;
- the MLOps goal is to build a full lifecycle, not only maximize model complexity;
- the model must be easy to package for FastAPI inference in HW3.

A transformer or embedding-based classifier is a future improvement after collecting more verified data.

---

## 28. Future Improvements

Planned improvements:

1. **Cross-validation**
   - 5-fold validation on manual splits;
   - compare stability of model choices.

2. **Two-stage classification**
   - Stage 1: about-player detection;
   - Stage 2: positive / neutral / negative classification.

3. **More targeted labeling**
   - more `not_about_player` examples;
   - more hard negative examples;
   - more neutral factual tweets;
   - more multi-player tweets.

4. **Better active learning**
   - uncertainty sampling;
   - model disagreement;
   - high-engagement examples;
   - player-specific undercoverage sampling.

5. **Embedding-based model**
   - sentence embeddings;
   - small transformer;
   - linear classifier on top.

6. **Additional features**
   - alias type;
   - number of candidate players in tweet;
   - player position;
   - engagement features;
   - source dataset features.

7. **Ranking calibration**
   - tune prior strength;
   - compare ranking stability;
   - add confidence intervals;
   - evaluate against external expert rankings.

8. **Production monitoring**
   - prediction distribution drift;
   - positive-rate drift;
   - player mention drift;
   - ranking drift;
   - latency and error rate.

---

## 29. Project Summary

FanPulse started as a small manually labeled tweet-player classification project and evolved into a full ML product pipeline.

Main achievements:

```text
raw public football tweets processed
57k+ candidate tweet-player pairs generated
240 manual Label Studio labels created
first supervised models trained and tracked
LLM-assisted labeling tested and audited
757 additional rows human-corrected
dataset v3 expanded to 997 verified rows
best macro-F1 improved from 0.4210 to 0.5358
production model registered in W&B
player-level FanPulse Power Rankings generated
```

The final model is not perfect, but it is a strong production candidate for the next MLOps stage.

The final product is not just a classifier. It is an end-to-end system that turns noisy football tweets into interpretable player-level fan perception rankings.