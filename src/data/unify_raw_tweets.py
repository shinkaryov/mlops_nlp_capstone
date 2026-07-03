import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


TWEET_FILES_CONFIG = {
    "fifa_world_cup_2022_tweets.csv": {
        "date_col": "Date Created",
        "id_col": None,
        "text_col": "Tweet",
        "lang_col": None,
        "tweet_type_col": None,
        "like_col": "Number of Likes",
        "retweet_col": None,
        "user_id_col": None,
        "username_col": None,
        "user_name_col": None,
        "followers_col": None,
        "following_col": None,
        "verified_col": None,
        "source_col": "Source of Tweet",
        "assume_english": False,
    },
    "tweets1.xlsx": {
        "date_col": "Tweet Posted Time",
        "id_col": "Tweet Id",
        "text_col": "Tweet Content",
        "lang_col": "Tweet Language",
        "tweet_type_col": "Tweet Type",
        "like_col": "Likes Received",
        "retweet_col": "Retweets Received",
        "user_id_col": "User  Id",
        "username_col": "Username",
        "user_name_col": "Name",
        "followers_col": "User Followers",
        "following_col": "User Following",
        "verified_col": "Verified or Non-Verified",
        "source_col": "Client",
        "assume_english": False,
    },
    "tweets2.xlsx": {
        "date_col": "Tweet Posted Time",
        "id_col": "Tweet Id",
        "text_col": "Tweet Content",
        "lang_col": "Tweet Language",
        "tweet_type_col": "Tweet Type",
        "like_col": "Likes Received",
        "retweet_col": "Retweets Received",
        "user_id_col": "User  Id",
        "username_col": "Username",
        "user_name_col": "Name",
        "followers_col": "User Followers",
        "following_col": "User Following",
        "verified_col": "Verified or Non-Verified",
        "source_col": "Client",
        "assume_english": False,
    },
    "tweets_football.csv": {
        "date_col": "Date",
        "id_col": None,
        "text_col": "Tweet",
        "lang_col": None,
        "tweet_type_col": None,
        "like_col": None,
        "retweet_col": None,
        "user_id_col": None,
        "username_col": "User",
        "user_name_col": "User",
        "followers_col": None,
        "following_col": None,
        "verified_col": None,
        "source_col": None,
        "assume_english": False,
        "extra_cols": {
            "fc": "source_metric_fc",
            "rc": "source_metric_rc",
        },
    },
    "twtdata.com_tweets_by_hashtag_Fifa_Xd8b0u2i9n.csv": {
        "date_col": "created_at",
        "id_col": "id",
        "text_col": "full_text",
        "lang_col": "lang",
        "tweet_type_col": None,
        "like_col": "favorite_count",
        "retweet_col": "retweet_count",
        "user_id_col": None,
        "username_col": "screen_name",
        "user_name_col": "name",
        "followers_col": "followers_count",
        "following_col": "friends_count",
        "verified_col": None,
        "source_col": "source",
        "assume_english": False,
    },
}


def safe_col(df: pd.DataFrame, col: str | None, default=pd.NA) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def stable_hash(value: str, n: int = 20) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace('"', "", regex=False)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def normalize_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r'^"(.*)"$', r"\1", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype("string")
        .str.replace('"', "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def strip_accents(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_for_match(text) -> str:
    if pd.isna(text):
        return ""

    text = strip_accents(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("@", " ").replace("#", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_text_dedup_key(text) -> str:
    text = normalize_for_match(text)
    text = re.sub(r"^rt\s+\w+\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_language_value(x):
    if pd.isna(x):
        return pd.NA

    x = str(x).replace('"', "").strip().lower()

    if x in {"", "nan", "none", "<na>"}:
        return pd.NA

    lang_map = {
        "english": "en",
        "eng": "en",
        "en": "en",
        "spanish": "es",
        "es": "es",
        "french": "fr",
        "fr": "fr",
        "hindi": "hi",
        "hi": "hi",
        "arabic": "ar",
        "ar": "ar",
        "portuguese": "pt",
        "pt": "pt",
        "german": "de",
        "de": "de",
        "japanese": "ja",
        "ja": "ja",
        "indonesian": "id",
        "in": "id",
        "italian": "it",
        "it": "it",
        "russian": "ru",
        "ru": "ru",
        "ukrainian": "uk",
        "uk": "uk",
        "turkish": "tr",
        "tr": "tr",
        "und": "und",
    }

    return lang_map.get(x, x)


def is_retweet_like(tweet_text, tweet_type) -> bool:
    text = "" if pd.isna(tweet_text) else str(tweet_text).strip().lower()
    ttype = "" if pd.isna(tweet_type) else str(tweet_type).strip().lower()

    return text.startswith("rt @") or ttype in {"retweet", "rt", "re tweet"}


def build_lingua_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder

        detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH,
            Language.SPANISH,
            Language.FRENCH,
            Language.PORTUGUESE,
            Language.ARABIC,
            Language.HINDI,
            Language.GERMAN,
            Language.JAPANESE,
            Language.ITALIAN,
            Language.RUSSIAN,
            Language.TURKISH,
            Language.DUTCH,
            Language.POLISH,
            Language.KOREAN,
            Language.CHINESE,
            Language.UKRAINIAN,
        ).build()

        return detector, Language.ENGLISH

    except Exception as exc:
        print("Lingua is not available.")
        print("Install with: pip install lingua-language-detector")
        print("Unknown-language rows will be dropped unless --assume-unknown-english is used.")
        print("Error:", exc)
        return None, None


def detect_english_with_lingua(text, detector, english_language):
    if detector is None:
        return pd.NA

    if pd.isna(text):
        return pd.NA

    text = str(text).strip()

    if len(text) < 20:
        return pd.NA

    try:
        detected = detector.detect_language_of(text[:500])
        if detected == english_language:
            return "en"
        return "not_en"
    except Exception:
        return pd.NA


def read_raw_tweet_file(path: Path, cfg: dict) -> pd.DataFrame:
    suffix = path.suffix.lower()
    dtype = {}

    for key in ["id_col", "user_id_col"]:
        col = cfg.get(key)
        if col is not None:
            dtype[col] = "string"

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name="Sheet1", dtype=dtype)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, dtype=dtype)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_one_tweet_file(path: Path, cfg: dict) -> pd.DataFrame:
    raw = read_raw_tweet_file(path, cfg)
    n = len(raw)

    tweet_id = normalize_id_series(safe_col(raw, cfg.get("id_col")))
    tweet_text = normalize_text_series(safe_col(raw, cfg.get("text_col")))

    created_at = pd.to_datetime(
        safe_col(raw, cfg.get("date_col")),
        errors="coerce",
        utc=True,
    )

    lang_raw = normalize_text_series(safe_col(raw, cfg.get("lang_col")))
    lang = lang_raw.map(normalize_language_value)

    if cfg.get("assume_english", False):
        lang = lang.fillna("en")

    out = pd.DataFrame(
        {
            "source_file": path.name,
            "source_row": np.arange(n),
            "tweet_id": tweet_id,
            "created_at_utc": created_at,
            "tweet_text": tweet_text,
            "tweet_type": normalize_text_series(safe_col(raw, cfg.get("tweet_type_col"))),
            "lang_raw": lang_raw,
            "lang": lang,
            "like_count": to_numeric_series(safe_col(raw, cfg.get("like_col"))),
            "retweet_count": to_numeric_series(safe_col(raw, cfg.get("retweet_col"))),
            "user_id": normalize_id_series(safe_col(raw, cfg.get("user_id_col"))),
            "username": normalize_text_series(safe_col(raw, cfg.get("username_col"))),
            "user_name": normalize_text_series(safe_col(raw, cfg.get("user_name_col"))),
            "user_followers": to_numeric_series(safe_col(raw, cfg.get("followers_col"))),
            "user_following": to_numeric_series(safe_col(raw, cfg.get("following_col"))),
            "verified_raw": normalize_text_series(safe_col(raw, cfg.get("verified_col"))),
            "tweet_source": normalize_text_series(safe_col(raw, cfg.get("source_col"))),
        }
    )

    for src_col, dst_col in cfg.get("extra_cols", {}).items():
        out[dst_col] = safe_col(raw, src_col)

    out["tweet_text_norm"] = out["tweet_text"].map(normalize_for_match)
    out["tweet_text_dedup_key"] = out["tweet_text"].map(make_text_dedup_key)

    out["is_retweet_like"] = [
        is_retweet_like(text, ttype)
        for text, ttype in zip(out["tweet_text"], out["tweet_type"])
    ]

    out["engagement_known"] = out["like_count"].fillna(0) + out["retweet_count"].fillna(0)

    out["tweet_key"] = np.where(
        out["tweet_id"].notna(),
        "id_" + out["tweet_id"].astype("string"),
        "hash_"
        + (
            out["source_file"].astype("string")
            + "|"
            + out["created_at_utc"].astype("string")
            + "|"
            + out["tweet_text_dedup_key"].astype("string")
        ).map(lambda x: stable_hash(x, 20)),
    )

    return out


def unify_raw_tweets(
    raw_dir: Path,
    output_path: Path,
    detect_language_for_unknown: bool,
    assume_unknown_english: bool,
) -> pd.DataFrame:
    tweet_dfs = []

    for file_name, cfg in TWEET_FILES_CONFIG.items():
        path = raw_dir / file_name

        if not path.exists():
            print(f"Missing file, skipped: {path}")
            continue

        print(f"Reading: {path}")
        part = normalize_one_tweet_file(path, cfg)
        print(f"  shape: {part.shape}")
        tweet_dfs.append(part)

    if not tweet_dfs:
        raise FileNotFoundError(f"No configured tweet files found in {raw_dir}")

    tweets_all = pd.concat(tweet_dfs, ignore_index=True)
    print(f"Raw unified tweets: {tweets_all.shape}")

    tweets_all = tweets_all[
        tweets_all["tweet_text"].notna() & tweets_all["tweet_text_norm"].ne("")
    ].copy()

    tweets_all["retweet_priority"] = tweets_all["is_retweet_like"].astype(int)

    tweets_with_id = tweets_all[tweets_all["tweet_id"].notna()].copy()
    tweets_without_id = tweets_all[tweets_all["tweet_id"].isna()].copy()

    tweets_with_id = (
        tweets_with_id.sort_values(
            ["tweet_id", "retweet_priority", "engagement_known", "source_row"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["tweet_id"], keep="first")
    )

    tweets_dedup = pd.concat([tweets_with_id, tweets_without_id], ignore_index=True)

    tweets_dedup = (
        tweets_dedup.sort_values(
            ["tweet_text_dedup_key", "retweet_priority", "engagement_known", "source_row"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["tweet_text_dedup_key"], keep="first")
        .reset_index(drop=True)
    )

    print(f"After ID/text dedup: {tweets_dedup.shape}")

    unknown_lang_mask = tweets_dedup["lang"].isna()

    if assume_unknown_english:
        tweets_dedup.loc[unknown_lang_mask, "lang"] = "en"
    elif unknown_lang_mask.any() and detect_language_for_unknown:
        detector, english_language = build_lingua_detector()

        if detector is not None:
            print("Detecting language for rows without language metadata...")
            tqdm.pandas()
            tweets_dedup.loc[unknown_lang_mask, "lang"] = (
                tweets_dedup.loc[unknown_lang_mask, "tweet_text"]
                .progress_map(lambda x: detect_english_with_lingua(x, detector, english_language))
            )

    tweets_en = tweets_dedup[tweets_dedup["lang"].eq("en")].copy()
    tweets_en = tweets_en.reset_index(drop=True)

    print(f"English tweets after dedup: {tweets_en.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tweets_en.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    return tweets_en


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/tweets", type=Path)
    parser.add_argument(
        "--output",
        default="data/interim/tweets_unified_english_dedup.csv",
        type=Path,
    )
    parser.add_argument("--detect-language-for-unknown", action="store_true")
    parser.add_argument("--assume-unknown-english", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    unify_raw_tweets(
        raw_dir=args.raw_dir,
        output_path=args.output,
        detect_language_for_unknown=args.detect_language_for_unknown,
        assume_unknown_english=args.assume_unknown_english,
    )