import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None


ALIAS_CONFIG = {
    "aaron ramsdale": {
        "safe": ["aaron ramsdale", "ramsdale", "aaronramsdale"],
        "contextual": {},
    },
    "alisson": {
        "safe": ["alisson", "alisson becker", "alissonbecker"],
        "contextual": {},
    },
    "ederson": {
        "safe": ["ederson"],
        "contextual": {},
    },
    "emiliano martinez": {
        "safe": ["emiliano martinez", "emi martinez", "dibu martinez", "dibu", "emilianomartinez"],
        "contextual": {
            "martinez": ["argentina", "dibu", "keeper", "goalkeeper", "save", "penalty"],
        },
    },
    "thibaut courtois": {
        "safe": ["thibaut courtois", "courtois", "thibautcourtois"],
        "contextual": {},
    },
    "antonio rudiger": {
        "safe": [
            "antonio rudiger",
            "rudiger",
            "rüdiger",
            "antoniorudiger",
            "tony rudiger",
            "tonyrudiger",
            "antoniorüdiger",
        ],
        "contextual": {},
    },
    "joao cancelo": {
        "safe": ["joao cancelo", "joão cancelo", "cancelo", "joaocancelo"],
        "contextual": {},
    },
    "kyle walker": {
        "safe": ["kyle walker", "kylewalker"],
        "contextual": {
            "walker": ["england", "man city", "city", "mbappe", "france"],
        },
    },
    "luke shaw": {
        "safe": ["luke shaw", "lukeshaw"],
        "contextual": {
            "shaw": ["england", "man utd", "united", "left back"],
        },
    },
    "nathan ake": {
        "safe": ["nathan ake", "nathan aké", "ake", "aké", "nathanaké", "nathanake"],
        "contextual": {},
    },
    "ruben dias": {
        "safe": ["ruben dias", "rúben dias", "rubendias", "rúbendias"],
        "contextual": {
            "dias": ["portugal", "man city", "city"],
        },
    },
    "thiago silva": {
        "safe": ["thiago silva", "thiagosilva"],
        "contextual": {
            "silva": ["chelsea", "psg", "brazil", "defender"],
        },
    },
    "trent alexander arnold": {
        "safe": [
            "trent alexander arnold",
            "trent alexander-arnold",
            "alexander arnold",
            "alexander-arnold",
            "taa",
            "trent",
        ],
        "contextual": {},
    },
    "bernardo silva": {
        "safe": ["bernardo silva", "bernardosilva"],
        "contextual": {
            "bernardo": ["portugal", "man city", "city"],
            "silva": ["portugal", "man city", "city"],
        },
    },
    "bruno fernandes": {
        "safe": ["bruno fernandes"],
        "contextual": {
            "bruno": ["portugal", "fernandes", "man utd", "united"],
            "fernandes": ["portugal", "bruno", "man utd", "united"],
        },
    },
    "casemiro": {
        "safe": ["casemiro"],
        "contextual": {
            "case": ["brazil", "fernandes", "man utd", "united"],
        },
    },
    "declan rice": {
        "safe": ["declan rice", "declanrice"],
        "contextual": {
            "rice": ["declan", "england", "arsenal", "west ham"],
        },
    },
    "eduardo camavinga": {
        "safe": ["eduardo camavinga", "camavinga", "eduardocamavinga"],
        "contextual": {
            "cama": ["real", "rm", "france", "mbappe"],
        },
    },
    "enzo fernandez": {
        "safe": ["enzo fernandez", "enzo fernández", "enzofernandez"],
        "contextual": {
            "enzo": ["argentina", "fernandez", "chelsea"],
            "fernandez": ["argentina", "enzo", "chelsea"],
        },
    },
    "ilkay gundogan": {
        "safe": ["ilkay gundogan", "ilkay gündogan", "gundogan", "gündogan", "ilkaygundogan"],
        "contextual": {},
    },
    "jude bellingham": {
        "safe": ["jude bellingham", "bellingham", "judebellingham"],
        "contextual": {
            "jude": ["bvb", "dortmund", "bee", "england"],
        },
    },
    "kevin de bruyne": {
        "safe": ["kevin de bruyne", "de bruyne", "kdb", "debruyne", "kevindebruyne"],
        "contextual": {},
    },
    "luka modric": {
        "safe": ["luka modric", "luka modrić", "modric", "modrić", "lukamodric", "lukamodrić"],
        "contextual": {
            "luka": ["real", "madrid", "croatia", "midfield", "goat", "maestro"],
        },
    },
    "mason mount": {
        "safe": ["mason mount", "masonmount"],
        "contextual": {
            "mount": ["mason", "england", "chelsea", "lampard", "tuchel"],
        },
    },
    "moises caicedo": {
        "safe": ["moises caicedo", "moisés caicedo", "caicedo", "moisescaicedo"],
        "contextual": {},
    },
    "pedri": {
        "safe": ["pedri"],
        "contextual": {},
    },
    "rodri": {
        "safe": ["rodri"],
        "contextual": {},
    },
    "rodrigo de paul": {
        "safe": ["rodrigo de paul", "de paul", "depaul"],
        "contextual": {},
    },
    "bukayo saka": {
        "safe": ["bukayo saka", "saka", "bukayosaka"],
        "contextual": {},
    },
    "cristiano ronaldo": {
        "safe": ["cristiano ronaldo", "cristiano", "ronaldo", "cr7"],
        "contextual": {
            "goat": ["ronaldo", "cr7", "cristiano", "portugal"],
            "cris": ["ronaldo", "cr7", "portugal"],
        },
    },
    "eden hazard": {
        "safe": ["eden hazard", "edenhazard"],
        "contextual": {
            "hazard": ["eden", "belgium", "chelsea", "real", "madrid"],
        },
    },
    "harry kane": {
        "safe": ["harry kane", "kane", "harrykane"],
        "contextual": {},
    },
    "jack grealish": {
        "safe": ["jack grealish", "grealish", "jackgrealish"],
        "contextual": {},
    },
    "julian alvarez": {
        "safe": ["julian alvarez", "julián alvarez", "alvarez", "álvarez", "julianalvarez"],
        "contextual": {},
    },
    "kaoru mitoma": {
        "safe": ["kaoru mitoma", "mitoma", "kaorumitoma"],
        "contextual": {},
    },
    "karim benzema": {
        "safe": ["karim benzema", "benzema", "karimbenzema"],
        "contextual": {},
    },
    "kylian mbappe": {
        "safe": [
            "kylian mbappe",
            "kylian mbappé",
            "mbappe",
            "mbappé",
            "mbappee",
            "kylianmbappe",
        ],
        "contextual": {
            "kylian": ["mbappe", "france", "psg"],
        },
    },
    "lionel messi": {
        "safe": ["lionel messi", "leo messi", "messi", "lm10", "leomessi"],
        "contextual": {
            "leo": ["messi", "argentina", "albiceleste", "lm10"],
            "goat": ["messi", "leo", "argentina", "albiceleste", "lm10"],
        },
    },
    "marcus rashford": {
        "safe": ["marcus rashford", "rashford", "marcusrashford"],
        "contextual": {},
    },
    "neymar": {
        "safe": ["neymar", "neymar jr", "neymar junior", "neymarjr"],
        "contextual": {
            "ney": ["brazil", "neymar", "psg"],
        },
    },
    "olivier giroud": {
        "safe": ["olivier giroud", "giroud"],
        "contextual": {},
    },
    "rafael leao": {
        "safe": ["rafael leao", "rafael leão", "rafa leao", "rafa leão", "leao", "leão", "rafaleao"],
        "contextual": {},
    },
    "robert lewandowski": {
        "safe": ["robert lewandowski", "lewandowski"],
        "contextual": {},
    },
    "sadio mane": {
        "safe": ["sadio mane", "sadio mané", "mane", "mané"],
        "contextual": {},
    },
    "son heung min": {
        "safe": ["son heung min", "son heung-min", "heung min son", "sonny"],
        "contextual": {
            "son": ["korea", "south korea", "tottenham", "spurs"],
        },
    },
    "vinicius junior": {
        "safe": [
            "vinicius junior",
            "vinícius júnior",
            "vinicius jr",
            "vinícius jr",
            "vini jr",
            "vini",
            "vinicius",
            "vinícius",
            "vinijr",
        ],
        "contextual": {},
    },
}


def normalize_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r'^"(.*)"$', r"\1", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def strip_accents(text: str) -> str:
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


def contains_term(text_norm: str, term: str) -> bool:
    term_norm = normalize_for_match(term)

    if not term_norm:
        return False

    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])",
            text_norm,
        )
    )


def contains_any(text_norm: str, terms: list[str]) -> bool:
    return any(contains_term(text_norm, term) for term in terms)


def load_players(players_path: Path) -> pd.DataFrame:
    players_raw = pd.read_csv(players_path, low_memory=False)
    players_raw.columns = [str(c).strip() for c in players_raw.columns]

    required_cols = {"id", "player_name", "position"}
    missing_cols = required_cols - set(players_raw.columns)

    if missing_cols:
        raise ValueError(f"{players_path} is missing columns: {missing_cols}")

    players = players_raw[["id", "player_name", "position"]].copy()
    players = players.rename(columns={"id": "player_id"})

    players["player_id"] = players["player_id"].astype("string")
    players["player_name"] = normalize_text_series(players["player_name"])
    players["position"] = normalize_text_series(players["position"])
    players = players[
        players["player_id"].notna() & players["player_name"].notna()
    ].drop_duplicates(subset=["player_id"])

    players["player_name_norm"] = players["player_name"].map(normalize_for_match)

    return players


def build_alias_tables(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_records = []
    contextual_records = []

    for _, row in players.iterrows():
        player_id = row["player_id"]
        player_name = row["player_name"]
        position = row["position"]
        player_name_norm = row["player_name_norm"]

        cfg = ALIAS_CONFIG.get(player_name_norm, {"safe": [], "contextual": {}})

        safe_aliases = set(cfg.get("safe", []))
        safe_aliases.add(player_name)

        for alias in safe_aliases:
            alias_norm = normalize_for_match(alias)

            if len(alias_norm) < 3:
                continue

            alias_records.append(
                {
                    "alias": alias_norm,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "player_name_norm": player_name_norm,
                }
            )

        for alias, context_terms in cfg.get("contextual", {}).items():
            alias_norm = normalize_for_match(alias)
            context_terms_norm = [normalize_for_match(x) for x in context_terms]

            contextual_records.append(
                {
                    "alias": alias_norm,
                    "context_terms": context_terms_norm,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "player_name_norm": player_name_norm,
                }
            )

    aliases_df = pd.DataFrame(alias_records).drop_duplicates()

    contextual_aliases_df = pd.DataFrame(contextual_records)
    if not contextual_aliases_df.empty:
        contextual_aliases_df = contextual_aliases_df.drop_duplicates(
            subset=["alias", "player_id"]
        )

    return aliases_df, contextual_aliases_df


def generate_ngrams(tokens: list[str], min_n: int = 1, max_n: int = 4) -> list[str]:
    out = []

    for n in range(min_n, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            candidate = " ".join(tokens[i : i + n])
            if len(candidate) >= 5:
                out.append(candidate)

    return list(dict.fromkeys(out))


def match_players(
    tweets_path: Path,
    players_path: Path,
    interim_dir: Path,
    reports_dir: Path,
    enable_fuzzy_fallback: bool,
    fuzzy_score_cutoff: int,
) -> None:
    tweets_en = pd.read_csv(tweets_path, low_memory=False)
    players = load_players(players_path)

    aliases_df, contextual_aliases_df = build_alias_tables(players)

    print(f"Curated players: {players.shape}")
    print(f"Safe aliases: {aliases_df.shape}")
    print(f"Contextual aliases: {contextual_aliases_df.shape}")

    alias_to_candidates = {}
    for alias, group in aliases_df.groupby("alias"):
        alias_to_candidates[alias] = group.to_dict("records")

    all_safe_aliases = sorted(alias_to_candidates.keys(), key=len, reverse=True)

    if not all_safe_aliases:
        raise ValueError("No safe aliases found.")

    safe_alias_pattern = re.compile(
        r"(?<![a-z0-9])("
        + "|".join(re.escape(alias) for alias in all_safe_aliases)
        + r")(?![a-z0-9])"
    )

    def match_safe_aliases(tweet_key: str, text_norm: str) -> list[dict]:
        records = []

        for match in safe_alias_pattern.finditer(text_norm):
            alias = match.group(1)
            candidates = alias_to_candidates.get(alias, [])

            for c in candidates:
                records.append(
                    {
                        "tweet_key": tweet_key,
                        "player_id": c["player_id"],
                        "player_name": c["player_name"],
                        "position": c["position"],
                        "matched_alias": alias,
                        "match_method": "safe_alias",
                        "match_score": 100.0,
                    }
                )

        return records

    def match_contextual_aliases(tweet_key: str, text_norm: str) -> list[dict]:
        records = []

        if contextual_aliases_df.empty:
            return records

        for _, c in contextual_aliases_df.iterrows():
            alias = c["alias"]
            context_terms = c["context_terms"]

            if not contains_term(text_norm, alias):
                continue

            if not contains_any(text_norm, context_terms):
                continue

            records.append(
                {
                    "tweet_key": tweet_key,
                    "player_id": c["player_id"],
                    "player_name": c["player_name"],
                    "position": c["position"],
                    "matched_alias": alias + "_contextual",
                    "match_method": "contextual_alias",
                    "match_score": 90.0,
                }
            )

        return records

    mention_records = []

    for row in tqdm(
        tweets_en[["tweet_key", "tweet_text_norm"]].itertuples(index=False),
        total=len(tweets_en),
        desc="Exact/contextual player matching",
    ):
        mention_records.extend(match_safe_aliases(row.tweet_key, row.tweet_text_norm))
        mention_records.extend(match_contextual_aliases(row.tweet_key, row.tweet_text_norm))

    mentions_exact = pd.DataFrame(mention_records)
    print(f"Exact/contextual mentions: {mentions_exact.shape}")

    mention_cols = [
        "tweet_key",
        "player_id",
        "player_name",
        "position",
        "matched_alias",
        "match_method",
        "match_score",
    ]

    if mentions_exact.empty:
        mentions_exact = pd.DataFrame(columns=mention_cols)

    mentions_fuzzy = pd.DataFrame(columns=mention_cols)

    if enable_fuzzy_fallback:
        if process is None or fuzz is None:
            raise ImportError("rapidfuzz is required for fuzzy matching. Install: pip install rapidfuzz")

        fuzzy_aliases = (
            aliases_df["alias"]
            .drop_duplicates()
            .loc[lambda s: (s.str.len() >= 6) | (s.str.contains(" "))]
            .tolist()
        )
        fuzzy_aliases = sorted(fuzzy_aliases, key=len, reverse=True)

        already_matched_keys = set(mentions_exact["tweet_key"].unique())
        tweets_for_fuzzy = tweets_en[
            ~tweets_en["tweet_key"].isin(already_matched_keys)
        ][["tweet_key", "tweet_text_norm"]].copy()

        fuzzy_records = []

        for row in tqdm(
            tweets_for_fuzzy.itertuples(index=False),
            total=len(tweets_for_fuzzy),
            desc="Strict fuzzy fallback matching",
        ):
            tokens = str(row.tweet_text_norm).split()
            ngrams = generate_ngrams(tokens, min_n=1, max_n=4)[:80]

            for ng in ngrams:
                best = process.extractOne(
                    ng,
                    fuzzy_aliases,
                    scorer=fuzz.WRatio,
                    score_cutoff=fuzzy_score_cutoff,
                )

                if best is None:
                    continue

                matched_alias, score, _ = best
                candidates = alias_to_candidates.get(matched_alias, [])

                for c in candidates:
                    fuzzy_records.append(
                        {
                            "tweet_key": row.tweet_key,
                            "player_id": c["player_id"],
                            "player_name": c["player_name"],
                            "position": c["position"],
                            "matched_alias": matched_alias,
                            "match_method": "fuzzy_alias",
                            "match_score": float(score),
                        }
                    )

        mentions_fuzzy = pd.DataFrame(fuzzy_records)
        if mentions_fuzzy.empty:
            mentions_fuzzy = pd.DataFrame(columns=mention_cols)

    print(f"Fuzzy mentions: {mentions_fuzzy.shape}")

    mentions_all = pd.concat(
        [mentions_exact[mention_cols], mentions_fuzzy[mention_cols]],
        ignore_index=True,
    )

    if mentions_all.empty:
        raise ValueError("No player mentions found. Check aliases and English filtering.")

    mentions_agg = (
        mentions_all.sort_values(
            ["tweet_key", "player_id", "match_score"],
            ascending=[True, True, False],
        )
        .groupby(["tweet_key", "player_id", "player_name", "position"], as_index=False)
        .agg(
            matched_aliases=("matched_alias", lambda x: "|".join(sorted(set(map(str, x))))),
            match_methods=("match_method", lambda x: "|".join(sorted(set(map(str, x))))),
            best_match_score=("match_score", "max"),
        )
    )

    matched_tweets_long = tweets_en.merge(mentions_agg, on="tweet_key", how="inner")
    matched_tweets_long["manual_label"] = pd.NA
    matched_tweets_long["manual_notes"] = pd.NA

    matched_tweets_wide = (
        matched_tweets_long.groupby("tweet_key", as_index=False)
        .agg(
            created_at_utc=("created_at_utc", "first"),
            tweet_text=("tweet_text", "first"),
            source_file=("source_file", "first"),
            tweet_id=("tweet_id", "first"),
            lang=("lang", "first"),
            like_count=("like_count", "first"),
            retweet_count=("retweet_count", "first"),
            username=("username", "first"),
            user_followers=("user_followers", "first"),
            mentioned_player_ids=("player_id", lambda x: "|".join(sorted(set(map(str, x))))),
            mentioned_players=("player_name", lambda x: "|".join(sorted(set(map(str, x))))),
            mentioned_positions=("position", lambda x: "|".join(sorted(set(map(str, x))))),
            matched_aliases=("matched_aliases", lambda x: " || ".join(sorted(set(map(str, x))))),
            match_methods=("match_methods", lambda x: "|".join(sorted(set(map(str, x))))),
            best_match_score=("best_match_score", "max"),
        )
    )

    matched_tweets_wide["manual_label"] = pd.NA
    matched_tweets_wide["manual_notes"] = pd.NA

    player_match_summary = (
        matched_tweets_long.groupby(["player_id", "player_name", "position"], as_index=False)
        .agg(
            tweet_mentions=("tweet_key", "nunique"),
            row_mentions=("tweet_key", "size"),
            total_known_likes=("like_count", "sum"),
            total_known_retweets=("retweet_count", "sum"),
            aliases_seen=("matched_aliases", lambda x: " | ".join(sorted(set(map(str, x)))[:30])),
            example_tweet=("tweet_text", "first"),
        )
        .sort_values(["tweet_mentions", "total_known_likes"], ascending=False)
        .reset_index(drop=True)
    )

    alias_match_summary = (
        matched_tweets_long.groupby(["player_name", "matched_aliases"], as_index=False)
        .agg(
            tweet_mentions=("tweet_key", "nunique"),
            example_tweet=("tweet_text", "first"),
        )
        .sort_values("tweet_mentions", ascending=False)
        .reset_index(drop=True)
    )

    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    players.to_csv(interim_dir / "players_curated_clean.csv", index=False)
    aliases_df.to_csv(interim_dir / "player_safe_aliases.csv", index=False)
    contextual_aliases_df.to_csv(interim_dir / "player_contextual_aliases.csv", index=False)

    # Also save reference copies because these are small and useful for Git tracking.
    reference_dir = Path("data/reference")
    reference_dir.mkdir(parents=True, exist_ok=True)
    players.to_csv(reference_dir / "players_curated_clean.csv", index=False)
    aliases_df.to_csv(reference_dir / "player_safe_aliases.csv", index=False)
    contextual_aliases_df.to_csv(reference_dir / "player_contextual_aliases.csv", index=False)

    mentions_agg.to_csv(interim_dir / "player_mentions_long_only_keys.csv", index=False)
    matched_tweets_long.to_csv(interim_dir / "matched_tweets_long_for_manual_labeling.csv", index=False)
    matched_tweets_wide.to_csv(interim_dir / "matched_tweets_wide_for_manual_labeling.csv", index=False)

    player_match_summary.to_csv(reports_dir / "player_match_summary.csv", index=False)
    alias_match_summary.to_csv(reports_dir / "alias_match_summary.csv", index=False)

    print("Saved outputs:")
    print(f"- {interim_dir / 'players_curated_clean.csv'}")
    print(f"- {interim_dir / 'player_safe_aliases.csv'}")
    print(f"- {interim_dir / 'player_contextual_aliases.csv'}")
    print(f"- {interim_dir / 'player_mentions_long_only_keys.csv'}")
    print(f"- {interim_dir / 'matched_tweets_long_for_manual_labeling.csv'}")
    print(f"- {interim_dir / 'matched_tweets_wide_for_manual_labeling.csv'}")
    print(f"- {reports_dir / 'player_match_summary.csv'}")
    print(f"- {reports_dir / 'alias_match_summary.csv'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tweets",
        default="data/interim/tweets_unified_english_dedup.csv",
        type=Path,
    )
    parser.add_argument("--players", default="data/reference/players.csv", type=Path)
    parser.add_argument("--interim-dir", default="data/interim", type=Path)
    parser.add_argument("--reports-dir", default="reports", type=Path)
    parser.add_argument("--enable-fuzzy-fallback", action="store_true")
    parser.add_argument("--fuzzy-score-cutoff", default=94, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    match_players(
        tweets_path=args.tweets,
        players_path=args.players,
        interim_dir=args.interim_dir,
        reports_dir=args.reports_dir,
        enable_fuzzy_fallback=args.enable_fuzzy_fallback,
        fuzzy_score_cutoff=args.fuzzy_score_cutoff,
    )