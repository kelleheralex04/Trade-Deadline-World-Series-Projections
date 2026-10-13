
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# AUGUST-7 OFFENSIVE SCORING DECOMPOSITION EXPERIMENT
# =============================================================================
#
# PURPOSE
#
# We previously found:
#
#   - % of runs scored via HR was almost useless by itself
#   - after controlling for RD + bullpen, HR-run share received a strongly
#     negative coefficient for reaching the World Series
#
# That does NOT necessarily mean "home runs are bad."
#
# It could instead mean:
#
#   non-HR scoring ability is valuable
#
# This script directly tests that hypothesis.
#
#
# OFFENSE IS DECOMPOSED INTO:
#
#   hr_runs_per_game
#   non_hr_runs_per_game
#
# where:
#
#   hr_runs =
#       runs scored on Statcast home-run plays
#
#   non_hr_runs =
#       total team runs - hr_runs
#
#
# We then compare:
#
#   RD + bullpen
#   RS + RA + bullpen
#
# against:
#
#   HR runs/game
#   non-HR runs/game
#   both HR and non-HR runs/game
#   HR share
#
# and combinations with:
#
#   raw away defense
#   context-adjusted away defense
#
#
# WINNER and WORLD SERIES PARTICIPANT targets are still selected separately.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"C:\Users\kelle\.spyder-py3"
)

BASE_CSV = (
    ROOT
    / "data"
    / "aug7_ws_backtest"
    / "aug7_all_team_snapshots_2015_2026.csv"
)

RUNS_CSV = (
    ROOT
    / "aug7_team_runs_scored_allowed_2015_2026.csv"
)

CONTEXT_FEATURES_CSV = (
    ROOT
    / "aug7_context_adjusted_defense_hr_share_2015_2026.csv"
)

OUTCOMES_1 = (
    ROOT
    / "playoff_team_postseason_outcomes_v2.parquet"
)

OUTCOMES_2 = (
    ROOT
    / "playoff_team_statcast_defense.parquet"
)

MODEL_RESULTS_CSV = (
    ROOT
    / "aug7_hr_vs_nonhr_scoring_model_comparison.csv"
)

SEASON_DETAIL_CSV = (
    ROOT
    / "aug7_hr_vs_nonhr_scoring_season_detail.csv"
)

COEFFICIENTS_CSV = (
    ROOT
    / "aug7_hr_vs_nonhr_scoring_coefficients.csv"
)

CORRELATIONS_CSV = (
    ROOT
    / "aug7_hr_vs_nonhr_scoring_correlations.csv"
)

CURRENT_2026_CSV = (
    ROOT
    / "aug7_2026_hr_vs_nonhr_scoring_rankings.csv"
)

STATCAST_CACHE_DIR = ROOT / "data" / "aug7_statcast_defense"
BULLPEN_VELOCITY_CSV = ROOT / "aug7_bullpen_velocity_variants_2015_2026.csv"


# =============================================================================
# SETTINGS
# =============================================================================

FIRST_SEASON = 2015
LAST_HISTORICAL_SEASON = 2025
CURRENT_SEASON = 2026

MIN_PRIOR_SEASONS = 4
RANDOM_STATE = 42

# =============================================================================
# FINAL MODEL CHOICES
# =============================================================================
#
# Winner model:
#   RD/G + all-pitch bullpen average velocity
#
# Participant model:
#   RD/G
#   + all-pitch bullpen average velocity
#   + raw away defensive residual
#   + non-HR runs/game
#
# Participant probabilities are then normalized within league using the
# historically selected softmax temperature T = 1.25.
#
# This means:
#   sum(AL pennant probabilities) = 1.0
#   sum(NL pennant probabilities) = 1.0
#
# Exactly one team from each league reaches the World Series.
#
FINAL_LEAGUE_SOFTMAX_T = 1.25
FINAL_LEAGUE_SCORE_COLUMN = "league_softmax_T125"


# =============================================================================
# VARIABLES
# =============================================================================

RD = "run_differential_per_game"
RS = "runs_scored_per_game"
RA = "runs_allowed_per_game"

BULLPEN = "bullpen_avg_velocity"
BULLPEN_FF = "bullpen_four_seam_avg_velocity"
BULLPEN_FASTBALL_FAMILY = "bullpen_fastball_family_avg_velocity"

HR_RUNS = "runs_scored_on_hr_plays"
HR_RUN_SHARE = "runs_scored_via_hr_share"

HR_RPG = "hr_runs_per_game"
NON_HR_RUNS = "non_hr_runs"
NON_HR_RPG = "non_hr_runs_per_game"

RAW_AWAY_DEF = (
    "def_raw_woba_prevented_per_100_bbe_away"
)

ADJ_AWAY_DEF = (
    "def_context_adj_woba_prevented_per_100_bbe_away"
)

RAW_ALL_DEF = (
    "def_raw_woba_prevented_per_100_bbe"
)

ADJ_ALL_DEF = (
    "def_context_adj_woba_prevented_per_100_bbe"
)


FINAL_WINNER_FEATURES = [
    RD,
    BULLPEN,
]

FINAL_PARTICIPANT_FEATURES = [
    RD,
    BULLPEN,
    RAW_AWAY_DEF,
    NON_HR_RPG,
]



# =============================================================================
# CURRENT-SEASON INPUT REPAIR
# =============================================================================


def repair_current_team_from_cached_statcast(data, team="ARI"):
    """
    Robust current-season repair directly from cached Statcast.

    Rebuilds, through Aug 7:
      - bullpen all-pitch velocity
      - bullpen FF velocity
      - bullpen FF+SI velocity
      - raw away defensive residual
      - HR runs/game
      - total runs/game from batter score changes
      - non-HR runs/game
      - HR share of runs

    This avoids relying on the incomplete merged 2026 Arizona row.
    """
    current_mask = (
        data["season"].eq(CURRENT_SEASON)
        & data["team"].eq(team)
    )

    if current_mask.sum() != 1:
        raise RuntimeError(
            f"Expected exactly one normalized {team} {CURRENT_SEASON} row "
            f"after duplicate coalescing; found {int(current_mask.sum())}."
        )

    needed = [
        BULLPEN,
        RAW_AWAY_DEF,
        NON_HR_RPG,
    ]

    if data.loc[current_mask, needed].notna().all(axis=1).iloc[0]:
        return data

    print(
        f"\nROBUST REPAIR — rebuilding {team} {CURRENT_SEASON} "
        "inputs directly from cached Statcast"
    )

    statcast = load_cached_statcast_season(
        CURRENT_SEASON
    ).copy()

    raw_team_codes = sorted(
        set(
            statcast.get(
                "home_team",
                pd.Series(dtype="object"),
            ).dropna().astype(str).unique()
        )
        | set(
            statcast.get(
                "away_team",
                pd.Series(dtype="object"),
            ).dropna().astype(str).unique()
        )
    )

    arizona_like_codes = [
        code
        for code in raw_team_codes
        if str(code).upper() in {"AZ", "ARI"}
    ]

    print(
        "Raw Statcast Arizona team code(s) found:",
        arizona_like_codes,
    )

    # ------------------------------------------------------------------
    # Normalize team codes and identify batting / pitching / fielding team
    # ------------------------------------------------------------------
    for col in [
        "home_team",
        "away_team",
    ]:
        if col in statcast.columns:
            statcast[col] = statcast[col].map(
                normalize_team
            )

    top_mask = (
        statcast[
            "inning_topbot"
        ]
        .astype(str)
        .str.lower()
        .eq("top")
    )

    statcast[
        "batting_team"
    ] = np.where(
        top_mask,
        statcast[
            "away_team"
        ],
        statcast[
            "home_team"
        ],
    )

    statcast[
        "pitching_team"
    ] = np.where(
        top_mask,
        statcast[
            "home_team"
        ],
        statcast[
            "away_team"
        ],
    )

    statcast[
        "fielding_team"
    ] = statcast[
        "pitching_team"
    ]

    statcast[
        "fielding_location"
    ] = np.where(
        top_mask,
        "home",
        "away",
    )

    # ------------------------------------------------------------------
    # Bullpen velocity
    # ------------------------------------------------------------------
    sort_cols = [
        c
        for c in [
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ]
        if c in statcast.columns
    ]

    statcast = statcast.sort_values(
        sort_cols
    ).copy()

    statcast[
        "pitcher"
    ] = pd.to_numeric(
        statcast[
            "pitcher"
        ],
        errors="coerce",
    )

    statcast[
        "release_speed"
    ] = pd.to_numeric(
        statcast[
            "release_speed"
        ],
        errors="coerce",
    )

    starter_map = (
        statcast.dropna(
            subset=[
                "game_pk",
                "pitching_team",
                "pitcher",
            ]
        )
        .groupby(
            [
                "game_pk",
                "pitching_team",
            ],
            as_index=False,
        )
        .first()[
            [
                "game_pk",
                "pitching_team",
                "pitcher",
            ]
        ]
        .rename(
            columns={
                "pitcher":
                    "starter_pitcher"
            }
        )
    )

    statcast = statcast.merge(
        starter_map,
        on=[
            "game_pk",
            "pitching_team",
        ],
        how="left",
    )

    team_bp = statcast.loc[
        statcast[
            "pitching_team"
        ].eq(team)
        & statcast[
            "pitcher"
        ].notna()
        & statcast[
            "starter_pitcher"
        ].notna()
        & statcast[
            "pitcher"
        ].ne(
            statcast[
                "starter_pitcher"
            ]
        )
        & statcast[
            "release_speed"
        ].notna()
    ].copy()

    print(
        f"{team} bullpen pitches found: "
        f"{len(team_bp):,}"
    )

    if not team_bp.empty:
        data.loc[
            current_mask,
            BULLPEN,
        ] = team_bp[
            "release_speed"
        ].mean()

        if "pitch_type" in team_bp.columns:
            ff = team_bp.loc[
                team_bp[
                    "pitch_type"
                ].eq("FF"),
                "release_speed",
            ]

            family = team_bp.loc[
                team_bp[
                    "pitch_type"
                ].isin(
                    [
                        "FF",
                        "SI",
                    ]
                ),
                "release_speed",
            ]

            if not ff.empty:
                data.loc[
                    current_mask,
                    BULLPEN_FF,
                ] = ff.mean()

            if not family.empty:
                data.loc[
                    current_mask,
                    BULLPEN_FASTBALL_FAMILY,
                ] = family.mean()

    # ------------------------------------------------------------------
    # Raw away defense
    # ------------------------------------------------------------------
    bbe = statcast.copy()

    for col in [
        "estimated_woba_using_speedangle",
        "woba_value",
    ]:
        bbe[
            col
        ] = pd.to_numeric(
            bbe[
                col
            ],
            errors="coerce",
        )

    bbe = bbe.loc[
        bbe[
            "events"
        ].notna()
        & bbe[
            "events"
        ].ne(
            "home_run"
        )
        & bbe[
            "estimated_woba_using_speedangle"
        ].notna()
        & bbe[
            "woba_value"
        ].notna()
    ].copy()

    bbe[
        "raw_def_residual"
    ] = (
        bbe[
            "estimated_woba_using_speedangle"
        ]
        - bbe[
            "woba_value"
        ]
    )

    team_away = bbe.loc[
        bbe[
            "fielding_team"
        ].eq(team)
        & bbe[
            "fielding_location"
        ].eq(
            "away"
        )
    ].copy()

    print(
        f"{team} away defensive BBE found: "
        f"{len(team_away):,}"
    )

    if len(team_away) >= 100:
        data.loc[
            current_mask,
            RAW_AWAY_DEF,
        ] = (
            100.0
            * team_away[
                "raw_def_residual"
            ].mean()
        )

    # ------------------------------------------------------------------
    # Scoring reconstruction
    #
    # Use completed PA rows only (events non-null). post_bat_score minus
    # bat_score gives runs scored on the play by the batting team.
    # This captures all runs, not just HR runs.
    # ------------------------------------------------------------------
    pa = statcast.loc[
        statcast[
            "events"
        ].notna()
        & statcast[
            "batting_team"
        ].eq(team)
    ].copy()

    for col in [
        "bat_score",
        "post_bat_score",
    ]:
        if col in pa.columns:
            pa[
                col
            ] = pd.to_numeric(
                pa[
                    col
                ],
                errors="coerce",
            )

    if {
        "bat_score",
        "post_bat_score",
    }.issubset(
        pa.columns
    ):
        pa[
            "runs_on_play"
        ] = (
            pa[
                "post_bat_score"
            ]
            - pa[
                "bat_score"
            ]
        )

        pa.loc[
            (
                pa[
                    "runs_on_play"
                ] < 0
            )
            | (
                pa[
                    "runs_on_play"
                ] > 4
            ),
            "runs_on_play",
        ] = np.nan

        # Dedupe to one row per completed PA.
        pa_dedupe = [
            c
            for c in [
                "game_pk",
                "at_bat_number",
            ]
            if c in pa.columns
        ]

        if pa_dedupe:
            pa = (
                pa.sort_values(
                    [
                        c
                        for c in [
                            "game_pk",
                            "at_bat_number",
                            "pitch_number",
                        ]
                        if c in pa.columns
                    ]
                )
                .drop_duplicates(
                    pa_dedupe,
                    keep="last",
                )
            )

        total_runs = pa[
            "runs_on_play"
        ].sum(
            min_count=1
        )

        hr_runs = pa.loc[
            pa[
                "events"
            ].eq(
                "home_run"
            ),
            "runs_on_play",
        ].sum(
            min_count=1
        )

        games = pa[
            "game_pk"
        ].nunique()

        print(
            f"{team} batting games found: {games}"
        )

        print(
            f"{team} total runs reconstructed: "
            f"{total_runs}"
        )

        print(
            f"{team} HR runs reconstructed: "
            f"{hr_runs}"
        )

        if (
            games > 0
            and pd.notna(
                total_runs
            )
            and pd.notna(
                hr_runs
            )
        ):
            rs_g = (
                total_runs
                / games
            )

            hr_rpg = (
                hr_runs
                / games
            )

            non_hr_rpg = (
                total_runs
                - hr_runs
            ) / games

            hr_share = (
                hr_runs
                / total_runs
                if total_runs > 0
                else np.nan
            )

            data.loc[
                current_mask,
                RS,
            ] = rs_g

            data.loc[
                current_mask,
                HR_RPG,
            ] = hr_rpg

            data.loc[
                current_mask,
                NON_HR_RPG,
            ] = non_hr_rpg

            data.loc[
                current_mask,
                HR_RUN_SHARE,
            ] = hr_share

            if "games_through_aug7" in data.columns:
                data.loc[
                    current_mask,
                    "games_through_aug7",
                ] = games

            if "runs_scored" in data.columns:
                data.loc[
                    current_mask,
                    "runs_scored",
                ] = total_runs

            if HR_RUNS in data.columns:
                data.loc[
                    current_mask,
                    HR_RUNS,
                ] = hr_runs

            # Preserve RD/G from the base snapshot and derive RA/G.
            if pd.notna(
                data.loc[
                    current_mask,
                    RD,
                ].iloc[0]
            ):
                data.loc[
                    current_mask,
                    RA,
                ] = (
                    rs_g
                    - data.loc[
                        current_mask,
                        RD,
                    ].iloc[0]
                )

    print(
        "\nRepaired row:"
    )

    print(
        data.loc[
            current_mask,
            [
                "team",
                RD,
                RS,
                RA,
                BULLPEN,
                BULLPEN_FF,
                BULLPEN_FASTBALL_FAMILY,
                RAW_AWAY_DEF,
                HR_RPG,
                NON_HR_RPG,
                HR_RUN_SHARE,
            ],
        ].to_string(
            index=False
        )
    )

    return data


# =============================================================================
# LEAGUE-COMPETITION ADJUSTMENT
# =============================================================================

def add_league_competition_scores(frame):
    """
    Convert independent participant probabilities into league-relative
    probabilities.

    Exactly one AL team and one NL team can reach the World Series.

    Variants tested:
      raw_probability:
          ordinary independent logistic probability

      league_share:
          p_i / sum(p_j in same league)

      logit_softmax_T1:
          exp(logit_i) / sum(exp(logit_j)) in same league

      logit_softmax_T075:
          same but sharper competition adjustment

      logit_softmax_T125:
          same but flatter competition adjustment

    The normalized scores sum to 1 within each league-season.
    """
    frame = frame.copy()

    frame[
        "league"
    ] = frame[
        "team"
    ].map(
        team_league
    )

    p = frame[
        "score"
    ].clip(
        1e-6,
        1 - 1e-6,
    )

    frame[
        "raw_probability"
    ] = p

    frame[
        "league_share"
    ] = (
        frame[
            "raw_probability"
        ]
        / frame.groupby(
            [
                "season",
                "league",
            ]
        )[
            "raw_probability"
        ].transform(
            "sum"
        )
    )

    logits = np.log(
        p
        / (
            1 - p
        )
    )

    for temp, name in [
        (
            1.00,
            "league_softmax_T100",
        ),
        (
            0.75,
            "league_softmax_T075",
        ),
        (
            1.25,
            "league_softmax_T125",
        ),
    ]:
        z = (
            logits
            / temp
        )

        # Numerical stability within each season/league.
        max_z = (
            pd.Series(
                z,
                index=frame.index,
            )
            .groupby(
                [
                    frame[
                        "season"
                    ],
                    frame[
                        "league"
                    ],
                ]
            )
            .transform(
                "max"
            )
        )

        exp_z = np.exp(
            z
            - max_z
        )

        frame[
            name
        ] = (
            exp_z
            / pd.Series(
                exp_z,
                index=frame.index,
            )
            .groupby(
                [
                    frame[
                        "season"
                    ],
                    frame[
                        "league"
                    ],
                ]
            )
            .transform(
                "sum"
            )
        )

    return frame


def summarize_participant_score_variant(
    predictions,
    score_col,
):
    rows = []

    for season, test in predictions.groupby(
        "season"
    ):
        test = test.copy()

        test[
            "variant_rank"
        ] = (
            test[
                score_col
            ]
            .rank(
                method="min",
                ascending=False,
            )
        )

        finalists = test.loc[
            test[
                "reached_world_series"
            ].eq(
                1
            )
        ]

        ranks = finalists[
            "variant_rank"
        ].dropna().tolist()

        if len(ranks) != 2:
            continue

        rows.append(
            {
                "season":
                    season,
                "both_top5":
                    int(
                        max(
                            ranks
                        ) <= 5
                    ),
                "top3":
                    int(
                        min(
                            ranks
                        ) <= 3
                    ),
                "top5":
                    int(
                        min(
                            ranks
                        ) <= 5
                    ),
                "worst_rank":
                    max(
                        ranks
                    ),
                "mean_rank":
                    np.mean(
                        ranks
                    ),
            }
        )

    season_df = pd.DataFrame(
        rows
    )

    y = predictions[
        "reached_world_series"
    ].astype(
        int
    )

    p = predictions[
        score_col
    ].astype(
        float
    )

    return {
        "score_variant":
            score_col,

        "both_finalists_top5_rate":
            season_df[
                "both_top5"
            ].mean(),

        "top3_rate":
            season_df[
                "top3"
            ].mean(),

        "top5_rate":
            season_df[
                "top5"
            ].mean(),

        "average_worst_rank":
            season_df[
                "worst_rank"
            ].mean(),

        "average_mean_rank":
            season_df[
                "mean_rank"
            ].mean(),

        "pr_auc":
            average_precision_score(
                y,
                p,
            ),

        "roc_auc":
            safe_auc(
                y,
                p,
            ),

        "brier":
            brier_score_loss(
                y,
                p,
            ),
    }


# =============================================================================
# MODEL SPECS
# =============================================================================
#
# The models are intentionally redundant in places.
# This is an experiment designed to answer specific baseball questions.
#
# =============================================================================

MODEL_SPECS = {
    "A_RD_BP_ALL_PITCH": [RD, BULLPEN],
    "B_RD_BP_FOUR_SEAM": [RD, BULLPEN_FF],
    "C_RD_BP_FASTBALL_FAMILY": [RD, BULLPEN_FASTBALL_FAMILY],
    "D_RD_ALL_3_VELOCITIES": [RD, BULLPEN, BULLPEN_FF, BULLPEN_FASTBALL_FAMILY],

    "E_RS_RA_BP_ALL_PITCH": [RS, RA, BULLPEN],
    "F_RS_RA_BP_FOUR_SEAM": [RS, RA, BULLPEN_FF],
    "G_RS_RA_BP_FASTBALL_FAMILY": [RS, RA, BULLPEN_FASTBALL_FAMILY],

    "H_RD_BP_ALL_NONHR": [RD, BULLPEN, NON_HR_RPG],
    "I_RD_BP_FF_NONHR": [RD, BULLPEN_FF, NON_HR_RPG],
    "J_RD_BP_FAMILY_NONHR": [RD, BULLPEN_FASTBALL_FAMILY, NON_HR_RPG],

    "K_RD_BP_ALL_RAW_AWAY_DEF": [RD, BULLPEN, RAW_AWAY_DEF],
    "L_RD_BP_FF_RAW_AWAY_DEF": [RD, BULLPEN_FF, RAW_AWAY_DEF],
    "M_RD_BP_FAMILY_RAW_AWAY_DEF": [RD, BULLPEN_FASTBALL_FAMILY, RAW_AWAY_DEF],

    "N_RD_BP_ALL_RAW_AWAY_DEF_NONHR": [RD, BULLPEN, RAW_AWAY_DEF, NON_HR_RPG],
    "O_RD_BP_FF_RAW_AWAY_DEF_NONHR": [RD, BULLPEN_FF, RAW_AWAY_DEF, NON_HR_RPG],
    "P_RD_BP_FAMILY_RAW_AWAY_DEF_NONHR": [RD, BULLPEN_FASTBALL_FAMILY, RAW_AWAY_DEF, NON_HR_RPG],

    "Q_HR_NONHR_RA_BP_ALL": [HR_RPG, NON_HR_RPG, RA, BULLPEN],
    "R_HR_NONHR_RA_BP_FF": [HR_RPG, NON_HR_RPG, RA, BULLPEN_FF],
    "S_HR_NONHR_RA_BP_FAMILY": [HR_RPG, NON_HR_RPG, RA, BULLPEN_FASTBALL_FAMILY],

    "T_HR_SHARE_ONLY": [HR_RUN_SHARE],
    "U_NONHR_RPG_ONLY": [NON_HR_RPG],
}


# =============================================================================
# TEAM NORMALIZATION
# =============================================================================

ALIASES = {
    "AZ": "ARI",
    "OAK": "ATH",
    "KCR": "KC",
    "CHW": "CWS",
    "SFG": "SF",
    "SDP": "SD",
    "TBR": "TB",
    "WSN": "WSH",
    "WAS": "WSH",
}


MLB_TEAMS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS",
    "CIN", "CLE", "COL", "DET", "HOU", "KC",
    "LAA", "LAD", "MIA", "MIL", "MIN", "NYM",
    "NYY", "ATH", "PHI", "PIT", "SD", "SF",
    "SEA", "STL", "TB", "TEX", "TOR", "WSH",
}


AL_TEAMS = {
    "BAL", "BOS", "CWS", "CLE", "DET",
    "HOU", "KC", "LAA", "MIN", "NYY",
    "ATH", "SEA", "TB", "TEX", "TOR",
}

NL_TEAMS = {
    "ARI", "ATL", "CHC", "CIN", "COL",
    "LAD", "MIA", "MIL", "NYM", "PHI",
    "PIT", "SD", "SF", "STL", "WSH",
}


def team_league(team):
    if team in AL_TEAMS:
        return "AL"
    if team in NL_TEAMS:
        return "NL"
    return np.nan


def normalize_team(
    value,
):
    if value is None or pd.isna(
        value
    ):
        return None

    value = str(
        value
    ).strip().upper()

    return ALIASES.get(
        value,
        value,
    )


def collapse_duplicate_team_seasons(frame, label):
    """
    After team-code normalization, aliases such as AZ -> ARI can create
    duplicate season/team rows.

    Coalesce duplicates column-by-column:
      1) rank duplicate rows by number of non-null values
      2) start with the most complete row
      3) fill its missing fields from the other duplicate row(s)

    This preserves real data from both rows without arbitrarily discarding
    the more complete Arizona record.
    """
    frame = frame.copy()

    dup_mask = frame.duplicated(
        subset=[
            "season",
            "team",
        ],
        keep=False,
    )

    if not dup_mask.any():
        return frame

    duplicates = (
        frame.loc[
            dup_mask,
            [
                "season",
                "team",
            ],
        ]
        .value_counts()
        .reset_index(
            name="rows"
        )
        .sort_values(
            [
                "season",
                "team",
            ]
        )
    )

    print(
        f"\\nDuplicate normalized team-season rows found in {label}:"
    )
    print(
        duplicates.to_string(
            index=False
        )
    )

    pieces = []

    for (
        season,
        team,
    ), group in frame.groupby(
        [
            "season",
            "team",
        ],
        dropna=False,
        sort=False,
    ):
        if len(group) == 1:
            pieces.append(
                group.iloc[
                    [0]
                ].copy()
            )
            continue

        work = group.copy()

        # Most-complete record first.
        work[
            "__nonnull_count"
        ] = work.notna().sum(
            axis=1
        )

        work = work.sort_values(
            "__nonnull_count",
            ascending=False,
        )

        combined = work.iloc[
            0
        ].copy()

        for idx in work.index[
            1:
        ]:
            row = work.loc[
                idx
            ]

            for col in frame.columns:
                if pd.isna(
                    combined[
                        col
                    ]
                ) and pd.notna(
                    row[
                        col
                    ]
                ):
                    combined[
                        col
                    ] = row[
                        col
                    ]

        combined = combined.drop(
            labels=[
                "__nonnull_count"
            ],
            errors="ignore",
        )

        pieces.append(
            combined.to_frame().T
        )

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    # Restore numeric season if pandas object-coerced it.
    out[
        "season"
    ] = pd.to_numeric(
        out[
            "season"
        ],
        errors="coerce",
    )

    remaining = out.duplicated(
        subset=[
            "season",
            "team",
        ],
        keep=False,
    )

    if remaining.any():
        raise RuntimeError(
            f"Duplicate team-season rows remain after coalescing {label}."
        )

    print(
        f"Duplicate rows in {label} successfully coalesced."
    )

    return out



# =============================================================================
# BULLPEN VELOCITY VARIANTS FROM CACHED STATCAST
# =============================================================================

def load_cached_statcast_season(season):
    files = sorted(STATCAST_CACHE_DIR.glob(f"statcast_{season}_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No cached Statcast files found for {season} in:\n{STATCAST_CACHE_DIR}"
        )

    parts = [pd.read_parquet(path) for path in files]
    data = pd.concat(parts, ignore_index=True)

    if "game_date" in data.columns:
        data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
        cutoff = pd.Timestamp(year=int(season), month=8, day=7)
        data = data.loc[data["game_date"].le(cutoff)].copy()

    if "game_type" in data.columns:
        data = data.loc[data["game_type"].eq("R")].copy()

    dedupe_cols = [
        c for c in ["game_pk", "at_bat_number", "pitch_number"]
        if c in data.columns
    ]
    if dedupe_cols:
        data = data.drop_duplicates(dedupe_cols)

    return data


def build_bullpen_velocity_variants():
    if BULLPEN_VELOCITY_CSV.exists():
        cached = pd.read_csv(BULLPEN_VELOCITY_CSV)
        needed = {
            "season", "team", BULLPEN, BULLPEN_FF,
            BULLPEN_FASTBALL_FAMILY
        }
        if needed.issubset(cached.columns):
            cached["team"] = cached["team"].map(normalize_team)

            cached[
                "season"
            ] = pd.to_numeric(
                cached[
                    "season"
                ],
                errors="coerce",
            )

            cached = collapse_duplicate_team_seasons(
                cached,
                "cached bullpen velocity variants",
            )

            print(
                f"Using cached bullpen velocity variants:\n"
                f"{BULLPEN_VELOCITY_CSV}"
            )

            return cached

    rows = []

    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        print(f"Building bullpen velocity variants: {season}")
        data = load_cached_statcast_season(season)

        required = [
            "game_pk", "pitcher", "inning_topbot", "home_team",
            "away_team", "release_speed", "pitch_type"
        ]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise KeyError(
                f"Statcast cache for {season} missing columns: {missing}"
            )

        sort_cols = [
            c for c in ["game_pk", "at_bat_number", "pitch_number"]
            if c in data.columns
        ]
        data = data.sort_values(sort_cols).copy()

        top_mask = (
            data["inning_topbot"]
            .astype(str)
            .str.lower()
            .eq("top")
        )

        data["pitching_team"] = np.where(
            top_mask,
            data["home_team"],
            data["away_team"],
        )
        data["pitching_team"] = data["pitching_team"].map(normalize_team)
        data["pitcher"] = pd.to_numeric(data["pitcher"], errors="coerce")
        data["release_speed"] = pd.to_numeric(
            data["release_speed"], errors="coerce"
        )

        starter_map = (
            data.dropna(subset=["game_pk", "pitching_team", "pitcher"])
            .groupby(["game_pk", "pitching_team"], as_index=False)
            .first()[["game_pk", "pitching_team", "pitcher"]]
            .rename(columns={"pitcher": "starter_pitcher"})
        )

        data = data.merge(
            starter_map,
            on=["game_pk", "pitching_team"],
            how="left",
        )

        bullpen = data.loc[
            data["pitcher"].notna()
            & data["starter_pitcher"].notna()
            & data["pitcher"].ne(data["starter_pitcher"])
            & data["release_speed"].notna()
            & data["pitching_team"].isin(MLB_TEAMS)
        ].copy()

        all_pitch = (
            bullpen.groupby("pitching_team")["release_speed"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={
                "pitching_team": "team",
                "mean": BULLPEN,
                "count": "bullpen_all_pitch_n",
            })
        )

        ff = (
            bullpen.loc[bullpen["pitch_type"].eq("FF")]
            .groupby("pitching_team")["release_speed"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={
                "pitching_team": "team",
                "mean": BULLPEN_FF,
                "count": "bullpen_four_seam_n",
            })
        )

        family = (
            bullpen.loc[bullpen["pitch_type"].isin(["FF", "SI"])]
            .groupby("pitching_team")["release_speed"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={
                "pitching_team": "team",
                "mean": BULLPEN_FASTBALL_FAMILY,
                "count": "bullpen_fastball_family_n",
            })
        )

        season_df = all_pitch.merge(ff, on="team", how="outer").merge(
            family, on="team", how="outer"
        )
        season_df["season"] = season
        rows.append(season_df)

    out = pd.concat(rows, ignore_index=True)
    out["team"] = out["team"].map(normalize_team)
    out = out.loc[out["team"].isin(MLB_TEAMS)].copy()
    out.to_csv(BULLPEN_VELOCITY_CSV, index=False)
    return out


# =============================================================================
# LOAD DATA
# =============================================================================

def load_outcomes():
    source = (
        OUTCOMES_1
        if OUTCOMES_1.exists()
        else OUTCOMES_2
    )

    if not source.exists():
        raise FileNotFoundError(
            "Could not find postseason outcomes parquet."
        )

    outcomes = pd.read_parquet(
        source
    ).copy()

    required = [
        "season",
        "team",
        "won_world_series",
        "reached_world_series",
    ]

    missing = [
        c
        for c in required
        if c not in outcomes.columns
    ]

    if missing:
        raise KeyError(
            f"Outcome file missing: {missing}"
        )

    outcomes[
        "season"
    ] = pd.to_numeric(
        outcomes[
            "season"
        ],
        errors="coerce",
    )

    outcomes[
        "team"
    ] = outcomes[
        "team"
    ].map(
        normalize_team
    )

    return outcomes[
        required
    ].drop_duplicates(
        [
            "season",
            "team",
        ]
    )



def load_data():
    for path in [
        BASE_CSV,
        CONTEXT_FEATURES_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    base = pd.read_csv(
        BASE_CSV
    )

    context = pd.read_csv(
        CONTEXT_FEATURES_CSV
    )

    for frame in [
        base,
        context,
    ]:
        frame[
            "season"
        ] = pd.to_numeric(
            frame[
                "season"
            ],
            errors="coerce",
        )

        frame[
            "team"
        ] = frame[
            "team"
        ].map(
            normalize_team
        )

        frame.drop(
            frame.loc[
                ~frame[
                    "team"
                ].isin(
                    MLB_TEAMS
                )
            ].index,
            inplace=True,
        )

    base = collapse_duplicate_team_seasons(
        base,
        "base snapshot",
    )

    context = collapse_duplicate_team_seasons(
        context,
        "context / HR feature file",
    )

    # -------------------------------------------------------------------------
    # CONTEXT / HR FILE
    #
    # This file already contains:
    #   runs_scored
    #   games_through_aug7
    #   HR runs / HR share
    #   defense metrics
    #
    # We therefore do NOT need the separate
    # aug7_team_runs_scored_allowed_2015_2026.csv file.
    # -------------------------------------------------------------------------

    context_keep = [
        "season",
        "team",
        "games_through_aug7",
        "runs_scored",
        HR_RUNS,
        HR_RUN_SHARE,
        RAW_AWAY_DEF,
        ADJ_AWAY_DEF,
        RAW_ALL_DEF,
        ADJ_ALL_DEF,
    ]

    missing_context = [
        c
        for c in context_keep
        if c not in context.columns
    ]

    if missing_context:
        raise KeyError(
            "Context feature file is missing required columns: "
            f"{missing_context}\n\n"
            f"File used:\n{CONTEXT_FEATURES_CSV}"
        )

    data = base.merge(
        context[
            context_keep
        ],
        on=[
            "season",
            "team",
        ],
        how="left",
    )

    bullpen_velo = build_bullpen_velocity_variants()

    if BULLPEN in data.columns:
        data = data.rename(columns={BULLPEN: f"{BULLPEN}_base"})

    data = data.merge(
        bullpen_velo,
        on=["season", "team"],
        how="left",
    )

    if f"{BULLPEN}_base" in data.columns:
        data[BULLPEN] = data[BULLPEN].fillna(
            data[f"{BULLPEN}_base"]
        )

    data = data.merge(
        load_outcomes(),
        on=[
            "season",
            "team",
        ],
        how="left",
    )

    data = collapse_duplicate_team_seasons(
        data,
        "final merged modeling table",
    )

    # Hard uniqueness check.
    if data.duplicated(
        subset=[
            "season",
            "team",
        ],
        keep=False,
    ).any():
        raise RuntimeError(
            "Duplicate season/team rows remain in final modeling table."
        )

    # -------------------------------------------------------------------------
    # TARGETS
    # -------------------------------------------------------------------------

    historical = data[
        "season"
    ].between(
        FIRST_SEASON,
        LAST_HISTORICAL_SEASON,
    )

    for target in [
        "won_world_series",
        "reached_world_series",
    ]:
        data.loc[
            historical,
            target,
        ] = (
            pd.to_numeric(
                data.loc[
                    historical,
                    target,
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .astype(
                int
            )
        )

    # -------------------------------------------------------------------------
    # NUMERIC CONVERSION
    # -------------------------------------------------------------------------

    for column in [
        "games_through_aug7",
        "runs_scored",
        HR_RUNS,
        HR_RUN_SHARE,
        RD,
        BULLPEN,
        BULLPEN_FF,
        BULLPEN_FASTBALL_FAMILY,
        RAW_AWAY_DEF,
        ADJ_AWAY_DEF,
        RAW_ALL_DEF,
        ADJ_ALL_DEF,
    ]:
        if column not in data.columns:
            data[
                column
            ] = np.nan

        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # RECONSTRUCT RS / RA FROM EXISTING FILES
    #
    # RS/G = total runs scored / games
    #
    # RD/G = RS/G - RA/G
    #
    # therefore:
    # RA/G = RS/G - RD/G
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # OFFENSIVE RATE RECONSTRUCTION
    #
    # Preferred:
    #   RS/G = runs_scored / games
    #
    # Fallback:
    #   HR share = HR runs / total runs
    #   therefore RS/G = HR runs/game / HR share
    #
    # This is needed because some 2026 rows have HR-run information but no
    # raw runs_scored value in the merged context file.
    # -------------------------------------------------------------------------

    data[
        HR_RPG
    ] = (
        data[
            HR_RUNS
        ]
        / data[
            "games_through_aug7"
        ]
    )

    data[
        RS
    ] = (
        data[
            "runs_scored"
        ]
        / data[
            "games_through_aug7"
        ]
    )

    fallback_mask = (
        data[
            RS
        ].isna()
        & data[
            HR_RPG
        ].notna()
        & data[
            HR_RUN_SHARE
        ].notna()
        & (
            data[
                HR_RUN_SHARE
            ] > 0
        )
    )

    data.loc[
        fallback_mask,
        RS,
    ] = (
        data.loc[
            fallback_mask,
            HR_RPG,
        ]
        / data.loc[
            fallback_mask,
            HR_RUN_SHARE,
        ]
    )

    # -------------------------------------------------------------------------
    # RUNS ALLOWED
    #
    # RD/G = RS/G - RA/G
    # -------------------------------------------------------------------------

    data[
        RA
    ] = (
        data[
            RS
        ]
        - data[
            RD
        ]
    )

    # -------------------------------------------------------------------------
    # OFFENSIVE DECOMPOSITION
    # -------------------------------------------------------------------------

    data[
        NON_HR_RPG
    ] = (
        data[
            RS
        ]
        - data[
            HR_RPG
        ]
    )

    data.loc[
        data[
            NON_HR_RPG
        ] < 0,
        NON_HR_RPG,
    ] = np.nan

    data[
        NON_HR_RUNS
    ] = (
        data[
            NON_HR_RPG
        ]
        * data[
            "games_through_aug7"
        ]
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    data[
        "offense_reconstruction_error"
    ] = (
        data[
            HR_RPG
        ]
        + data[
            NON_HR_RPG
        ]
        - data[
            RS
        ]
    )

    data[
        "rd_reconstruction_error"
    ] = (
        data[
            RS
        ]
        - data[
            RA
        ]
        - data[
            RD
        ]
    )

    return data


# =============================================================================
# MODEL
# =============================================================================

def make_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=
                        RANDOM_STATE,
                ),
            ),
        ]
    )


def safe_auc(
    y,
    p,
):
    if len(
        np.unique(
            y
        )
    ) < 2:
        return np.nan

    return roc_auc_score(
        y,
        p,
    )


# =============================================================================
# BACKTEST
# =============================================================================

def evaluate_model(
    data,
    model_name,
    features,
    target,
):
    prediction_parts = []
    season_rows = []

    for test_season in range(
        FIRST_SEASON,
        LAST_HISTORICAL_SEASON + 1,
    ):
        prior_seasons = list(
            range(
                FIRST_SEASON,
                test_season,
            )
        )

        if len(
            prior_seasons
        ) < MIN_PRIOR_SEASONS:
            continue

        train = data.loc[
            data[
                "season"
            ].isin(
                prior_seasons
            )
        ].copy()

        test = data.loc[
            data[
                "season"
            ].eq(
                test_season
            )
        ].copy()

        if test.empty:
            continue

        model = make_model()

        model.fit(
            train[
                features
            ],
            train[
                target
            ].astype(
                int
            ),
        )

        test[
            "score"
        ] = model.predict_proba(
            test[
                features
            ]
        )[:, 1]

        test[
            "rank"
        ] = (
            test[
                "score"
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(
                int
            )
        )

        positive = test.loc[
            test[
                target
            ].eq(
                1
            )
        ]

        ranks = (
            positive[
                "rank"
            ]
            .astype(
                int
            )
            .tolist()
        )

        if not ranks:
            continue

        if target == "won_world_series":
            champion_rank = ranks[
                0
            ]

            season_rows.append(
                {
                    "model":
                        model_name,

                    "target":
                        target,

                    "season":
                        test_season,

                    "top3":
                        int(
                            champion_rank <= 3
                        ),

                    "top5":
                        int(
                            champion_rank <= 5
                        ),

                    "best_rank":
                        champion_rank,

                    "worst_rank":
                        champion_rank,

                    "mean_rank":
                        champion_rank,

                    "both_top5":
                        np.nan,
                }
            )

        else:
            season_rows.append(
                {
                    "model":
                        model_name,

                    "target":
                        target,

                    "season":
                        test_season,

                    "top3":
                        int(
                            min(
                                ranks
                            ) <= 3
                        ),

                    "top5":
                        int(
                            min(
                                ranks
                            ) <= 5
                        ),

                    "best_rank":
                        min(
                            ranks
                        ),

                    "worst_rank":
                        max(
                            ranks
                        ),

                    "mean_rank":
                        np.mean(
                            ranks
                        ),

                    "both_top5":
                        int(
                            len(
                                ranks
                            ) == 2
                            and max(
                                ranks
                            ) <= 5
                        ),
                }
            )

        temp = test[
            [
                "season",
                "team",
                target,
                "score",
                "rank",
            ]
        ].copy()

        temp[
            "model"
        ] = model_name

        temp[
            "target_name"
        ] = target

        prediction_parts.append(
            temp
        )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    seasons = pd.DataFrame(
        season_rows
    )

    y = predictions[
        target
    ].astype(
        int
    )

    p = predictions[
        "score"
    ].astype(
        float
    )

    summary = {
        "model":
            model_name,

        "target":
            target,

        "features":
            " | ".join(
                features
            ),

        "n_features":
            len(
                features
            ),

        "pr_auc":
            average_precision_score(
                y,
                p,
            ),

        "roc_auc":
            safe_auc(
                y,
                p,
            ),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "top3_rate":
            seasons[
                "top3"
            ].mean(),

        "top5_rate":
            seasons[
                "top5"
            ].mean(),

        "average_best_rank":
            seasons[
                "best_rank"
            ].mean(),

        "average_worst_rank":
            seasons[
                "worst_rank"
            ].mean(),

        "average_mean_rank":
            seasons[
                "mean_rank"
            ].mean(),

        "worst_observed_rank":
            seasons[
                "worst_rank"
            ].max(),

        "both_finalists_top5_rate":
            (
                seasons[
                    "both_top5"
                ].mean()
                if target
                == "reached_world_series"
                else np.nan
            ),
    }

    return (
        summary,
        seasons,
        predictions,
    )


# =============================================================================
# SELECTION SCORE
# =============================================================================

def quality_high(
    series,
):
    lo = series.min()
    hi = series.max()

    if hi == lo:
        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        series
        - lo
    ) / (
        hi
        - lo
    )


def quality_low(
    series,
):
    lo = series.min()
    hi = series.max()

    if hi == lo:
        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        hi
        - series
    ) / (
        hi
        - lo
    )


def add_selection_scores(
    results,
):
    results = results.copy()

    results[
        "selection_score"
    ] = np.nan

    winner = results.loc[
        results[
            "target"
        ].eq(
            "won_world_series"
        )
    ].copy()

    winner[
        "selection_score"
    ] = (
        0.25
        * winner[
            "top5_rate"
        ]
        + 0.20
        * winner[
            "top3_rate"
        ]
        + 0.18
        * quality_low(
            winner[
                "average_mean_rank"
            ]
        )
        + 0.12
        * quality_low(
            winner[
                "worst_observed_rank"
            ]
        )
        + 0.10
        * quality_high(
            winner[
                "pr_auc"
            ]
        )
        + 0.08
        * quality_high(
            winner[
                "roc_auc"
            ]
        )
        + 0.07
        * quality_low(
            winner[
                "brier"
            ]
        )
    )

    results.loc[
        winner.index,
        "selection_score",
    ] = winner[
        "selection_score"
    ]

    participant = results.loc[
        results[
            "target"
        ].eq(
            "reached_world_series"
        )
    ].copy()

    participant[
        "selection_score"
    ] = (
        0.20
        * participant[
            "both_finalists_top5_rate"
        ]
        + 0.13
        * participant[
            "top3_rate"
        ]
        + 0.10
        * participant[
            "top5_rate"
        ]
        + 0.15
        * quality_low(
            participant[
                "average_worst_rank"
            ]
        )
        + 0.10
        * quality_low(
            participant[
                "average_mean_rank"
            ]
        )
        + 0.12
        * quality_high(
            participant[
                "pr_auc"
            ]
        )
        + 0.10
        * quality_high(
            participant[
                "roc_auc"
            ]
        )
        + 0.10
        * quality_low(
            participant[
                "brier"
            ]
        )
    )

    results.loc[
        participant.index,
        "selection_score",
    ] = participant[
        "selection_score"
    ]

    return results


# =============================================================================
# CORRELATIONS
# =============================================================================

def safe_corr(
    x,
    y,
):
    x = pd.to_numeric(
        x,
        errors="coerce",
    ).astype(
        "float64"
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    ).astype(
        "float64"
    )

    mask = (
        x.notna()
        & y.notna()
        & np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x.loc[
        mask
    ]

    y = y.loc[
        mask
    ]

    if (
        len(
            x
        ) < 3
        or x.std() == 0
        or y.std() == 0
    ):
        return (
            len(
                x
            ),
            np.nan,
            np.nan,
        )

    pearson = np.corrcoef(
        x.to_numpy(
            dtype=float
        ),
        y.to_numpy(
            dtype=float
        ),
    )[0, 1]

    xr = x.rank(
        method="average"
    )

    yr = y.rank(
        method="average"
    )

    spearman = np.corrcoef(
        xr.to_numpy(
            dtype=float
        ),
        yr.to_numpy(
            dtype=float
        ),
    )[0, 1]

    return (
        len(
            x
        ),
        float(
            pearson
        ),
        float(
            spearman
        ),
    )


def build_correlations(
    data,
):
    historical = data.loc[
        data[
            "season"
        ].between(
            FIRST_SEASON,
            LAST_HISTORICAL_SEASON,
        )
    ].copy()

    rows = []

    for feature in [
        HR_RUN_SHARE,
        HR_RPG,
        NON_HR_RPG,
        RS,
        RD,
        BULLPEN,
        BULLPEN_FF,
        BULLPEN_FASTBALL_FAMILY,
        RAW_AWAY_DEF,
        ADJ_AWAY_DEF,
    ]:
        for target in [
            "won_world_series",
            "reached_world_series",
        ]:
            n, pearson, spearman = safe_corr(
                historical[
                    feature
                ],
                historical[
                    target
                ],
            )

            rows.append(
                {
                    "feature":
                        feature,

                    "target":
                        target,

                    "n":
                        n,

                    "pearson":
                        pearson,

                    "spearman":
                        spearman,
                }
            )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        CORRELATIONS_CSV,
        index=False,
    )

    return out


# =============================================================================
# COEFFICIENTS
# =============================================================================

def build_coefficient_table(
    data,
):
    historical = data.loc[
        data[
            "season"
        ].between(
            FIRST_SEASON,
            LAST_HISTORICAL_SEASON,
        )
    ].copy()

    rows = []

    for target in [
        "won_world_series",
        "reached_world_series",
    ]:
        for model_name, features in (
            MODEL_SPECS.items()
        ):
            model = make_model()

            model.fit(
                historical[
                    features
                ],
                historical[
                    target
                ].astype(
                    int
                ),
            )

            names = (
                model.named_steps[
                    "imputer"
                ]
                .get_feature_names_out(
                    features
                )
            )

            coefs = (
                model.named_steps[
                    "model"
                ]
                .coef_[0]
            )

            for feature, coef in zip(
                names,
                coefs,
            ):
                rows.append(
                    {
                        "target":
                            target,

                        "model":
                            model_name,

                        "feature":
                            feature,

                        "standardized_coefficient":
                            coef,
                    }
                )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        COEFFICIENTS_CSV,
        index=False,
    )

    return out


# =============================================================================
# 2026
# =============================================================================


def score_2026(
    data,
    winner_selected,
    participant_selected,
):
    historical = data.loc[
        data[
            "season"
        ].between(
            FIRST_SEASON,
            LAST_HISTORICAL_SEASON,
        )
    ].copy()

    current = data.loc[
        data[
            "season"
        ].eq(
            CURRENT_SEASON
        )
    ].copy()

    winner_features = str(
        winner_selected[
            "features"
        ]
    ).split(
        " | "
    )

    participant_features = str(
        participant_selected[
            "features"
        ]
    ).split(
        " | "
    )

    winner_model = make_model()
    participant_model = make_model()

    winner_model.fit(
        historical[
            winner_features
        ],
        historical[
            "won_world_series"
        ].astype(
            int
        ),
    )

    participant_model.fit(
        historical[
            participant_features
        ],
        historical[
            "reached_world_series"
        ].astype(
            int
        ),
    )

    current[
        "winner_complete"
    ] = current[
        winner_features
    ].notna().all(
        axis=1
    )

    current[
        "participant_complete"
    ] = current[
        participant_features
    ].notna().all(
        axis=1
    )

    current[
        "winner_missing_features"
    ] = current.apply(
        lambda row: " | ".join(
            [
                feature
                for feature in winner_features
                if pd.isna(
                    row[
                        feature
                    ]
                )
            ]
        ),
        axis=1,
    )

    current[
        "participant_missing_features"
    ] = current.apply(
        lambda row: " | ".join(
            [
                feature
                for feature in participant_features
                if pd.isna(
                    row[
                        feature
                    ]
                )
            ]
        ),
        axis=1,
    )

    # Keep raw model probabilities available for diagnostics, but do not
    # rank incomplete teams.
    current[
        "winner_score"
    ] = winner_model.predict_proba(
        current[
            winner_features
        ]
    )[:, 1]

    current[
        "participant_score"
    ] = participant_model.predict_proba(
        current[
            participant_features
        ]
    )[:, 1]

    current[
        "winner_rank"
    ] = np.nan

    current.loc[
        current[
            "winner_complete"
        ],
        "winner_rank",
    ] = (
        current.loc[
            current[
                "winner_complete"
            ],
            "winner_score",
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    current[
        "participant_rank"
    ] = np.nan

    current.loc[
        current[
            "participant_complete"
        ],
        "participant_rank",
    ] = (
        current.loc[
            current[
                "participant_complete"
            ],
            "participant_score",
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    current[
        "winner_model"
    ] = winner_selected[
        "model"
    ]

    current[
        "participant_model"
    ] = participant_selected[
        "model"
    ]

    current[
        "league"
    ] = current[
        "team"
    ].map(
        team_league
    )

    # Add same-league competition adjusted participant scores.
    complete_for_league = current.loc[
        current[
            "participant_complete"
        ]
    ].copy()

    complete_for_league[
        "score"
    ] = complete_for_league[
        "participant_score"
    ]

    complete_for_league[
        "reached_world_series"
    ] = 0

    complete_for_league[
        "season"
    ] = CURRENT_SEASON

    complete_for_league = add_league_competition_scores(
        complete_for_league
    )

    for col in [
        "league_share",
        "league_softmax_T075",
        "league_softmax_T100",
        "league_softmax_T125",
    ]:
        current[
            col
        ] = np.nan

        current.loc[
            complete_for_league.index,
            col,
        ] = complete_for_league[
            col
        ]

    return current.sort_values(
        [
            "participant_complete",
            "participant_rank",
            "winner_rank",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )



# =============================================================================
# FINAL FIXED 2026 MODEL
# =============================================================================

def score_final_2026_model(data):
    """
    Fit the final selected models on all historical seasons through 2025,
    score all 30 teams for 2026, and apply the fixed league competition
    adjustment selected by the historical backtest.

    No exploratory model selection occurs here.
    """
    historical = data.loc[
        data["season"].between(
            FIRST_SEASON,
            LAST_HISTORICAL_SEASON,
        )
    ].copy()

    current = data.loc[
        data["season"].eq(
            CURRENT_SEASON
        )
    ].copy()

    # ------------------------------------------------------------------
    # Require complete current-season participant inputs.
    # ------------------------------------------------------------------
    current[
        "participant_complete"
    ] = current[
        FINAL_PARTICIPANT_FEATURES
    ].notna().all(
        axis=1
    )

    current[
        "participant_missing_features"
    ] = current.apply(
        lambda row: " | ".join(
            [
                feature
                for feature in FINAL_PARTICIPANT_FEATURES
                if pd.isna(row[feature])
            ]
        ),
        axis=1,
    )

    incomplete = current.loc[
        ~current[
            "participant_complete"
        ],
        [
            "team",
            "participant_missing_features",
        ],
    ].copy()

    if not incomplete.empty:
        raise RuntimeError(
            "Final 2026 model requires all 30 teams to have complete "
            "participant inputs. Missing rows:\\n"
            + incomplete.to_string(index=False)
        )

    if current["team"].nunique() != 30:
        raise RuntimeError(
            f"Expected 30 unique 2026 MLB teams; found "
            f"{current['team'].nunique()}."
        )

    if "ARI" not in set(current["team"]):
        raise RuntimeError(
            "Arizona (ARI) is not present in the final 2026 data."
        )

    # ------------------------------------------------------------------
    # Fit final winner model.
    # ------------------------------------------------------------------
    winner_model = make_model()

    winner_model.fit(
        historical[
            FINAL_WINNER_FEATURES
        ],
        historical[
            "won_world_series"
        ].astype(int),
    )

    current[
        "winner_score"
    ] = winner_model.predict_proba(
        current[
            FINAL_WINNER_FEATURES
        ]
    )[:, 1]

    current[
        "winner_rank"
    ] = (
        current[
            "winner_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Fit final participant model.
    # ------------------------------------------------------------------
    participant_model = make_model()

    participant_model.fit(
        historical[
            FINAL_PARTICIPANT_FEATURES
        ],
        historical[
            "reached_world_series"
        ].astype(int),
    )

    current[
        "participant_raw_score"
    ] = participant_model.predict_proba(
        current[
            FINAL_PARTICIPANT_FEATURES
        ]
    )[:, 1]

    current[
        "league"
    ] = current[
        "team"
    ].map(
        team_league
    )

    # ------------------------------------------------------------------
    # League-softmax normalization, T = 1.25.
    # ------------------------------------------------------------------
    p = current[
        "participant_raw_score"
    ].clip(
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        p / (1 - p)
    )

    z = (
        logits
        / FINAL_LEAGUE_SOFTMAX_T
    )

    max_z = (
        pd.Series(
            z,
            index=current.index,
        )
        .groupby(
            [
                current["season"],
                current["league"],
            ]
        )
        .transform("max")
    )

    exp_z = np.exp(
        z - max_z
    )

    current[
        "participant_league_probability"
    ] = (
        exp_z
        / pd.Series(
            exp_z,
            index=current.index,
        )
        .groupby(
            [
                current["season"],
                current["league"],
            ]
        )
        .transform("sum")
    )

    current[
        "participant_rank"
    ] = (
        current[
            "participant_league_probability"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Sanity checks: exactly 15 AL + 15 NL, each league sums to 1.
    # ------------------------------------------------------------------
    league_counts = (
        current.groupby(
            "league"
        )[
            "team"
        ]
        .nunique()
        .to_dict()
    )

    if league_counts.get("AL") != 15 or league_counts.get("NL") != 15:
        raise RuntimeError(
            f"Expected 15 AL and 15 NL teams; got {league_counts}."
        )

    league_sums = (
        current.groupby(
            "league"
        )[
            "participant_league_probability"
        ]
        .sum()
    )

    for league, total in league_sums.items():
        if not np.isclose(
            total,
            1.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"{league} participant probabilities sum to {total}, "
                "not 1.0."
            )

    return current.sort_values(
        [
            "participant_rank",
            "participant_league_probability",
        ],
        ascending=[
            True,
            False,
        ],
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    data = load_data()

    # Repair any missing Arizona current-season inputs directly from cache.
    data = repair_current_team_from_cached_statcast(
        data,
        team="ARI",
    )

    ari_check = data.loc[
        data["season"].eq(CURRENT_SEASON)
        & data["team"].eq("ARI"),
        [
            BULLPEN,
            RAW_AWAY_DEF,
            NON_HR_RPG,
        ],
    ]

    if ari_check.empty:
        raise RuntimeError(
            "Arizona 2026 row is missing after repair."
        )

    ari_missing = [
        col
        for col in [
            BULLPEN,
            RAW_AWAY_DEF,
            NON_HR_RPG,
        ]
        if pd.isna(
            ari_check.iloc[0][col]
        )
    ]

    if ari_missing:
        raise RuntimeError(
            "Arizona repair still failed for: "
            + ", ".join(ari_missing)
            + "\nLook just above this error for the raw Statcast Arizona team code."
        )

    print("\nArizona repair validation: PASSED")

    # Attach league labels for later competition adjustment.
    data[
        "league"
    ] = data[
        "team"
    ].map(
        team_league
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 145
    )

    print(
        "OFFENSIVE DECOMPOSITION QC"
    )

    print(
        "=" * 145
    )

    print(
        "2026 rows reconstructed from HR RPG / HR share:",
        int(
            (
                data["season"].eq(CURRENT_SEASON)
                & data["runs_scored"].isna()
                & data[RS].notna()
            ).sum()
        ),
    )

    print(
        "2026 rows still missing RS/G:",
        int(
            (
                data["season"].eq(CURRENT_SEASON)
                & data[RS].isna()
            ).sum()
        ),
    )

    qc = (
        data.groupby(
            "season"
        )
        .agg(
            teams=(
                "team",
                "nunique",
            ),

            avg_runs_scored_per_game=(
                RS,
                "mean",
            ),

            avg_hr_runs_per_game=(
                HR_RPG,
                "mean",
            ),

            avg_non_hr_runs_per_game=(
                NON_HR_RPG,
                "mean",
            ),

            avg_hr_run_share=(
                HR_RUN_SHARE,
                "mean",
            ),

            mean_abs_reconstruction_error=(
                "offense_reconstruction_error",
                lambda x:
                    np.nanmean(
                        np.abs(
                            x
                        )
                    ),
            ),

            max_abs_reconstruction_error=(
                "offense_reconstruction_error",
                lambda x:
                    np.nanmax(
                        np.abs(
                            x
                        )
                    ),
            ),

            mean_abs_rd_reconstruction_error=(
                "rd_reconstruction_error",
                lambda x:
                    np.nanmean(
                        np.abs(
                            x
                        )
                    ),
            ),
        )
        .reset_index()
    )

    print(
        qc.to_string(
            index=False,
            formatters={
                "avg_runs_scored_per_game":
                    lambda x:
                        f"{x:.3f}",

                "avg_hr_runs_per_game":
                    lambda x:
                        f"{x:.3f}",

                "avg_non_hr_runs_per_game":
                    lambda x:
                        f"{x:.3f}",

                "avg_hr_run_share":
                    lambda x:
                        f"{x:.1%}",

                "mean_abs_reconstruction_error":
                    lambda x:
                        f"{x:.6f}",

                "max_abs_reconstruction_error":
                    lambda x:
                        f"{x:.6f}",

                "mean_abs_rd_reconstruction_error":
                    lambda x:
                        f"{x:.6f}",
            },
        )
    )


    print("\n" + "=" * 145)
    print("BULLPEN VELOCITY DEFINITION QC")
    print("=" * 145)

    velocity_qc = (
        data.groupby("season")
        .agg(
            teams_all_pitch=(BULLPEN, "count"),
            teams_four_seam=(BULLPEN_FF, "count"),
            teams_fastball_family=(BULLPEN_FASTBALL_FAMILY, "count"),
            mean_all_pitch=(BULLPEN, "mean"),
            mean_four_seam=(BULLPEN_FF, "mean"),
            mean_fastball_family=(BULLPEN_FASTBALL_FAMILY, "mean"),
        )
        .reset_index()
    )

    print(
        velocity_qc.to_string(
            index=False,
            formatters={
                "mean_all_pitch": lambda x: f"{x:.2f}",
                "mean_four_seam": lambda x: f"{x:.2f}",
                "mean_fastball_family": lambda x: f"{x:.2f}",
            },
        )
    )

    hist_velo = data.loc[
        data["season"].between(FIRST_SEASON, LAST_HISTORICAL_SEASON),
        [BULLPEN, BULLPEN_FF, BULLPEN_FASTBALL_FAMILY],
    ].copy()

    print("\nHistorical correlations among velocity definitions:")
    print(hist_velo.corr().round(3).to_string())

    # -------------------------------------------------------------------------
    # CORRELATIONS
    # -------------------------------------------------------------------------

    correlations = build_correlations(
        data
    )

    print(
        "\n"
        + "=" * 145
    )

    print(
        "SIMPLE CORRELATIONS — HR VS NON-HR SCORING"
    )

    print(
        "=" * 145
    )

    print(
        correlations.to_string(
            index=False,
            formatters={
                "pearson":
                    lambda x:
                        f"{x:+.4f}",

                "spearman":
                    lambda x:
                        f"{x:+.4f}",
            },
        )
    )

    # -------------------------------------------------------------------------
    # BACKTEST
    # -------------------------------------------------------------------------

    summaries = []
    season_parts = []
    oof_prediction_parts = []

    for target in [
        "won_world_series",
        "reached_world_series",
    ]:
        for model_name, features in (
            MODEL_SPECS.items()
        ):
            print(
                f"\nBacktesting "
                f"{target}: "
                f"{model_name}"
            )

            (
                summary,
                seasons,
                predictions,
            ) = evaluate_model(
                data=data,
                model_name=
                    model_name,
                features=
                    features,
                target=
                    target,
            )

            summaries.append(
                summary
            )

            season_parts.append(
                seasons
            )

            predictions = predictions.copy()
            predictions[
                "model"
            ] = model_name

            oof_prediction_parts.append(
                predictions
            )

    results = pd.DataFrame(
        summaries
    )

    results = add_selection_scores(
        results
    )

    season_detail = pd.concat(
        season_parts,
        ignore_index=True,
    )

    results.to_csv(
        MODEL_RESULTS_CSV,
        index=False,
    )

    season_detail.to_csv(
        SEASON_DETAIL_CSV,
        index=False,
    )

    winner = (
        results.loc[
            results[
                "target"
            ].eq(
                "won_world_series"
            )
        ]
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    participant = (
        results.loc[
            results[
                "target"
            ].eq(
                "reached_world_series"
            )
        ]
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # PRINT WINNER
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 190
    )

    print(
        "WINNER MODEL — HR VS NON-HR SCORING"
    )

    print(
        "=" * 190
    )

    print(
        winner[
            [
                "model",
                "selection_score",
                "top3_rate",
                "top5_rate",
                "average_mean_rank",
                "worst_observed_rank",
                "pr_auc",
                "roc_auc",
                "brier",
            ]
        ].to_string(
            index=False,
            formatters={
                "selection_score":
                    lambda x:
                        f"{x:.3f}",

                "top3_rate":
                    lambda x:
                        f"{x:.1%}",

                "top5_rate":
                    lambda x:
                        f"{x:.1%}",

                "average_mean_rank":
                    lambda x:
                        f"{x:.2f}",

                "worst_observed_rank":
                    lambda x:
                        f"{x:.0f}",

                "pr_auc":
                    lambda x:
                        f"{x:.3f}",

                "roc_auc":
                    lambda x:
                        f"{x:.3f}",

                "brier":
                    lambda x:
                        f"{x:.3f}",
            },
        )
    )

    # -------------------------------------------------------------------------
    # PRINT PARTICIPANT
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 200
    )

    print(
        "WORLD SERIES PARTICIPANT MODEL — HR VS NON-HR SCORING"
    )

    print(
        "=" * 200
    )

    print(
        participant[
            [
                "model",
                "selection_score",
                "both_finalists_top5_rate",
                "top3_rate",
                "top5_rate",
                "average_worst_rank",
                "average_mean_rank",
                "worst_observed_rank",
                "pr_auc",
                "roc_auc",
                "brier",
            ]
        ].to_string(
            index=False,
            formatters={
                "selection_score":
                    lambda x:
                        f"{x:.3f}",

                "both_finalists_top5_rate":
                    lambda x:
                        f"{x:.1%}",

                "top3_rate":
                    lambda x:
                        f"{x:.1%}",

                "top5_rate":
                    lambda x:
                        f"{x:.1%}",

                "average_worst_rank":
                    lambda x:
                        f"{x:.2f}",

                "average_mean_rank":
                    lambda x:
                        f"{x:.2f}",

                "worst_observed_rank":
                    lambda x:
                        f"{x:.0f}",

                "pr_auc":
                    lambda x:
                        f"{x:.3f}",

                "roc_auc":
                    lambda x:
                        f"{x:.3f}",

                "brier":
                    lambda x:
                        f"{x:.3f}",
            },
        )
    )


    # -------------------------------------------------------------------------
    # LEAGUE-COMPETITION ADJUSTMENT TEST
    # -------------------------------------------------------------------------

    selected_participant_name = (
        participant.iloc[
            0
        ][
            "model"
        ]
    )

    all_oof_predictions = pd.concat(
        oof_prediction_parts,
        ignore_index=True,
    )

    participant_oof = all_oof_predictions.loc[
        (
            all_oof_predictions[
                "target_name"
            ].eq(
                "reached_world_series"
            )
        )
        & (
            all_oof_predictions[
                "model"
            ].eq(
                selected_participant_name
            )
        )
    ].copy()

    participant_oof[
        "league"
    ] = participant_oof[
        "team"
    ].map(
        team_league
    )

    participant_oof = add_league_competition_scores(
        participant_oof
    )

    league_rows = []

    for score_col in [
        "raw_probability",
        "league_share",
        "league_softmax_T075",
        "league_softmax_T100",
        "league_softmax_T125",
    ]:
        league_rows.append(
            summarize_participant_score_variant(
                participant_oof,
                score_col,
            )
        )

    league_results = pd.DataFrame(
        league_rows
    )

    print(
        "\n"
        + "=" * 190
    )

    print(
        "WORLD SERIES PARTICIPANT — SAME-LEAGUE COMPETITION ADJUSTMENT TEST"
    )

    print(
        "=" * 190
    )

    print(
        league_results.to_string(
            index=False,
            formatters={
                "both_finalists_top5_rate":
                    lambda x:
                        f"{x:.1%}",

                "top3_rate":
                    lambda x:
                        f"{x:.1%}",

                "top5_rate":
                    lambda x:
                        f"{x:.1%}",

                "average_worst_rank":
                    lambda x:
                        f"{x:.2f}",

                "average_mean_rank":
                    lambda x:
                        f"{x:.2f}",

                "pr_auc":
                    lambda x:
                        f"{x:.3f}",

                "roc_auc":
                    lambda x:
                        f"{x:.3f}",

                "brier":
                    lambda x:
                        f"{x:.3f}",
            },
        )
    )

    league_results.to_csv(
        ROOT
        / "aug7_participant_league_competition_backtest.csv",
        index=False,
    )

    # Choose best league-adjustment variant with the same participant ranking
    # priorities used elsewhere, but do not force an adjustment if raw wins.
    temp = league_results.copy()

    temp[
        "selection_score"
    ] = (
        0.20
        * temp[
            "both_finalists_top5_rate"
        ]
        + 0.13
        * temp[
            "top3_rate"
        ]
        + 0.10
        * temp[
            "top5_rate"
        ]
        + 0.15
        * quality_low(
            temp[
                "average_worst_rank"
            ]
        )
        + 0.10
        * quality_low(
            temp[
                "average_mean_rank"
            ]
        )
        + 0.12
        * quality_high(
            temp[
                "pr_auc"
            ]
        )
        + 0.10
        * quality_high(
            temp[
                "roc_auc"
            ]
        )
        + 0.10
        * quality_low(
            temp[
                "brier"
            ]
        )
    )

    temp = temp.sort_values(
        "selection_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    selected_league_variant = temp.iloc[
        0
    ][
        "score_variant"
    ]

    print(
        "\nSelected league-competition score variant:",
        selected_league_variant,
    )

    # -------------------------------------------------------------------------
    # COEFFICIENTS
    # -------------------------------------------------------------------------

    coefficients = (
        build_coefficient_table(
            data
        )
    )

    interesting_features = {
        RD,
        RS,
        RA,
        BULLPEN,
        BULLPEN_FF,
        BULLPEN_FASTBALL_FAMILY,
        HR_RUN_SHARE,
        HR_RPG,
        NON_HR_RPG,
        RAW_AWAY_DEF,
        ADJ_AWAY_DEF,
        RAW_ALL_DEF,
        ADJ_ALL_DEF,
    }

    interesting = (
        coefficients.loc[
            coefficients[
                "feature"
            ].isin(
                interesting_features
            )
        ]
    )

    print(
        "\n"
        + "=" * 210
    )

    print(
        "STANDARDIZED COEFFICIENTS — OFFENSIVE DECOMPOSITION MODELS"
    )

    print(
        "=" * 210
    )

    print(
        interesting.to_string(
            index=False,
            formatters={
                "standardized_coefficient":
                    lambda x:
                        f"{x:+.4f}",
            },
        )
    )

    # -------------------------------------------------------------------------
    # SELECT MODELS
    # -------------------------------------------------------------------------

    winner_selected = winner.iloc[
        0
    ]

    participant_selected = (
        participant.iloc[
            0
        ]
    )

    print(
        "\n"
        + "=" * 130
    )

    print(
        "SELECTED MODELS"
    )

    print(
        "=" * 130
    )

    print(
        "\nWINNER:"
    )

    print(
        winner_selected[
            "model"
        ]
    )

    print(
        winner_selected[
            "features"
        ]
    )

    print(
        "\nPARTICIPANT:"
    )

    print(
        participant_selected[
            "model"
        ]
    )

    print(
        participant_selected[
            "features"
        ]
    )

    # -------------------------------------------------------------------------
    # 2026
    # -------------------------------------------------------------------------

    current = score_2026(
        data,
        winner_selected,
        participant_selected,
    )

    current.to_csv(
        CURRENT_2026_CSV,
        index=False,
    )

    print(
        "\n"
        + "=" * 220
    )

    print(
        "AUGUST 7, 2026 — HR VS NON-HR SCORING MODEL"
    )

    print(
        "=" * 220
    )

    print(
        current[
            [
                "winner_rank",
                "team",
                "winner_score",
                "participant_rank",
                "participant_score",
                RD,
                RS,
                RA,
                HR_RPG,
                NON_HR_RPG,
                HR_RUN_SHARE,
                BULLPEN,
                BULLPEN_FF,
                BULLPEN_FASTBALL_FAMILY,
                RAW_AWAY_DEF,
                ADJ_AWAY_DEF,
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False,
            formatters={
                "winner_score":
                    lambda x:
                        f"{x:.3f}",

                "participant_score":
                    lambda x:
                        f"{x:.3f}",

                RD:
                    lambda x:
                        f"{x:+.3f}",

                RS:
                    lambda x:
                        f"{x:.3f}",

                RA:
                    lambda x:
                        f"{x:.3f}",

                HR_RPG:
                    lambda x:
                        f"{x:.3f}",

                NON_HR_RPG:
                    lambda x:
                        f"{x:.3f}",

                HR_RUN_SHARE:
                    lambda x:
                        f"{x:.1%}",

                BULLPEN:
                    lambda x:
                        f"{x:.2f}",

                BULLPEN_FF:
                    lambda x:
                        f"{x:.2f}",

                BULLPEN_FASTBALL_FAMILY:
                    lambda x:
                        f"{x:.2f}",

                RAW_AWAY_DEF:
                    lambda x:
                        f"{x:+.3f}",

                ADJ_AWAY_DEF:
                    lambda x:
                        f"{x:+.3f}",
            },
        )
    )




    # -------------------------------------------------------------------------
    # FULL 2026 WORLD SERIES PARTICIPANT RANKING
    # -------------------------------------------------------------------------

    complete_participants = current.loc[
        current[
            "participant_complete"
        ]
    ].copy()

    # Use the historically selected league-adjusted score variant.
    if selected_league_variant == "raw_probability":
        complete_participants[
            "final_participant_score"
        ] = complete_participants[
            "participant_score"
        ]
    else:
        complete_participants[
            "final_participant_score"
        ] = complete_participants[
            selected_league_variant
        ]

    complete_participants[
        "final_participant_rank"
    ] = (
        complete_participants[
            "final_participant_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    full_rank = complete_participants.sort_values(
        [
            "final_participant_rank",
            "final_participant_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).copy()

    print("\n" + "=" * 190)
    print("2026 WORLD SERIES PARTICIPANT RANKING — ALL 30 TEAMS / LEAGUE-NORMALIZED")
    print("=" * 190)

    print("Base participant model:")
    print(participant_selected["model"])
    print(participant_selected["features"])

    print(
        "\nLeague-competition adjustment used:",
        selected_league_variant,
    )

    participant_columns = [
        "final_participant_rank",
        "team",
        "league",
        "final_participant_score",
        "participant_score",
        RD,
        BULLPEN,
        RAW_AWAY_DEF,
        NON_HR_RPG,
        HR_RUN_SHARE,
    ]

    print(
        full_rank[
            participant_columns
        ].to_string(
            index=False,
            formatters={
                "final_participant_rank":
                    lambda x:
                        f"{int(x)}",

                "final_participant_score":
                    lambda x:
                        f"{x:.3f}",

                "participant_score":
                    lambda x:
                        f"{x:.3f}",

                RD:
                    lambda x:
                        f"{x:+.3f}",

                BULLPEN:
                    lambda x:
                        f"{x:.2f}",

                RAW_AWAY_DEF:
                    lambda x:
                        f"{x:+.3f}",

                NON_HR_RPG:
                    lambda x:
                        f"{x:.3f}",

                HR_RUN_SHARE:
                    lambda x:
                        f"{x:.1%}",
            },
        )
    )

    incomplete = current.loc[
        ~current[
            "participant_complete"
        ],
        [
            "team",
            "league",
            "participant_missing_features",
        ],
    ].copy()

    if not incomplete.empty:
        print(
            "\nWARNING — incomplete teams still excluded:"
        )

        print(
            incomplete.to_string(
                index=False
            )
        )

    full30_path = (
        ROOT
        / "aug7_2026_all30_world_series_participant_odds.csv"
    )

    full_rank[
        participant_columns
    ].to_csv(
        full30_path,
        index=False,
    )

    print(
        f"\nFull 2026 participant ranking saved to:\n{full30_path}"
    )


    # -------------------------------------------------------------------------
    # FINAL FIXED MODEL OUTPUT
    # -------------------------------------------------------------------------

    final_2026 = score_final_2026_model(
        data
    )

    print(
        "\n"
        + "=" * 210
    )

    print(
        "FINAL 2026 WORLD SERIES PARTICIPANT MODEL — ALL 30 TEAMS"
    )

    print(
        "=" * 210
    )

    print(
        "Participant features:"
    )

    print(
        " | ".join(
            FINAL_PARTICIPANT_FEATURES
        )
    )

    print(
        f"League competition adjustment: softmax T="
        f"{FINAL_LEAGUE_SOFTMAX_T}"
    )

    print(
        "\nAL participant probability sum:",
        f"{final_2026.loc[final_2026['league'].eq('AL'), 'participant_league_probability'].sum():.6f}",
    )

    print(
        "NL participant probability sum:",
        f"{final_2026.loc[final_2026['league'].eq('NL'), 'participant_league_probability'].sum():.6f}",
    )

    final_columns = [
        "participant_rank",
        "team",
        "league",
        "participant_league_probability",
        "participant_raw_score",
        RD,
        BULLPEN,
        RAW_AWAY_DEF,
        NON_HR_RPG,
        HR_RUN_SHARE,
        "winner_rank",
        "winner_score",
    ]

    print(
        final_2026[
            final_columns
        ].to_string(
            index=False,
            formatters={
                "participant_league_probability":
                    lambda x: f"{x:.1%}",

                "participant_raw_score":
                    lambda x: f"{x:.3f}",

                RD:
                    lambda x: f"{x:+.3f}",

                BULLPEN:
                    lambda x: f"{x:.2f}",

                RAW_AWAY_DEF:
                    lambda x: f"{x:+.3f}",

                NON_HR_RPG:
                    lambda x: f"{x:.3f}",

                HR_RUN_SHARE:
                    lambda x: f"{x:.1%}",

                "winner_score":
                    lambda x: f"{x:.3f}",
            },
        )
    )

    final_out = (
        ROOT
        / "final_2026_world_series_participant_model.csv"
    )

    final_2026[
        final_columns
    ].to_csv(
        final_out,
        index=False,
    )

    print(
        f"\nFinal 30-team CSV saved to:\n{final_out}"
    )

    print(
        "\nFILES CREATED:"
    )

    for path in [
        MODEL_RESULTS_CSV,
        SEASON_DETAIL_CSV,
        COEFFICIENTS_CSV,
        CORRELATIONS_CSV,
        CURRENT_2026_CSV,
    ]:
        print(
            path
        )


if __name__ == "__main__":
    main()
