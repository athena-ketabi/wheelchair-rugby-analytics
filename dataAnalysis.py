from __future__ import annotations

import pandas as pd
import numpy as np

# Optional: read the docx data dictionary
from docx import Document


# -----------------------------
# Paths (adjust if needed)
# -----------------------------
STINT_PATH = "./stint_data.csv"
PLAYER_PATH = "./player_data.csv"
DICT_PATH  = "./data_dictionary.docx"


# -----------------------------
# Step A: Load data
# -----------------------------
stints = pd.read_csv(STINT_PATH)
players = pd.read_csv(PLAYER_PATH)

print("Loaded:")
print("  stints:", stints.shape, "rows x cols")
print("  players:", players.shape, "rows x cols")
print()


# -----------------------------
# Step B: (Optional) Show data dictionary text
# -----------------------------
def print_docx_text(path: str, max_paragraphs: int = 50) -> None:
    """
    Prints the first N paragraphs from a .docx.
    Useful for seeing definitions/terms from the data dictionary.
    """
    try:
        doc = Document(path)
    except Exception as e:
        print("Could not open data dictionary:", e)
        return

    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    print("=== Data Dictionary (first paragraphs) ===")
    for i, t in enumerate(paras[:max_paragraphs], start=1):
        print(f"{i:02d}. {t}")
    if len(paras) > max_paragraphs:
        print(f"... ({len(paras) - max_paragraphs} more paragraphs not shown)")
    print()

print_docx_text(DICT_PATH, max_paragraphs=30)


# -----------------------------
# Step C: Basic sanity checks
# -----------------------------
required_cols = [
    "game_id", "h_team", "a_team", "minutes", "h_goals", "a_goals",
    "home1", "home2", "home3", "home4",
    "away1", "away2", "away3", "away4"
]
missing = [c for c in required_cols if c not in stints.columns]
if missing:
    raise ValueError(f"stint_data.csv is missing columns: {missing}")

if "player" not in players.columns or "rating" not in players.columns:
    raise ValueError("player_data.csv should have columns: player, rating")


# -----------------------------
# Step D: Teams list
# -----------------------------
teams = sorted(set(stints["h_team"]).union(set(stints["a_team"])))
print("=== Teams ===")
print("Number of teams:", len(teams))
print("Teams:", teams)
print()


# -----------------------------
# Step E: Build GAME-LEVEL table from STINTS
# Each game_id has many stints, we sum them into one game result.
# -----------------------------
games = (
    stints.groupby(["game_id", "h_team", "a_team"], as_index=False)
          .agg(
              total_minutes=("minutes", "sum"),
              home_goals=("h_goals", "sum"),
              away_goals=("a_goals", "sum"),
          )
)

# Determine winner
games["winner"] = np.where(
    games["home_goals"] > games["away_goals"], games["h_team"],
    np.where(games["away_goals"] > games["home_goals"], games["a_team"], "TIE")
)

games["home_win"] = (games["winner"] == games["h_team"]).astype(int)
games["away_win"] = (games["winner"] == games["a_team"]).astype(int)
games["tie"]      = (games["winner"] == "TIE").astype(int)

print("=== Games summary ===")
print("Number of games:", len(games))
print(games.head(5))
print()


# -----------------------------
# Step F: Team win rates (simple)
# Count wins for each team / games played
# -----------------------------
home_stats = games.groupby("h_team").agg(
    home_games=("game_id", "count"),
    home_wins=("home_win", "sum"),
    home_ties=("tie", "sum"),
)
away_stats = games.groupby("a_team").agg(
    away_games=("game_id", "count"),
    away_wins=("away_win", "sum"),
    away_ties=("tie", "sum"),
)

team_summary = (
    home_stats.join(away_stats, how="outer")
              .fillna(0)
              .assign(
                  games_played=lambda d: d["home_games"] + d["away_games"],
                  wins=lambda d: d["home_wins"] + d["away_wins"],
                  ties=lambda d: d["home_ties"] + d["away_ties"],
                  win_rate=lambda d: np.where(d["games_played"] > 0, d["wins"] / d["games_played"], np.nan),
              )
              .sort_values("win_rate", ascending=False)
)

print("=== Team win rates ===")
print(team_summary[["games_played", "wins", "ties", "win_rate"]].round(3))
print()


# -----------------------------
# Step G: Home advantage
# -----------------------------
home_win_rate = games["home_win"].mean()
print("=== Home advantage ===")
print(f"Home win rate: {home_win_rate:.3f} ({home_win_rate*100:.1f}%)")
print()


# -----------------------------
# Step H: "Overtime-ish" games
# Many wheelchair rugby games are ~32 minutes.
# If total minutes is much more than 32, likely overtime or unusual timing.
# We'll list games with total_minutes > 32.1 (small buffer).
# -----------------------------
overtime_like = games.loc[games["total_minutes"] > 32.1].sort_values("total_minutes", ascending=False)
print("=== Games with total_minutes > 32.1 (overtime-ish) ===")
print(overtime_like[["game_id", "h_team", "a_team", "total_minutes", "home_goals", "away_goals"]].head(20))
print()


# -----------------------------
# Step I: Canada head-to-head results vs each opponent
# -----------------------------
def canada_head_to_head(games_df: pd.DataFrame, canada_name: str = "Canada") -> pd.DataFrame:
    g = games_df.copy()
    # figure out which games include Canada
    mask = (g["h_team"] == canada_name) | (g["a_team"] == canada_name)
    g = g.loc[mask].copy()

    # Opponent column
    g["opponent"] = np.where(g["h_team"] == canada_name, g["a_team"], g["h_team"])

    # Canada score & opponent score
    g["canada_goals"] = np.where(g["h_team"] == canada_name, g["home_goals"], g["away_goals"])
    g["opp_goals"]    = np.where(g["h_team"] == canada_name, g["away_goals"], g["home_goals"])

    # Result
    g["canada_win"] = (g["canada_goals"] > g["opp_goals"]).astype(int)
    g["canada_loss"] = (g["canada_goals"] < g["opp_goals"]).astype(int)
    g["canada_tie"] = (g["canada_goals"] == g["opp_goals"]).astype(int)

    summary = (
        g.groupby("opponent", as_index=False)
         .agg(
             games=("game_id", "count"),
             wins=("canada_win", "sum"),
             losses=("canada_loss", "sum"),
             ties=("canada_tie", "sum"),
             canada_goals=("canada_goals", "sum"),
             opp_goals=("opp_goals", "sum"),
         )
    )
    summary["goal_diff"] = summary["canada_goals"] - summary["opp_goals"]
    summary["win_rate"] = np.where(summary["games"] > 0, summary["wins"] / summary["games"], np.nan)
    return summary.sort_values(["win_rate", "goal_diff"], ascending=False)

h2h = canada_head_to_head(games, "Canada")
print("=== Canada head-to-head ===")
print(h2h.round(3))
print()


# -----------------------------
# Step J: Player impact for Canada using STINTS
# We'll compute: minutes played, goals_for, goals_against, goal_diff, diff_per_10_min
# from stints where the player is on court in Canada stints.
# -----------------------------
HOME_PLAYER_COLS = ["home1", "home2", "home3", "home4"]
AWAY_PLAYER_COLS = ["away1", "away2", "away3", "away4"]

def canada_player_impact(stints_df: pd.DataFrame, canada_name: str = "Canada") -> pd.DataFrame:
    # Separate stints where Canada is home vs away, so we know "goals for/against"
    home = stints_df.loc[stints_df["h_team"] == canada_name].copy()
    away = stints_df.loc[stints_df["a_team"] == canada_name].copy()

    # For stints where Canada is home:
    # goals_for = h_goals, goals_against = a_goals, players = home1..home4
    home_long = home.melt(
        id_vars=["minutes", "h_goals", "a_goals"],
        value_vars=HOME_PLAYER_COLS,
        var_name="slot",
        value_name="player",
    ).dropna(subset=["player"])

    home_long["goals_for"] = home_long["h_goals"]
    home_long["goals_against"] = home_long["a_goals"]

    # For stints where Canada is away:
    # goals_for = a_goals, goals_against = h_goals, players = away1..away4
    away_long = away.melt(
        id_vars=["minutes", "h_goals", "a_goals"],
        value_vars=AWAY_PLAYER_COLS,
        var_name="slot",
        value_name="player",
    ).dropna(subset=["player"])

    away_long["goals_for"] = away_long["a_goals"]
    away_long["goals_against"] = away_long["h_goals"]

    long = pd.concat([home_long[["player", "minutes", "goals_for", "goals_against"]],
                      away_long[["player", "minutes", "goals_for", "goals_against"]]],
                     ignore_index=True)

    impact = (
        long.groupby("player", as_index=False)
            .agg(
                minutes=("minutes", "sum"),
                goals_for=("goals_for", "sum"),
                goals_against=("goals_against", "sum"),
            )
    )
    impact["goal_diff"] = impact["goals_for"] - impact["goals_against"]
    impact["diff_per_min"] = np.where(impact["minutes"] > 0, impact["goal_diff"] / impact["minutes"], np.nan)
    impact["diff_per_10_min"] = impact["diff_per_min"] * 10

    # Attach rating if available
    impact = impact.merge(players, on="player", how="left")

    return impact.sort_values("diff_per_10_min", ascending=False)

impact = canada_player_impact(stints, "Canada")

print("=== Canada player impact (top) ===")
print(impact.head(15).round(3))
print()

print("=== Canada player impact (bottom) ===")
print(impact.tail(15).round(3))
print()


# Optional: filter to "meaningful minutes" to avoid tiny samples
MIN_MINUTES = 30  # you can change this
impact_big = impact.loc[impact["minutes"] >= MIN_MINUTES].copy()

print(f"=== Canada player impact with minutes >= {MIN_MINUTES} ===")
print(impact_big[["player", "rating", "minutes", "goal_diff", "diff_per_10_min"]].round(3).head(20))
print()


# -----------------------------
# Step K: Canada 4-player lineup performance
# Each stint has 4 Canada players. We'll treat a lineup as an unordered set of 4.
# Compute minutes, goals_for, goals_against, goal_diff, diff_per_10_min.
# -----------------------------
def canada_lineup_performance(stints_df: pd.DataFrame, canada_name: str = "Canada") -> pd.DataFrame:
    home = stints_df.loc[stints_df["h_team"] == canada_name].copy()
    away = stints_df.loc[stints_df["a_team"] == canada_name].copy()

    def lineup_key(row: pd.Series, cols: list[str]) -> str:
        # Sort names so same lineup in different order counts as same lineup
        return "|".join(sorted([row[c] for c in cols]))

    if len(home) > 0:
        home["lineup"] = home.apply(lambda r: lineup_key(r, HOME_PLAYER_COLS), axis=1)
        home["goals_for"] = home["h_goals"]
        home["goals_against"] = home["a_goals"]

    if len(away) > 0:
        away["lineup"] = away.apply(lambda r: lineup_key(r, AWAY_PLAYER_COLS), axis=1)
        away["goals_for"] = away["a_goals"]
        away["goals_against"] = away["h_goals"]

    combined = pd.concat(
        [
            home[["lineup", "minutes", "goals_for", "goals_against"]],
            away[["lineup", "minutes", "goals_for", "goals_against"]],
        ],
        ignore_index=True,
    )

    perf = (
        combined.groupby("lineup", as_index=False)
                .agg(
                    minutes=("minutes", "sum"),
                    goals_for=("goals_for", "sum"),
                    goals_against=("goals_against", "sum"),
                )
    )
    perf["goal_diff"] = perf["goals_for"] - perf["goals_against"]
    perf["diff_per_min"] = np.where(perf["minutes"] > 0, perf["goal_diff"] / perf["minutes"], np.nan)
    perf["diff_per_10_min"] = perf["diff_per_min"] * 10

    return perf.sort_values("diff_per_10_min", ascending=False)

lineups = canada_lineup_performance(stints, "Canada")

# Filter lineups by minimum minutes so the "best" isn't just 1 small stint
MIN_LINEUP_MINUTES = 10
lineups_big = lineups.loc[lineups["minutes"] >= MIN_LINEUP_MINUTES].copy()

print(f"=== Canada lineup performance (minutes >= {MIN_LINEUP_MINUTES}) ===")
print(lineups_big.head(15).round(3))
print()


# -----------------------------
# Step L: Final "who is most likely to win?"
# In simple terms: win_rate ranking from team_summary
# -----------------------------
print("=== Likely strongest teams (by win_rate) ===")
ranked = team_summary[["games_played", "wins", "win_rate"]].copy()
ranked["win_rate"] = ranked["win_rate"].round(3)
print(ranked.head(12))
print()

print("Done.")
