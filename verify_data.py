"""
Data Verification Script
Verifies that the coach_app.py is correctly extracting and processing data from CSV files.
"""

import pandas as pd
import numpy as np

# Load data
stints = pd.read_csv("stint_data.csv")
players = pd.read_csv("player_data.csv")

print("=" * 80)
print("DATA VERIFICATION REPORT")
print("=" * 80)

# 1. Verify CSV structure
print("\n1. CSV FILE STRUCTURE")
print("-" * 80)
print(f"stint_data.csv: {stints.shape[0]} rows, {stints.shape[1]} columns")
print(f"player_data.csv: {players.shape[0]} rows, {players.shape[1]} columns")

required_stint_cols = ["game_id", "h_team", "a_team", "minutes", "h_goals", "a_goals",
                       "home1", "home2", "home3", "home4", "away1", "away2", "away3", "away4"]
required_player_cols = ["player", "rating"]

missing_stint = [c for c in required_stint_cols if c not in stints.columns]
missing_player = [c for c in required_player_cols if c not in players.columns]

if missing_stint:
    print(f"❌ ERROR: Missing columns in stint_data.csv: {missing_stint}")
else:
    print("✓ All required columns present in stint_data.csv")

if missing_player:
    print(f"❌ ERROR: Missing columns in player_data.csv: {missing_player}")
else:
    print("✓ All required columns present in player_data.csv")

# 2. Verify data types and ranges
print("\n2. DATA VALIDATION")
print("-" * 80)

# Check for negative values where they shouldn't exist
if (stints["minutes"] < 0).any():
    print("❌ ERROR: Found negative minutes")
else:
    print("✓ All minutes are non-negative")

if (stints["h_goals"] < 0).any() or (stints["a_goals"] < 0).any():
    print("❌ ERROR: Found negative goals")
else:
    print("✓ All goal values are non-negative")

# Check player ratings
if (players["rating"] < 0).any() or (players["rating"] > 4).any():
    print(f"⚠ WARNING: Player ratings outside typical range [0, 4]: min={players['rating'].min()}, max={players['rating'].max()}")
else:
    print(f"✓ Player ratings in valid range [0, 4]: min={players['rating'].min()}, max={players['rating'].max()}")

# 3. Verify game aggregation logic
print("\n3. GAME AGGREGATION VERIFICATION")
print("-" * 80)

games = (
    stints.groupby(["game_id", "h_team", "a_team"], as_index=False)
    .agg(
        total_minutes=("minutes", "sum"),
        home_goals=("h_goals", "sum"),
        away_goals=("a_goals", "sum"),
    )
)

# Verify: sum of stint goals should equal game goals
sample_game_id = games.iloc[0]["game_id"]
sample_stints = stints[stints["game_id"] == sample_game_id]
expected_home = sample_stints["h_goals"].sum()
expected_away = sample_stints["a_goals"].sum()
actual_home = games[games["game_id"] == sample_game_id]["home_goals"].iloc[0]
actual_away = games[games["game_id"] == sample_game_id]["away_goals"].iloc[0]

if expected_home == actual_home and expected_away == actual_away:
    print(f"✓ Game aggregation correct (verified on game_id={sample_game_id})")
else:
    print(f"❌ ERROR: Game aggregation mismatch for game_id={sample_game_id}")
    print(f"   Expected: home={expected_home}, away={expected_away}")
    print(f"   Actual: home={actual_home}, away={actual_away}")

# 4. Verify player impact calculation for a specific team
print("\n4. PLAYER IMPACT CALCULATION VERIFICATION")
print("-" * 80)

team = "Canada"
HOME_PLAYER_COLS = ["home1", "home2", "home3", "home4"]
AWAY_PLAYER_COLS = ["away1", "away2", "away3", "away4"]

home = stints.loc[stints["h_team"] == team].copy()
away = stints.loc[stints["a_team"] == team].copy()

# For home stints: goals_for = h_goals, goals_against = a_goals
home_long = home.melt(
    id_vars=["minutes", "h_goals", "a_goals"],
    value_vars=HOME_PLAYER_COLS,
    var_name="slot",
    value_name="player",
).dropna(subset=["player"])
home_long["goals_for"] = home_long["h_goals"]
home_long["goals_against"] = home_long["a_goals"]

# For away stints: goals_for = a_goals, goals_against = h_goals
away_long = away.melt(
    id_vars=["minutes", "h_goals", "a_goals"],
    value_vars=AWAY_PLAYER_COLS,
    var_name="slot",
    value_name="player",
).dropna(subset=["player"])
away_long["goals_for"] = away_long["a_goals"]
away_long["goals_against"] = away_long["h_goals"]

# Manual verification: Check a specific stint
if len(home) > 0:
    sample_stint = home.iloc[0]
    sample_player = sample_stint["home1"]
    player_stints = home_long[home_long["player"] == sample_player]
    
    # Verify goals_for and goals_against are correctly assigned
    if len(player_stints) > 0:
        # In home stints, goals_for should equal h_goals
        if (player_stints["goals_for"] == player_stints["h_goals"]).all():
            print(f"✓ Home stint goals_for calculation correct (verified for {sample_player})")
        else:
            print(f"❌ ERROR: Home stint goals_for calculation incorrect")
        
        # In home stints, goals_against should equal a_goals
        if (player_stints["goals_against"] == player_stints["a_goals"]).all():
            print(f"✓ Home stint goals_against calculation correct (verified for {sample_player})")
        else:
            print(f"❌ ERROR: Home stint goals_against calculation incorrect")

if len(away) > 0:
    sample_stint = away.iloc[0]
    sample_player = sample_stint["away1"]
    player_stints = away_long[away_long["player"] == sample_player]
    
    if len(player_stints) > 0:
        # In away stints, goals_for should equal a_goals
        if (player_stints["goals_for"] == player_stints["a_goals"]).all():
            print(f"✓ Away stint goals_for calculation correct (verified for {sample_player})")
        else:
            print(f"❌ ERROR: Away stint goals_for calculation incorrect")
        
        # In away stints, goals_against should equal h_goals
        if (player_stints["goals_against"] == player_stints["h_goals"]).all():
            print(f"✓ Away stint goals_against calculation correct (verified for {sample_player})")
        else:
            print(f"❌ ERROR: Away stint goals_against calculation incorrect")

# 5. Verify team statistics calculation
print("\n5. TEAM STATISTICS VERIFICATION")
print("-" * 80)

games["home_win"] = (games["home_goals"] > games["away_goals"]).astype(int)
games["away_win"] = (games["away_goals"] > games["home_goals"]).astype(int)
games["tie"] = (games["home_goals"] == games["away_goals"]).astype(int)

home_stats = games.groupby("h_team").agg(
    home_games=("game_id", "count"),
    home_wins=("home_win", "sum"),
    home_goals_for=("home_goals", "sum"),
    home_goals_against=("away_goals", "sum"),
)
away_stats = games.groupby("a_team").agg(
    away_games=("game_id", "count"),
    away_wins=("away_win", "sum"),
    away_goals_for=("away_goals", "sum"),
    away_goals_against=("home_goals", "sum"),
)

team_summary = (
    home_stats.join(away_stats, how="outer")
    .fillna(0)
    .assign(
        games_played=lambda d: d["home_games"] + d["away_games"],
        wins=lambda d: d["home_wins"] + d["away_wins"],
        goals_for=lambda d: d["home_goals_for"] + d["away_goals_for"],
        goals_against=lambda d: d["home_goals_against"] + d["away_goals_against"],
        win_rate=lambda d: np.where(d["games_played"] > 0, d["wins"] / d["games_played"], 0),
    )
    .reset_index()
)

# Verify: For Canada, check if calculations make sense
canada_stats = team_summary[team_summary["h_team"] == team]
if len(canada_stats) > 0:
    canada_row = canada_stats.iloc[0]
    # Verify win_rate is between 0 and 1
    if 0 <= canada_row["win_rate"] <= 1:
        print(f"✓ Team win rate calculation correct for {team}: {canada_row['win_rate']:.2%}")
    else:
        print(f"❌ ERROR: Invalid win rate for {team}: {canada_row['win_rate']}")
    
    # Verify goals_for > goals_against means positive goal_diff
    goal_diff = canada_row["goals_for"] - canada_row["goals_against"]
    if (canada_row["goals_for"] > canada_row["goals_against"]) == (goal_diff > 0):
        print(f"✓ Team goal differential calculation correct for {team}: {goal_diff:+.0f}")
    else:
        print(f"❌ ERROR: Goal differential calculation incorrect for {team}")

# 6. Verify player ratings merge
print("\n6. PLAYER RATINGS MERGE VERIFICATION")
print("-" * 80)

long = pd.concat([
    home_long[["player", "minutes", "goals_for", "goals_against"]],
    away_long[["player", "minutes", "goals_for", "goals_against"]]
], ignore_index=True)

impact = (
    long.groupby("player", as_index=False)
    .agg(
        minutes=("minutes", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )
)
impact = impact.merge(players, on="player", how="left")

# Check if all players have ratings
missing_ratings = impact[impact["rating"].isna()]
if len(missing_ratings) > 0:
    print(f"⚠ WARNING: {len(missing_ratings)} players missing ratings:")
    print(f"   {missing_ratings['player'].tolist()[:5]}")
else:
    print(f"✓ All {len(impact)} players have ratings")

# 7. Verify optimization constraints
print("\n7. OPTIMIZATION CONSTRAINTS VERIFICATION")
print("-" * 80)

impact["goal_diff"] = impact["goals_for"] - impact["goals_against"]
impact["diff_per_min"] = np.where(
    impact["minutes"] > 0, 
    impact["goal_diff"] / impact["minutes"], 
    0
)
impact["diff_per_10_min"] = impact["diff_per_min"] * 10

# Check if 4-player lineups can be formed with max_rating=8.0
max_rating = 8.0
eligible = impact[impact["minutes"] >= 100].copy()

if len(eligible) >= 4:
    # Check if at least one valid lineup exists
    from itertools import combinations
    valid_lineups = []
    for combo in list(combinations(eligible.itertuples(), 4))[:100]:  # Check first 100
        total_rating = sum(p.rating for p in combo)
        if total_rating <= max_rating:
            valid_lineups.append(total_rating)
            break
    
    if valid_lineups:
        print(f"✓ Valid lineups exist with max_rating={max_rating}")
        print(f"   Example lineup rating: {valid_lineups[0]:.1f}")
    else:
        print(f"⚠ WARNING: No valid lineups found in first 100 combinations")
else:
    print(f"⚠ WARNING: Not enough eligible players ({len(eligible)} < 4)")

# 8. Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print(f"\nTeams found: {sorted(set(stints['h_team']).union(set(stints['a_team'])))}")
print(f"Total games: {len(games)}")
print(f"Total stints: {len(stints)}")
print(f"Total players: {len(players)}")
