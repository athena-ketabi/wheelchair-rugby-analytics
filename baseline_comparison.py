"""
Baseline Comparison Analysis for Wheelchair Rugby Optimizer
============================================================
This script computes real performance metrics comparing our optimizer
against baseline strategies, providing quantified results for the report.
"""

import pandas as pd
import numpy as np
from itertools import combinations
import random
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, value

# =============================================================================
# Load Data
# =============================================================================
print("=" * 60)
print("WHEELCHAIR RUGBY OPTIMIZER - BASELINE COMPARISON ANALYSIS")
print("=" * 60)
print()

stints = pd.read_csv("stint_data.csv")
players = pd.read_csv("player_data.csv")

# Choose team to analyze (Canada)
TEAM = "Canada"
MIN_MINUTES = 50  # Minimum minutes for eligibility
MAX_RATING = 8.0
NUM_STINTS = 8
FATIGUE_PENALTY = 0.15
MAX_CONSECUTIVE = 3

# =============================================================================
# Compute Player Impact Metrics
# =============================================================================
def compute_player_impact(stints_df, players_df, team):
    """Compute player impact metrics for a specific team."""
    HOME_PLAYER_COLS = ["home1", "home2", "home3", "home4"]
    AWAY_PLAYER_COLS = ["away1", "away2", "away3", "away4"]
    
    home = stints_df.loc[stints_df["h_team"] == team].copy()
    away = stints_df.loc[stints_df["a_team"] == team].copy()
    
    home_long = home.melt(
        id_vars=["minutes", "h_goals", "a_goals"],
        value_vars=HOME_PLAYER_COLS,
        var_name="slot",
        value_name="player",
    ).dropna(subset=["player"])
    home_long["goals_for"] = home_long["h_goals"]
    home_long["goals_against"] = home_long["a_goals"]
    
    away_long = away.melt(
        id_vars=["minutes", "h_goals", "a_goals"],
        value_vars=AWAY_PLAYER_COLS,
        var_name="slot",
        value_name="player",
    ).dropna(subset=["player"])
    away_long["goals_for"] = away_long["a_goals"]
    away_long["goals_against"] = away_long["h_goals"]
    
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
    impact["goal_diff"] = impact["goals_for"] - impact["goals_against"]
    impact["diff_per_min"] = np.where(impact["minutes"] > 0, impact["goal_diff"] / impact["minutes"], 0)
    impact["diff_per_10_min"] = impact["diff_per_min"] * 10
    impact = impact.merge(players_df, on="player", how="left")
    impact["rating"] = impact["rating"].fillna(2.0)
    
    return impact.sort_values("diff_per_10_min", ascending=False)

player_impact = compute_player_impact(stints, players, TEAM)
eligible = player_impact[player_impact["minutes"] >= MIN_MINUTES].copy()

print(f"Team: {TEAM}")
print(f"Total players: {len(player_impact)}")
print(f"Eligible players (>= {MIN_MINUTES} min): {len(eligible)}")
print()

# =============================================================================
# SINGLE-STINT ANALYSIS
# =============================================================================
print("=" * 60)
print("PART 1: SINGLE-STINT LINEUP OPTIMIZATION")
print("=" * 60)
print()

# --- Optimizer Solution ---
def optimize_lineup(player_df, max_rating=8.0):
    """Run the optimizer and return results."""
    prob = LpProblem("Lineup_Optimization", LpMaximize)
    
    player_vars = {
        row["player"]: LpVariable(f"select_{row['player']}", cat=LpBinary)
        for _, row in player_df.iterrows()
    }
    
    prob += lpSum([
        player_vars[row["player"]] * row["diff_per_10_min"]
        for _, row in player_df.iterrows()
    ]), "Maximize_Performance"
    
    prob += lpSum(player_vars.values()) == 4, "Exactly_4_Players"
    
    prob += lpSum([
        player_vars[row["player"]] * row["rating"]
        for _, row in player_df.iterrows()
    ]) <= max_rating, "Max_Rating"
    
    prob.solve()
    
    selected = [p for p, var in player_vars.items() if value(var) == 1]
    selected_df = player_df[player_df["player"].isin(selected)]
    
    return {
        "players": selected,
        "total_rating": selected_df["rating"].sum(),
        "total_diff_per_10": selected_df["diff_per_10_min"].sum(),
        "avg_diff_per_10": selected_df["diff_per_10_min"].mean(),
    }

optimizer_result = optimize_lineup(eligible, MAX_RATING)
print("OPTIMIZER SOLUTION:")
print(f"  Selected players: {optimizer_result['players']}")
print(f"  Total rating: {optimizer_result['total_rating']:.2f}")
print(f"  Sum of diff_per_10_min: {optimizer_result['total_diff_per_10']:+.3f}")
print(f"  Avg diff_per_10_min: {optimizer_result['avg_diff_per_10']:+.3f}")
print()

# --- Greedy Baseline 1: Experience-based (Most minutes played) ---
# A coach might naively pick the most "experienced" players
print("GREEDY BASELINE 1 - EXPERIENCE-BASED (Most minutes played):")
greedy_experience = eligible.nlargest(4, "minutes")
exp_rating = greedy_experience["rating"].sum()
exp_valid = exp_rating <= MAX_RATING

if exp_valid:
    exp_total_perf = greedy_experience["diff_per_10_min"].sum()
    exp_avg_perf = greedy_experience["diff_per_10_min"].mean()
    print(f"  Selected players: {greedy_experience['player'].tolist()}")
    print(f"  Total rating: {exp_rating:.2f} (VALID)")
    print(f"  Sum of diff_per_10_min: {exp_total_perf:+.3f}")
    print(f"  Avg diff_per_10_min: {exp_avg_perf:+.3f}")
else:
    # Find feasible experience-based lineup
    sorted_by_exp = eligible.sort_values("minutes", ascending=False)
    selected_exp = []
    current_rating = 0
    for _, player in sorted_by_exp.iterrows():
        if current_rating + player["rating"] <= MAX_RATING and len(selected_exp) < 4:
            selected_exp.append(player)
            current_rating += player["rating"]
    exp_df = pd.DataFrame(selected_exp)
    exp_total_perf = exp_df["diff_per_10_min"].sum()
    exp_avg_perf = exp_df["diff_per_10_min"].mean()
    exp_rating = current_rating
    print(f"  Selected players: {exp_df['player'].tolist()}")
    print(f"  Total rating: {exp_rating:.2f} (constrained to be valid)")
    print(f"  Sum of diff_per_10_min: {exp_total_perf:+.3f}")
    print(f"  Avg diff_per_10_min: {exp_avg_perf:+.3f}")
print()

# --- Greedy Baseline 2: Rating-maximizing (Use full budget) ---
# A coach might think "we should use all 8.0 points to get the strongest players"
print("GREEDY BASELINE 2 - RATING-MAXIMIZING (Fill the 8.0 budget):")
sorted_by_rating = eligible.sort_values("rating", ascending=False)
selected_rating_max = []
current_rating = 0
for _, player in sorted_by_rating.iterrows():
    if current_rating + player["rating"] <= MAX_RATING and len(selected_rating_max) < 4:
        selected_rating_max.append(player)
        current_rating += player["rating"]
    if len(selected_rating_max) == 4:
        break

# If we don't have 4 players yet, add lowest-rated players
if len(selected_rating_max) < 4:
    for _, player in eligible.sort_values("rating").iterrows():
        if player["player"] not in [p["player"] for p in selected_rating_max]:
            if current_rating + player["rating"] <= MAX_RATING and len(selected_rating_max) < 4:
                selected_rating_max.append(player)
                current_rating += player["rating"]

rating_max_df = pd.DataFrame(selected_rating_max)
rating_max_total_perf = rating_max_df["diff_per_10_min"].sum()
rating_max_avg_perf = rating_max_df["diff_per_10_min"].mean()
rating_max_rating = rating_max_df["rating"].sum()
print(f"  Selected players: {rating_max_df['player'].tolist()}")
print(f"  Total rating: {rating_max_rating:.2f}")
print(f"  Sum of diff_per_10_min: {rating_max_total_perf:+.3f}")
print(f"  Avg diff_per_10_min: {rating_max_avg_perf:+.3f}")
print()

# --- Greedy Baseline 3: Goals-for only (ignore defense) ---
# A coach might focus only on scoring, ignoring goals against
print("GREEDY BASELINE 3 - OFFENSE-ONLY (Most goals scored, ignore defense):")
eligible_with_gf_rate = eligible.copy()
eligible_with_gf_rate["goals_for_per_10"] = (eligible_with_gf_rate["goals_for"] / eligible_with_gf_rate["minutes"]) * 10
sorted_by_gf = eligible_with_gf_rate.sort_values("goals_for_per_10", ascending=False)
selected_offense = []
current_rating = 0
for _, player in sorted_by_gf.iterrows():
    if current_rating + player["rating"] <= MAX_RATING and len(selected_offense) < 4:
        selected_offense.append(player)
        current_rating += player["rating"]
    if len(selected_offense) == 4:
        break

offense_df = pd.DataFrame(selected_offense)
offense_total_perf = offense_df["diff_per_10_min"].sum()
offense_avg_perf = offense_df["diff_per_10_min"].mean()
offense_rating = offense_df["rating"].sum()
print(f"  Selected players: {offense_df['player'].tolist()}")
print(f"  Total rating: {offense_rating:.2f}")
print(f"  Sum of diff_per_10_min: {offense_total_perf:+.3f}")
print(f"  Avg diff_per_10_min: {offense_avg_perf:+.3f}")
print()

# --- Random Feasible Baseline (average of 1000 random valid lineups) ---
print("RANDOM FEASIBLE BASELINE (1000 random valid lineups):")
random_perfs = []
all_players = eligible.to_dict('records')

random.seed(42)  # For reproducibility
attempts = 0
max_attempts = 50000

while len(random_perfs) < 1000 and attempts < max_attempts:
    sample = random.sample(all_players, 4)
    total_rating = sum(p["rating"] for p in sample)
    if total_rating <= MAX_RATING:
        total_perf = sum(p["diff_per_10_min"] for p in sample)
        random_perfs.append(total_perf)
    attempts += 1

if len(random_perfs) > 0:
    random_avg_total = np.mean(random_perfs)
    random_std_total = np.std(random_perfs)
    random_best = np.max(random_perfs)
    random_worst = np.min(random_perfs)
    print(f"  Valid lineups sampled: {len(random_perfs)}")
    print(f"  Avg sum of diff_per_10_min: {random_avg_total:+.3f} (std: {random_std_total:.3f})")
    print(f"  Best random: {random_best:+.3f}, Worst random: {random_worst:+.3f}")
else:
    random_avg_total = 0
    print("  Could not find enough valid random lineups")
print()

# --- All Valid Lineups (exhaustive for comparison) ---
print("EXHAUSTIVE SEARCH (All valid 4-player combinations):")
all_valid_lineups = []
for combo in combinations(all_players, 4):
    total_rating = sum(p["rating"] for p in combo)
    if total_rating <= MAX_RATING:
        total_perf = sum(p["diff_per_10_min"] for p in combo)
        all_valid_lineups.append({
            "players": [p["player"] for p in combo],
            "rating": total_rating,
            "performance": total_perf
        })

if all_valid_lineups:
    all_valid_df = pd.DataFrame(all_valid_lineups)
    best_exhaustive = all_valid_df.loc[all_valid_df["performance"].idxmax()]
    avg_all_valid = all_valid_df["performance"].mean()
    print(f"  Total valid lineups: {len(all_valid_lineups)}")
    print(f"  Best lineup performance: {best_exhaustive['performance']:+.3f}")
    print(f"  Average lineup performance: {avg_all_valid:+.3f}")
    print(f"  Optimizer found optimal: {abs(optimizer_result['total_diff_per_10'] - best_exhaustive['performance']) < 0.001}")
print()

# --- Summary Statistics ---
print("-" * 60)
print("SINGLE-STINT COMPARISON SUMMARY:")
print("-" * 60)
print(f"  Optimizer performance:         {optimizer_result['total_diff_per_10']:+.3f}")
print()
print(f"  Experience-based greedy:       {exp_total_perf:+.3f}")
exp_improvement = ((optimizer_result['total_diff_per_10'] - exp_total_perf) / abs(exp_total_perf) * 100) if exp_total_perf != 0 else float('inf')
print(f"    Improvement over experience: {exp_improvement:+.1f}%")
print()
print(f"  Rating-maximizing greedy:      {rating_max_total_perf:+.3f}")
rating_improvement = ((optimizer_result['total_diff_per_10'] - rating_max_total_perf) / abs(rating_max_total_perf) * 100) if rating_max_total_perf != 0 else float('inf')
print(f"    Improvement over rating-max: {rating_improvement:+.1f}%")
print()
print(f"  Offense-only greedy:           {offense_total_perf:+.3f}")
offense_improvement = ((optimizer_result['total_diff_per_10'] - offense_total_perf) / abs(offense_total_perf) * 100) if offense_total_perf != 0 else float('inf')
print(f"    Improvement over offense:    {offense_improvement:+.1f}%")
print()
print(f"  Random baseline (avg):         {random_avg_total:+.3f}")
if random_avg_total != 0:
    random_improvement = ((optimizer_result['total_diff_per_10'] - random_avg_total) / abs(random_avg_total) * 100)
    print(f"    Improvement over random:     {random_improvement:+.1f}%")
print()

# =============================================================================
# MULTI-STINT ROTATION ANALYSIS
# =============================================================================
print("=" * 60)
print("PART 2: MULTI-STINT GAME ROTATION PLANNING")
print("=" * 60)
print()

def plan_game_rotation(player_df, num_stints=8, max_rating=8.0, max_consecutive=3, fatigue_penalty=0.15):
    """Plan optimal rotations for entire game."""
    players = player_df.to_dict('records')
    
    player_state = {
        p["player"]: {
            "base_performance": p["diff_per_10_min"],
            "rating": p["rating"],
            "consecutive_stints": 0,
            "total_stints_played": 0,
        }
        for p in players
    }
    
    rotation_plan = []
    total_performance = 0
    
    for stint_num in range(num_stints):
        effective_players = []
        for p in players:
            state = player_state[p["player"]]
            fatigue_multiplier = max(0.5, 1.0 - (state["consecutive_stints"] * fatigue_penalty))
            effective_perf = state["base_performance"] * fatigue_multiplier
            must_rest = state["consecutive_stints"] >= max_consecutive
            
            effective_players.append({
                "player": p["player"],
                "rating": p["rating"],
                "base_perf": state["base_performance"],
                "effective_perf": effective_perf,
                "fatigue_multiplier": fatigue_multiplier,
                "must_rest": must_rest,
            })
        
        prob = LpProblem(f"Stint_{stint_num}", LpMaximize)
        player_vars = {p["player"]: LpVariable(f"x_{p['player']}_{stint_num}", cat=LpBinary) for p in effective_players}
        
        prob += lpSum([player_vars[p["player"]] * p["effective_perf"] for p in effective_players])
        prob += lpSum(player_vars.values()) == 4
        prob += lpSum([player_vars[p["player"]] * p["rating"] for p in effective_players]) <= max_rating
        
        for p in effective_players:
            if p["must_rest"]:
                prob += player_vars[p["player"]] == 0
        
        prob.solve()
        
        selected = [p["player"] for p in effective_players if value(player_vars[p["player"]]) == 1]
        stint_perf = sum(p["effective_perf"] for p in effective_players if p["player"] in selected)
        total_performance += stint_perf
        
        rotation_plan.append({
            "stint": stint_num + 1,
            "players": selected,
            "performance": stint_perf
        })
        
        # Update states
        for p in players:
            if p["player"] in selected:
                player_state[p["player"]]["consecutive_stints"] += 1
                player_state[p["player"]]["total_stints_played"] += 1
            else:
                player_state[p["player"]]["consecutive_stints"] = 0
    
    # Compute playing time stats
    playing_time_stats = []
    for p in players:
        state = player_state[p["player"]]
        pct = state["total_stints_played"] / num_stints
        playing_time_stats.append({
            "player": p["player"],
            "stints_played": state["total_stints_played"],
            "pct": pct,
            "met_25pct": pct >= 0.25
        })
    
    return {
        "rotation_plan": rotation_plan,
        "total_performance": total_performance,
        "avg_per_stint": total_performance / num_stints,
        "playing_time_stats": playing_time_stats
    }

# --- Run Optimizer Rotation ---
opt_rotation = plan_game_rotation(eligible, NUM_STINTS, MAX_RATING, MAX_CONSECUTIVE, FATIGUE_PENALTY)

print("OPTIMIZER ROTATION PLAN:")
for stint in opt_rotation["rotation_plan"]:
    print(f"  Stint {stint['stint']}: {stint['players']} (perf: {stint['performance']:+.3f})")
print()
print(f"  Total game performance: {opt_rotation['total_performance']:+.3f}")
print(f"  Average per stint: {opt_rotation['avg_per_stint']:+.3f}")
players_met_min = sum(1 for p in opt_rotation['playing_time_stats'] if p['met_25pct'])
print(f"  Players meeting 25% min time: {players_met_min}/{len(opt_rotation['playing_time_stats'])}")
print()

# --- Static Lineup Baseline (same 4 players all game, with fatigue) ---
print("STATIC LINEUP BASELINE (Best lineup, no rotation, with fatigue):")

# Use the optimal single-stint lineup for all stints
static_players = optimizer_result["players"]
static_total_perf = 0

print(f"  Using lineup: {static_players}")
for stint_num in range(NUM_STINTS):
    # All players have been playing since start
    consecutive = stint_num
    if consecutive >= MAX_CONSECUTIVE:
        # Would need to rest, but static means no subs - performance tanks
        fatigue_mult = 0.5  # Floor
    else:
        fatigue_mult = max(0.5, 1.0 - consecutive * FATIGUE_PENALTY)
    
    stint_base_perf = optimizer_result["total_diff_per_10"]
    stint_actual_perf = stint_base_perf * fatigue_mult
    static_total_perf += stint_actual_perf
    print(f"  Stint {stint_num + 1}: fatigue={1-fatigue_mult:.0%}, perf={stint_actual_perf:+.3f}")

print()
print(f"  Total game performance: {static_total_perf:+.3f}")
print(f"  Average per stint: {static_total_perf / NUM_STINTS:+.3f}")
print()

# --- Comparison ---
print("-" * 60)
print("ROTATION PLANNING COMPARISON SUMMARY:")
print("-" * 60)
print(f"  Optimizer (with rotation):  {opt_rotation['total_performance']:+.3f}")
print(f"  Static lineup (no rotation): {static_total_perf:+.3f}")
rotation_improvement = ((opt_rotation['total_performance'] - static_total_perf) / abs(static_total_perf) * 100) if static_total_perf != 0 else 0
print(f"  Improvement from rotation:  {rotation_improvement:+.1f}%")
print()

# =============================================================================
# FINAL SUMMARY FOR REPORT
# =============================================================================
print("=" * 60)
print("FINAL METRICS FOR YOUR REPORT")
print("=" * 60)
print()
print("SINGLE-STINT OPTIMIZATION:")
print(f"  - Optimizer lineup performance: {optimizer_result['avg_diff_per_10']:+.2f} avg diff/10min per player")
print(f"  - Optimizer total: {optimizer_result['total_diff_per_10']:+.2f}")
print()
print("  BASELINE COMPARISONS:")
print(f"  - Experience-based greedy: {exp_total_perf:+.2f} (Improvement: {exp_improvement:+.1f}%)")
print(f"  - Rating-maximizing greedy: {rating_max_total_perf:+.2f} (Improvement: {rating_improvement:+.1f}%)")
print(f"  - Offense-only greedy: {offense_total_perf:+.2f} (Improvement: {offense_improvement:+.1f}%)")
print(f"  - Random baseline avg: {random_avg_total:+.2f} (Improvement: {random_improvement:+.1f}%)")
print()
print("GAME ROTATION PLANNING:")
print(f"  - Optimized rotation total: {opt_rotation['total_performance']:+.2f}")
print(f"  - Static lineup total: {static_total_perf:+.2f}")
print(f"  - Improvement from rotation: {rotation_improvement:+.1f}%")
print(f"  - Players meeting min playing time: {players_met_min}/{len(opt_rotation['playing_time_stats'])} ({players_met_min/len(opt_rotation['playing_time_stats'])*100:.0f}%)")
print()
print("=" * 60)
