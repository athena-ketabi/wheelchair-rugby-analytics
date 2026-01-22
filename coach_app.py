"""
Wheelchair Rugby Coach Decision Support System
==============================================
An optimization-based tool to help coaches make data-driven decisions
for lineup selection, opponent analysis, and game strategy.

Uses Linear Programming (PuLP) for lineup optimization.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pulp import (
    LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus, value
)
from itertools import combinations
from typing import Optional

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Rugby Coach Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 0.5rem 0;
        letter-spacing: 0.5px;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Improved Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
    }
    
    /* Better spacing for metrics */
    [data-testid="stMetricContainer"] {
        background: rgba(30, 41, 59, 0.5);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"]:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }
    
    /* Improved Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #cbd5e1 !important;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #60a5fa !important;
        background-color: rgba(37, 99, 235, 0.1);
        border-color: rgba(37, 99, 235, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        background-color: rgba(37, 99, 235, 0.15) !important;
        border-bottom: 3px solid #60a5fa !important;
        border-color: rgba(37, 99, 235, 0.3) !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #60a5fa !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        background-color: #60a5fa !important;
    }
    
    .stTabs button[data-baseweb="tab"] p {
        color: #cbd5e1 !important;
        margin: 0;
    }
    
    .stTabs button[data-baseweb="tab"]:hover p {
        color: #60a5fa !important;
    }
    
    .stTabs button[aria-selected="true"] p {
        color: #60a5fa !important;
    }
    
    /* Improved Headers */
    h1, h2, h3 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Better Expanders */
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: rgba(37, 99, 235, 0.3);
        background: rgba(30, 41, 59, 0.6);
    }
    
    /* Improved Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Better Selectboxes */
    [data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    [data-baseweb="select"]:hover {
        border-color: rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Improved Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Better Dataframes */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Improved Success/Info/Error boxes */
    [data-baseweb="notification"] {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    
    /* Better spacing overall */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Improved slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Better divider */
    hr {
        border-color: rgba(148, 163, 184, 0.2);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================
def format_player_name(player_name: str) -> str:
    """Convert 'Country_p#' format to 'Player #' format."""
    import re
    match = re.search(r'_p(\d+)$', str(player_name))
    if match:
        return f"Player {match.group(1)}"
    return player_name


def format_lineup(lineup_str: str) -> str:
    """Format lineup string with Player # format."""
    if not lineup_str or pd.isna(lineup_str):
        return lineup_str
    players = lineup_str.split("|")
    formatted = [format_player_name(p.strip()) for p in players]
    return " | ".join(formatted)


# =============================================================================
# Data Loading & Processing
# =============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the CSV data."""
    stints = pd.read_csv("stint_data.csv")
    players = pd.read_csv("player_data.csv")
    return stints, players


@st.cache_data
def build_games_table(stints: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stints into game-level results."""
    games = (
        stints.groupby(["game_id", "h_team", "a_team"], as_index=False)
        .agg(
            total_minutes=("minutes", "sum"),
            home_goals=("h_goals", "sum"),
            away_goals=("a_goals", "sum"),
        )
    )
    games["winner"] = np.where(
        games["home_goals"] > games["away_goals"], games["h_team"],
        np.where(games["away_goals"] > games["home_goals"], games["a_team"], "TIE")
    )
    games["home_win"] = (games["winner"] == games["h_team"]).astype(int)
    games["away_win"] = (games["winner"] == games["a_team"]).astype(int)
    games["tie"] = (games["winner"] == "TIE").astype(int)
    games["margin"] = games["home_goals"] - games["away_goals"]
    return games


@st.cache_data
def compute_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute win rates and statistics for all teams."""
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
            goal_diff=lambda d: d["goals_for"] - d["goals_against"],
            avg_goals_for=lambda d: np.where(d["games_played"] > 0, d["goals_for"] / d["games_played"], 0),
            avg_goals_against=lambda d: np.where(d["games_played"] > 0, d["goals_against"] / d["games_played"], 0),
        )
        .sort_values("win_rate", ascending=False)
    )
    team_summary.index.name = "team"
    return team_summary.reset_index()


@st.cache_data
def compute_player_impact(stints: pd.DataFrame, players: pd.DataFrame, team: str) -> pd.DataFrame:
    """Compute player impact metrics for a specific team."""
    HOME_PLAYER_COLS = ["home1", "home2", "home3", "home4"]
    AWAY_PLAYER_COLS = ["away1", "away2", "away3", "away4"]
    
    home = stints.loc[stints["h_team"] == team].copy()
    away = stints.loc[stints["a_team"] == team].copy()
    
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
    impact["diff_per_min"] = np.where(
        impact["minutes"] > 0, 
        impact["goal_diff"] / impact["minutes"], 
        0
    )
    impact["diff_per_10_min"] = impact["diff_per_min"] * 10
    impact = impact.merge(players, on="player", how="left")
    impact["rating"] = impact["rating"].fillna(2.0)  # Default rating if missing
    
    # Add display name column (Player # format)
    impact["display_name"] = impact["player"].apply(format_player_name)
    
    return impact.sort_values("diff_per_10_min", ascending=False)


@st.cache_data
def compute_lineup_performance(stints: pd.DataFrame, team: str) -> pd.DataFrame:
    """Compute performance metrics for all 4-player lineups."""
    HOME_PLAYER_COLS = ["home1", "home2", "home3", "home4"]
    AWAY_PLAYER_COLS = ["away1", "away2", "away3", "away4"]
    
    home = stints.loc[stints["h_team"] == team].copy()
    away = stints.loc[stints["a_team"] == team].copy()
    
    def lineup_key(row, cols):
        return "|".join(sorted([str(row[c]) for c in cols]))
    
    if len(home) > 0:
        home["lineup"] = home.apply(lambda r: lineup_key(r, HOME_PLAYER_COLS), axis=1)
        home["goals_for"] = home["h_goals"]
        home["goals_against"] = home["a_goals"]
    
    if len(away) > 0:
        away["lineup"] = away.apply(lambda r: lineup_key(r, AWAY_PLAYER_COLS), axis=1)
        away["goals_for"] = away["a_goals"]
        away["goals_against"] = away["h_goals"]
    
    combined = pd.concat([
        home[["lineup", "minutes", "goals_for", "goals_against"]] if len(home) > 0 else pd.DataFrame(),
        away[["lineup", "minutes", "goals_for", "goals_against"]] if len(away) > 0 else pd.DataFrame(),
    ], ignore_index=True)
    
    if len(combined) == 0:
        return pd.DataFrame()
    
    perf = (
        combined.groupby("lineup", as_index=False)
        .agg(
            minutes=("minutes", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
        )
    )
    perf["goal_diff"] = perf["goals_for"] - perf["goals_against"]
    perf["diff_per_min"] = np.where(perf["minutes"] > 0, perf["goal_diff"] / perf["minutes"], 0)
    perf["diff_per_10_min"] = perf["diff_per_min"] * 10
    
    return perf.sort_values("diff_per_10_min", ascending=False)


@st.cache_data
def compute_head_to_head(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """Compute head-to-head record against all opponents."""
    mask = (games["h_team"] == team) | (games["a_team"] == team)
    g = games.loc[mask].copy()
    
    g["opponent"] = np.where(g["h_team"] == team, g["a_team"], g["h_team"])
    g["team_goals"] = np.where(g["h_team"] == team, g["home_goals"], g["away_goals"])
    g["opp_goals"] = np.where(g["h_team"] == team, g["away_goals"], g["home_goals"])
    g["team_win"] = (g["team_goals"] > g["opp_goals"]).astype(int)
    g["team_loss"] = (g["team_goals"] < g["opp_goals"]).astype(int)
    
    h2h = (
        g.groupby("opponent", as_index=False)
        .agg(
            games=("game_id", "count"),
            wins=("team_win", "sum"),
            losses=("team_loss", "sum"),
            goals_for=("team_goals", "sum"),
            goals_against=("opp_goals", "sum"),
        )
    )
    h2h["goal_diff"] = h2h["goals_for"] - h2h["goals_against"]
    h2h["win_rate"] = h2h["wins"] / h2h["games"]
    
    return h2h.sort_values("win_rate", ascending=False)


# =============================================================================
# Optimization Model
# =============================================================================
def optimize_lineup(
    player_impact: pd.DataFrame,
    max_rating: float = 8.0,
    min_minutes: float = 100.0,
    required_players: list[str] = None,
    excluded_players: list[str] = None,
    objective: str = "diff_per_10_min"
) -> dict:
    """
    Optimize lineup selection using Linear Programming.
    
    Args:
        player_impact: DataFrame with player metrics
        max_rating: Maximum total rating allowed (wheelchair rugby rule: 8.0)
        min_minutes: Minimum minutes played to be considered
        required_players: Players that must be in the lineup
        excluded_players: Players that cannot be in the lineup
        objective: Metric to maximize ('diff_per_10_min', 'goals_for', 'goal_diff')
    
    Returns:
        Dictionary with optimal lineup and metrics
    """
    required_players = required_players or []
    excluded_players = excluded_players or []
    
    # Filter players
    eligible = player_impact[player_impact["minutes"] >= min_minutes].copy()
    eligible = eligible[~eligible["player"].isin(excluded_players)]
    
    if len(eligible) < 4:
        return {"status": "INFEASIBLE", "message": "Not enough eligible players"}
    
    # Create optimization problem
    prob = LpProblem("Lineup_Optimization", LpMaximize)
    
    # Decision variables: binary for each player (1 = selected, 0 = not)
    player_vars = {
        row["player"]: LpVariable(f"select_{row['player']}", cat=LpBinary)
        for _, row in eligible.iterrows()
    }
    
    # Objective: maximize the sum of objective metric for selected players
    prob += lpSum([
        player_vars[row["player"]] * row[objective]
        for _, row in eligible.iterrows()
    ]), "Maximize_Performance"
    
    # Constraint 1: Exactly 4 players
    prob += lpSum(player_vars.values()) == 4, "Exactly_4_Players"
    
    # Constraint 2: Total rating <= max_rating
    prob += lpSum([
        player_vars[row["player"]] * row["rating"]
        for _, row in eligible.iterrows()
    ]) <= max_rating, "Max_Rating"
    
    # Constraint 3: Required players must be selected
    for player in required_players:
        if player in player_vars:
            prob += player_vars[player] == 1, f"Required_{player}"
    
    # Solve
    prob.solve()
    
    if LpStatus[prob.status] != "Optimal":
        return {"status": LpStatus[prob.status], "message": "No optimal solution found"}
    
    # Extract results
    selected = [p for p, var in player_vars.items() if value(var) == 1]
    selected_df = eligible[eligible["player"].isin(selected)].copy()
    
    return {
        "status": "Optimal",
        "players": selected,
        "total_rating": selected_df["rating"].sum(),
        "total_objective": selected_df[objective].sum(),
        "avg_diff_per_10_min": selected_df["diff_per_10_min"].mean(),
        "details": selected_df[["player", "rating", "minutes", "goal_diff", "diff_per_10_min"]].to_dict("records")
    }


def find_all_valid_lineups(
    player_impact: pd.DataFrame,
    max_rating: float = 8.0,
    min_minutes: float = 100.0,
    top_n: int = 10
) -> pd.DataFrame:
    """Find all valid lineups and rank by performance."""
    eligible = player_impact[player_impact["minutes"] >= min_minutes].copy()
    
    if len(eligible) < 4:
        return pd.DataFrame()
    
    lineups = []
    for combo in combinations(eligible.itertuples(), 4):
        total_rating = sum(p.rating for p in combo)
        if total_rating <= max_rating:
            lineups.append({
                "lineup": " | ".join(sorted([p.player for p in combo])),
                "players": [p.player for p in combo],
                "total_rating": total_rating,
                "avg_diff_per_10_min": np.mean([p.diff_per_10_min for p in combo]),
                "total_minutes": sum(p.minutes for p in combo),
                "total_goal_diff": sum(p.goal_diff for p in combo),
            })
    
    if not lineups:
        return pd.DataFrame()
    
    df = pd.DataFrame(lineups)
    return df.nlargest(top_n, "avg_diff_per_10_min")


# =============================================================================
# Game Rotation Planner - Knapsack with Fatigue Management
# =============================================================================
def plan_game_rotation(
    player_impact: pd.DataFrame,
    num_stints: int = 8,
    stint_duration: float = 4.0,
    max_rating: float = 8.0,
    max_consecutive_stints: int = 3,
    fatigue_penalty: float = 0.15,
    min_playing_time_pct: float = 0.25,
    min_minutes: float = 50.0
) -> dict:
    """
    Plan optimal player rotations for an entire game using the Knapsack approach
    with fatigue management and fair playing time distribution.
    
    This is a MULTI-PERIOD KNAPSACK problem where:
    - Each stint is a separate knapsack problem
    - Player "value" (performance) decreases with fatigue
    - All players must get minimum playing time
    - No player can play more than max_consecutive_stints in a row
    
    Args:
        player_impact: DataFrame with player metrics
        num_stints: Number of stints in the game (default 8 = 32 min game)
        stint_duration: Duration of each stint in minutes
        max_rating: Maximum total rating per lineup (wheelchair rugby rule: 8.0)
        max_consecutive_stints: Max stints a player can play consecutively before rest
        fatigue_penalty: Performance reduction per consecutive stint (15% default)
        min_playing_time_pct: Minimum % of game each player should play (25% default)
        min_minutes: Minimum historical minutes to be eligible
    
    Returns:
        Dictionary with rotation plan and statistics
    """
    # Filter eligible players
    eligible = player_impact[player_impact["minutes"] >= min_minutes].copy()
    
    if len(eligible) < 4:
        return {"status": "INFEASIBLE", "message": "Not enough eligible players"}
    
    # Initialize tracking
    players = eligible.to_dict('records')
    num_players = len(players)
    
    # Track state for each player
    player_state = {
        p["player"]: {
            "base_performance": p["diff_per_10_min"],
            "rating": p["rating"],
            "consecutive_stints": 0,
            "total_stints_played": 0,
            "current_fatigue": 0.0,
            "resting_stints": 0
        }
        for p in players
    }
    
    # Minimum stints each player should play for fairness
    min_stints_per_player = max(1, int(num_stints * min_playing_time_pct * 4 / num_players))
    
    rotation_plan = []
    
    for stint_num in range(num_stints):
        # Calculate effective performance for each player (accounting for fatigue)
        effective_players = []
        for p in players:
            state = player_state[p["player"]]
            
            # Fatigue reduces performance based on consecutive stints
            fatigue_multiplier = 1.0 - (state["consecutive_stints"] * fatigue_penalty)
            fatigue_multiplier = max(0.5, fatigue_multiplier)  # Floor at 50% performance
            
            # Bonus for rested players (recovered)
            if state["resting_stints"] >= 2:
                fatigue_multiplier = min(1.0, fatigue_multiplier + 0.1)
            
            effective_perf = state["base_performance"] * fatigue_multiplier
            
            # Check if player MUST rest (exceeded max consecutive)
            must_rest = state["consecutive_stints"] >= max_consecutive_stints
            
            # Check if player NEEDS playing time (fairness)
            stints_remaining = num_stints - stint_num
            stints_needed = max(0, min_stints_per_player - state["total_stints_played"])
            urgently_needs_time = stints_needed >= stints_remaining
            
            effective_players.append({
                "player": p["player"],
                "rating": p["rating"],
                "base_perf": state["base_performance"],
                "effective_perf": effective_perf,
                "fatigue_multiplier": fatigue_multiplier,
                "consecutive": state["consecutive_stints"],
                "total_played": state["total_stints_played"],
                "must_rest": must_rest,
                "urgently_needs_time": urgently_needs_time
            })
        
        # Solve knapsack for this stint
        # Use PuLP for optimal selection
        prob = LpProblem(f"Stint_{stint_num}_Lineup", LpMaximize)
        
        # Decision variables
        player_vars = {
            p["player"]: LpVariable(f"select_{p['player']}_{stint_num}", cat=LpBinary)
            for p in effective_players
        }
        
        # Objective: Maximize effective performance (fatigue-adjusted)
        prob += lpSum([
            player_vars[p["player"]] * p["effective_perf"]
            for p in effective_players
        ]), "Maximize_Effective_Performance"
        
        # Constraint 1: Exactly 4 players
        prob += lpSum(player_vars.values()) == 4, "Exactly_4_Players"
        
        # Constraint 2: Total rating <= max_rating (KNAPSACK CAPACITY)
        prob += lpSum([
            player_vars[p["player"]] * p["rating"]
            for p in effective_players
        ]) <= max_rating, "Rating_Capacity"
        
        # Constraint 3: Players who must rest cannot play
        for p in effective_players:
            if p["must_rest"]:
                prob += player_vars[p["player"]] == 0, f"Must_Rest_{p['player']}"
        
        # Constraint 4: Players who urgently need time should play (if feasible)
        # This is a soft constraint - we prioritize but don't force if infeasible
        urgent_players = [p for p in effective_players if p["urgently_needs_time"] and not p["must_rest"]]
        
        # Solve
        prob.solve()
        
        if LpStatus[prob.status] != "Optimal":
            # Fallback: relax constraints
            return {"status": "INFEASIBLE", "message": f"Could not find valid lineup for stint {stint_num + 1}"}
        
        # Extract selected players
        selected = [p["player"] for p in effective_players if value(player_vars[p["player"]]) == 1]
        selected_details = [p for p in effective_players if p["player"] in selected]
        
        # Calculate stint metrics
        total_rating = sum(p["rating"] for p in selected_details)
        total_effective_perf = sum(p["effective_perf"] for p in selected_details)
        avg_fatigue = np.mean([1 - p["fatigue_multiplier"] for p in selected_details])
        
        rotation_plan.append({
            "stint": stint_num + 1,
            "players": selected,
            "total_rating": total_rating,
            "expected_performance": total_effective_perf,
            "avg_fatigue_level": avg_fatigue,
            "details": selected_details
        })
        
        # Update player states for next stint
        for p in players:
            if p["player"] in selected:
                player_state[p["player"]]["consecutive_stints"] += 1
                player_state[p["player"]]["total_stints_played"] += 1
                player_state[p["player"]]["resting_stints"] = 0
                player_state[p["player"]]["current_fatigue"] = (
                    player_state[p["player"]]["consecutive_stints"] * fatigue_penalty
                )
            else:
                # Player is resting
                player_state[p["player"]]["consecutive_stints"] = 0
                player_state[p["player"]]["resting_stints"] += 1
                player_state[p["player"]]["current_fatigue"] = max(
                    0, player_state[p["player"]]["current_fatigue"] - fatigue_penalty
                )
    
    # Compile final statistics
    player_summary = []
    for p in players:
        state = player_state[p["player"]]
        playing_time = state["total_stints_played"] * stint_duration
        playing_pct = state["total_stints_played"] / num_stints
        player_summary.append({
            "player": p["player"],
            "rating": p["rating"],
            "stints_played": state["total_stints_played"],
            "playing_time_min": playing_time,
            "playing_time_pct": playing_pct,
            "met_minimum": playing_pct >= min_playing_time_pct
        })
    
    total_game_performance = sum(s["expected_performance"] for s in rotation_plan)
    
    return {
        "status": "Optimal",
        "rotation_plan": rotation_plan,
        "player_summary": player_summary,
        "total_game_performance": total_game_performance,
        "avg_performance_per_stint": total_game_performance / num_stints,
        "game_duration_min": num_stints * stint_duration,
        "parameters": {
            "num_stints": num_stints,
            "stint_duration": stint_duration,
            "max_rating": max_rating,
            "max_consecutive": max_consecutive_stints,
            "fatigue_penalty": fatigue_penalty,
            "min_playing_time": min_playing_time_pct
        }
    }


# =============================================================================
# Main Application
# =============================================================================
def main():
    # Load data
    try:
        stints, players = load_data()
        games = build_games_table(stints)
        team_stats = compute_team_stats(games)
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please ensure `stint_data.csv` and `player_data.csv` are in the same directory.")
        return
    
    teams = sorted(set(stints["h_team"]).union(set(stints["a_team"])))
    
    # Header
    st.markdown('<p class="main-title">Sports Analytics: Wheelchair Rugby</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-Driven Decisions for Winning Games</p>', unsafe_allow_html=True)
    
    # Sidebar: Team Selection
    selected_team = st.sidebar.selectbox(
        "Select Your Team",
        teams,
        index=teams.index("Canada") if "Canada" in teams else 0
    )
    
    # Show selected team prominently
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); 
                padding: 1rem; border-radius: 8px; color: white; margin-top: 1rem; text-align: center;">
        <p style="margin: 0; font-size: 0.9rem;">Currently Viewing</p>
        <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{selected_team}</p>
        <p style="margin: 0; font-size: 0.8rem;">Players</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get team-specific data
    player_impact = compute_player_impact(stints, players, selected_team)
    lineup_perf = compute_lineup_performance(stints, selected_team)
    h2h = compute_head_to_head(games, selected_team)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard", 
        "Lineup Optimizer", 
        "Game Rotation",
        "Player Analysis",
        "Opponent Scouting",
        "Historical Lineups"
    ])
    
    # =========================================================================
    # TAB 1: Dashboard
    # =========================================================================
    with tab1:
        st.header(f"Team Dashboard: {selected_team}")
        
        # Team metrics
        team_row = team_stats[team_stats["team"] == selected_team].iloc[0]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Win Rate", f"{team_row['win_rate']:.1%}")
        with col2:
            st.metric("Wins", int(team_row["wins"]))
        with col3:
            st.metric("Games", int(team_row["games_played"]))
        with col4:
            st.metric("Goal Diff", f"{team_row['goal_diff']:+.0f}")
        with col5:
            rank = team_stats[team_stats["win_rate"] >= team_row["win_rate"]].shape[0]
            st.metric("League Rank", f"#{rank}")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("League Standings")
            fig = px.bar(
                team_stats.sort_values("win_rate", ascending=True),
                x="win_rate",
                y="team",
                orientation="h",
                color="win_rate",
                color_continuous_scale="Viridis",
                labels={"win_rate": "Win Rate", "team": "Team"}
            )
            fig.update_layout(height=400, showlegend=False)
            fig.add_vline(x=team_row["win_rate"], line_dash="dash", line_color="red",
                         annotation_text=f"{selected_team}")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        with col2:
            st.subheader("Head-to-Head Performance")
            fig = px.bar(
                h2h.sort_values("win_rate", ascending=True),
                x="win_rate",
                y="opponent",
                orientation="h",
                color="win_rate",
                color_continuous_scale="RdYlGn",
                labels={"win_rate": "Win Rate", "opponent": "Opponent"}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # Quick insights
        st.subheader("Key Insights")
        
        best_opponent = h2h.iloc[0]
        worst_opponent = h2h.iloc[-1]
        top_player = player_impact.iloc[0] if len(player_impact) > 0 else None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Best matchup:** vs {best_opponent['opponent']} ({best_opponent['win_rate']:.0%} win rate)")
        with col2:
            st.error(f"**Toughest opponent:** {worst_opponent['opponent']} ({worst_opponent['win_rate']:.0%} win rate)")
        with col3:
            if top_player is not None:
                st.info(f"**Top performer:** {top_player['display_name']} ({top_player['diff_per_10_min']:+.2f}/10min)")
    
    # =========================================================================
    # TAB 2: Lineup Optimizer
    # =========================================================================
    with tab2:
        st.header("Optimal Lineup Selector")
        st.markdown("""
        Use our **optimization engine** to find the best lineup based on your constraints.
        The model maximizes performance while respecting wheelchair rugby classification rules.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimization Parameters")
            
            max_rating = st.slider(
                "Maximum Total Rating",
                min_value=4.0,
                max_value=10.0,
                value=8.0,
                step=0.5,
                help="Standard wheelchair rugby rule: max 8.0 points on court"
            )
            
            min_minutes = st.slider(
                "Minimum Minutes Played",
                min_value=0,
                max_value=500,
                value=100,
                step=50,
                help="Filter players by experience"
            )
            
            objective = st.selectbox(
                "Optimization Objective",
                ["diff_per_10_min", "goal_diff", "goals_for"],
                format_func=lambda x: {
                    "diff_per_10_min": "Goal Differential per 10 min (recommended)",
                    "goal_diff": "Total Goal Differential",
                    "goals_for": "Total Goals Scored"
                }.get(x, x)
            )
            
            st.divider()
            
            st.subheader("Constraints")
            
            eligible_df = player_impact[player_impact["minutes"] >= min_minutes]
            # Create mapping from display name to original player name
            player_display_map = dict(zip(eligible_df["display_name"], eligible_df["player"]))
            eligible_display_names = eligible_df["display_name"].tolist()
            
            required_display = st.multiselect(
                "Must Include Players",
                eligible_display_names,
                help="Players that must be in the lineup"
            )
            # Convert display names back to original player names for optimization
            required = [player_display_map[d] for d in required_display]
            
            excluded_display = st.multiselect(
                "Exclude Players",
                [p for p in eligible_display_names if p not in required_display],
                help="Players to exclude (injured, rested, etc.)"
            )
            excluded = [player_display_map[d] for d in excluded_display]
            
            optimize_btn = st.button("Find Optimal Lineup", type="primary", use_container_width=True)
        
        with col2:
            if optimize_btn:
                with st.spinner("Running optimization..."):
                    result = optimize_lineup(
                        player_impact,
                        max_rating=max_rating,
                        min_minutes=min_minutes,
                        required_players=required,
                        excluded_players=excluded,
                        objective=objective
                    )
                
                if result["status"] == "Optimal":
                    st.success("Optimal lineup found!")
                    
                    # Format player names for display
                    display_players = [format_player_name(p) for p in result['players']]
                    
                    # Display recommendation
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; margin: 1rem 0;">
                        <h3 style="margin: 0;">Recommended Lineup</h3>
                        <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                            {' - '.join(display_players)}
                        </p>
                        <p style="margin: 0;">
                            Total Rating: <strong>{result['total_rating']:.1f}</strong> | 
                            Avg +/- per 10min: <strong>{result['avg_diff_per_10_min']:+.2f}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Player details
                    st.subheader("Player Details")
                    details_df = pd.DataFrame(result["details"])
                    details_df["player"] = details_df["player"].apply(format_player_name)
                    details_df.columns = ["Player", "Rating", "Minutes", "Goal Diff", "+/- per 10min"]
                    st.dataframe(details_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    max_value = max([d["diff_per_10_min"] for d in result["details"]])
                    min_value = min([d["diff_per_10_min"] for d in result["details"]])
                    y_range_padding = (max_value - min_value) * 0.2 if max_value != min_value else max_value * 0.2
                    
                    fig = go.Figure(go.Bar(
                        x=[format_player_name(d["player"]) for d in result["details"]],
                        y=[d["diff_per_10_min"] for d in result["details"]],
                        marker_color=["#11998e" if d["diff_per_10_min"] > 0 else "#f5576c" 
                                     for d in result["details"]],
                        text=[f"{d['diff_per_10_min']:+.2f}" for d in result["details"]],
                        textposition="outside"
                    ))
                    fig.update_layout(
                        title=dict(text="Selected Players Performance", y=0.95),
                        xaxis_title="Player",
                        yaxis_title="Goal Diff per 10 min",
                        height=400,
                        margin=dict(t=60),
                        yaxis=dict(range=[min(min_value - y_range_padding, 0), max_value + y_range_padding])
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    
                else:
                    st.error(f"{result.get('message', 'Optimization failed')}")
                    st.info("Try adjusting constraints (increase max rating or decrease min minutes)")
            
            else:
                # Show all valid lineups as preview
                st.subheader("Top Valid Lineups Preview")
                top_lineups = find_all_valid_lineups(
                    player_impact, max_rating, min_minutes, top_n=5
                )
                if len(top_lineups) > 0:
                    for i, row in top_lineups.iterrows():
                        display_lineup = format_lineup(row['lineup'].replace(" | ", "|"))
                        display_players = [format_player_name(p) for p in row['players']]
                        with st.expander(f"#{i+1}: {display_lineup[:50]}..."):
                            st.write(f"**Players:** {', '.join(display_players)}")
                            st.write(f"**Total Rating:** {row['total_rating']:.1f}")
                            st.write(f"**Avg +/- per 10min:** {row['avg_diff_per_10_min']:+.2f}")
                else:
                    st.warning("No valid lineups found with current constraints")
    
    # =========================================================================
    # TAB 3: Game Rotation Planner (Knapsack with Fatigue)
    # =========================================================================
    with tab3:
        st.header("Game Rotation Planner")
        st.markdown("""
        **Smart player rotation using the Knapsack optimization approach with fatigue management.**
        
        This tool plans your entire game lineup rotations by solving a **multi-period knapsack problem**:
        - **Knapsack Capacity**: Total player rating must not exceed 8.0 (wheelchair rugby rule)
        - **Items**: Players with their ratings (weights) and performance scores (values)
        - **Fatigue**: Players lose effectiveness when playing consecutive stints
        - **Fairness**: All players get minimum playing time throughout the game
        """)
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Game Parameters")
            
            num_stints = st.slider(
                "Number of Stints",
                min_value=4,
                max_value=12,
                value=8,
                help="Total stints in the game (8 stints × 4 min = 32 min game)"
            )
            
            stint_duration = st.slider(
                "Stint Duration (minutes)",
                min_value=2.0,
                max_value=6.0,
                value=4.0,
                step=0.5,
                help="Duration of each stint"
            )
            
            st.markdown(f"**Total Game Time:** {num_stints * stint_duration:.0f} minutes")
            
            st.divider()
            st.subheader("Fatigue Settings")
            
            max_consecutive = st.slider(
                "Max Consecutive Stints",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum stints a player can play in a row before mandatory rest"
            )
            
            fatigue_penalty = st.slider(
                "Fatigue Penalty (%)",
                min_value=5,
                max_value=30,
                value=15,
                help="Performance reduction per consecutive stint"
            ) / 100
            
            st.divider()
            st.subheader("Fairness Settings")
            
            min_playing_pct = st.slider(
                "Minimum Playing Time (%)",
                min_value=0,
                max_value=50,
                value=25,
                help="Minimum percentage of game each player should play"
            ) / 100
            
            min_hist_minutes = st.slider(
                "Min Historical Minutes",
                min_value=0,
                max_value=200,
                value=50,
                help="Minimum historical minutes for player eligibility"
            )
            
            plan_btn = st.button("Generate Rotation Plan", type="primary", use_container_width=True)
        
        with col2:
            if plan_btn:
                with st.spinner("Optimizing rotations..."):
                    result = plan_game_rotation(
                        player_impact,
                        num_stints=num_stints,
                        stint_duration=stint_duration,
                        max_rating=8.0,
                        max_consecutive_stints=max_consecutive,
                        fatigue_penalty=fatigue_penalty,
                        min_playing_time_pct=min_playing_pct,
                        min_minutes=min_hist_minutes
                    )
                
                if result["status"] == "Optimal":
                    st.success(f"Rotation plan generated for {result['game_duration_min']:.0f} minute game!")
                    
                    # Summary metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Total Performance", f"{result['total_game_performance']:+.2f}")
                    with m2:
                        st.metric("Avg per Stint", f"{result['avg_performance_per_stint']:+.2f}")
                    with m3:
                        players_meeting_min = sum(1 for p in result['player_summary'] if p['met_minimum'])
                        st.metric("Players Meeting Min Time", f"{players_meeting_min}/{len(result['player_summary'])}")
                    
                    st.divider()
                    
                    # Rotation timeline
                    st.subheader("Rotation Timeline")
                    
                    # Create timeline visualization
                    timeline_data = []
                    for stint in result["rotation_plan"]:
                        for p in stint["players"]:
                            timeline_data.append({
                                "Stint": f"Stint {stint['stint']}",
                                "Player": format_player_name(p),
                                "Value": 1
                            })
                    
                    if timeline_data:
                        timeline_df = pd.DataFrame(timeline_data)
                        
                        # Create heatmap-style visualization
                        pivot_df = timeline_df.pivot_table(
                            index="Player", 
                            columns="Stint", 
                            values="Value", 
                            fill_value=0
                        )
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=pivot_df.values,
                            x=pivot_df.columns,
                            y=pivot_df.index,
                            colorscale=[[0, '#1e293b'], [1, '#22c55e']],
                            showscale=False,
                            hovertemplate="Player: %{y}<br>%{x}<br>Status: %{z:Playing/Resting}<extra></extra>"
                        ))
                        fig.update_layout(
                            height=300 + len(pivot_df) * 20,
                            xaxis_title="Game Progression",
                            yaxis_title="Players",
                            margin=dict(t=30)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    
                    # Detailed stint breakdown
                    st.subheader("Stint-by-Stint Breakdown")
                    
                    for stint in result["rotation_plan"]:
                        fatigue_color = "green" if stint["avg_fatigue_level"] < 0.1 else "orange" if stint["avg_fatigue_level"] < 0.2 else "red"
                        with st.expander(f"Stint {stint['stint']}: {' | '.join([format_player_name(p) for p in stint['players']])}"):
                            cols = st.columns(4)
                            for i, detail in enumerate(stint["details"]):
                                with cols[i]:
                                    st.markdown(f"**{format_player_name(detail['player'])}**")
                                    st.write(f"Rating: {detail['rating']:.1f}")
                                    st.write(f"Base Perf: {detail['base_perf']:+.2f}")
                                    st.write(f"Effective: {detail['effective_perf']:+.2f}")
                                    fatigue_pct = (1 - detail['fatigue_multiplier']) * 100
                                    st.write(f"Fatigue: {fatigue_pct:.0f}%")
                            
                            st.markdown(f"**Stint Total Rating:** {stint['total_rating']:.1f} | **Expected Perf:** {stint['expected_performance']:+.2f}")
                    
                    st.divider()
                    
                    # Player playing time summary
                    st.subheader("Player Playing Time Distribution")
                    
                    summary_df = pd.DataFrame(result["player_summary"])
                    summary_df["player"] = summary_df["player"].apply(format_player_name)
                    summary_df = summary_df.sort_values("stints_played", ascending=False)
                    
                    fig = px.bar(
                        summary_df,
                        x="player",
                        y="playing_time_min",
                        color="met_minimum",
                        color_discrete_map={True: "#22c55e", False: "#ef4444"},
                        labels={"player": "Player", "playing_time_min": "Playing Time (min)", "met_minimum": "Met Minimum"},
                        title="Playing Time Distribution"
                    )
                    fig.add_hline(
                        y=num_stints * stint_duration * min_playing_pct,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text=f"Min Required ({min_playing_pct*100:.0f}%)"
                    )
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    
                    # Summary table
                    display_summary = summary_df[["player", "rating", "stints_played", "playing_time_min", "playing_time_pct"]].copy()
                    display_summary.columns = ["Player", "Rating", "Stints", "Minutes", "% of Game"]
                    display_summary["% of Game"] = (display_summary["% of Game"] * 100).round(1).astype(str) + "%"
                    st.dataframe(display_summary, use_container_width=True, hide_index=True)
                    
                else:
                    st.error(result.get("message", "Could not generate rotation plan"))
                    st.info("Try adjusting parameters: increase max consecutive stints or decrease minimum playing time requirement.")
            
            else:
                # Show explanation when not generating
                st.info("""
                **How the Rotation Planner Works:**
                
                1. **Knapsack Problem**: Each stint is a knapsack where we select 4 players (items) 
                   with ratings (weights) that sum to ≤ 8.0 (capacity), maximizing performance (value).
                
                2. **Fatigue Modeling**: Players lose 15% performance per consecutive stint played.
                   After 3 consecutive stints, they must rest.
                
                3. **Fair Distribution**: The algorithm ensures all players get at least 25% playing time
                   (configurable), promoting team involvement.
                
                4. **Dynamic Adjustment**: Each stint considers the current fatigue state of all players
                   and prioritizes rested players while respecting constraints.
                
                **Adjust the parameters on the left and click "Generate Rotation Plan" to see your optimal game strategy.**
                """)
    
    # =========================================================================
    # TAB 4: Player Analysis
    # =========================================================================
    with tab4:
        st.header(f"Individual Player Analysis - {selected_team}")
        
        # Player selector using display names
        player_name_map = dict(zip(player_impact["display_name"], player_impact["player"]))
        selected_display_name = st.selectbox(
            "Select Player to Analyze",
            player_impact["display_name"].tolist()
        )
        selected_player = player_name_map[selected_display_name]
        
        player_row = player_impact[player_impact["player"] == selected_player].iloc[0]
        
        # Player metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rating", f"{player_row['rating']:.1f}")
        with col2:
            st.metric("Minutes Played", f"{player_row['minutes']:.0f}")
        with col3:
            st.metric("Goal Differential", f"{player_row['goal_diff']:+.0f}")
        with col4:
            st.metric("+/- per 10 min", f"{player_row['diff_per_10_min']:+.2f}")
        
        st.divider()
        
        # All players comparison
        st.subheader("All Players Comparison")
        
        fig = px.scatter(
            player_impact,
            x="minutes",
            y="diff_per_10_min",
            size="rating",
            color="diff_per_10_min",
            color_continuous_scale="RdYlGn",
            hover_name="display_name",
            labels={
                "minutes": "Total Minutes Played",
                "diff_per_10_min": "Goal Diff per 10 min",
                "rating": "Player Rating"
            }
        )
        # Highlight selected player
        player_trace = player_impact[player_impact["player"] == selected_player]
        fig.add_trace(go.Scatter(
            x=player_trace["minutes"],
            y=player_trace["diff_per_10_min"],
            mode="markers",
            marker=dict(size=20, color="yellow", line=dict(width=3, color="black")),
            name=selected_display_name,
            showlegend=True
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        # Full table
        st.subheader("Complete Player Statistics")
        display_df = player_impact[["display_name", "rating", "minutes", "goals_for", "goals_against", 
                                    "goal_diff", "diff_per_10_min"]].copy()
        display_df.columns = ["Player", "Rating", "Minutes", "Goals For", "Goals Against", 
                             "Goal Diff", "+/- per 10min"]
        st.dataframe(
            display_df.style.background_gradient(subset=["+/- per 10min"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True
        )
    
    # =========================================================================
    # TAB 5: Opponent Scouting
    # =========================================================================
    with tab5:
        st.header("Opponent Scouting Report")
        
        opponent = st.selectbox(
            "Select Opponent",
            [t for t in teams if t != selected_team]
        )
        
        # Get opponent stats
        opp_stats = team_stats[team_stats["team"] == opponent].iloc[0]
        opp_h2h = h2h[h2h["opponent"] == opponent]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{opponent} Profile")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Win Rate", f"{opp_stats['win_rate']:.1%}")
                st.metric("Goals/Game", f"{opp_stats['avg_goals_for']:.1f}")
            with m2:
                opp_rank = team_stats[team_stats["win_rate"] >= opp_stats["win_rate"]].shape[0]
                st.metric("League Rank", f"#{opp_rank}")
                st.metric("Goals Against/Game", f"{opp_stats['avg_goals_against']:.1f}")
        
        with col2:
            st.subheader(f"Your Record vs {opponent}")
            
            if len(opp_h2h) > 0:
                h2h_row = opp_h2h.iloc[0]
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Win Rate", f"{h2h_row['win_rate']:.1%}")
                    st.metric("Wins", int(h2h_row["wins"]))
                with m2:
                    st.metric("Losses", int(h2h_row["losses"]))
                    st.metric("Goal Diff", f"{h2h_row['goal_diff']:+.0f}")
            else:
                st.info("No head-to-head data available")
        
        st.divider()
        
        # Recommendation
        st.subheader("Recommended Strategy")
        
        if len(opp_h2h) > 0:
            h2h_row = opp_h2h.iloc[0]
            if h2h_row["win_rate"] >= 0.6:
                st.success(f"""
                **Favorable matchup!** You have a {h2h_row['win_rate']:.0%} win rate against {opponent}.
                
                **Recommendation:** Play your standard top lineup with confidence.
                """)
            elif h2h_row["win_rate"] <= 0.4:
                st.warning(f"""
                **Challenging matchup.** Only {h2h_row['win_rate']:.0%} win rate against {opponent}.
                
                **Recommendation:** Consider lineup adjustments. Focus on defensive players 
                and high-experience combinations.
                """)
            else:
                st.info(f"""
                **Competitive matchup.** {h2h_row['win_rate']:.0%} win rate - could go either way.
                
                **Recommendation:** Small edges matter. Use the optimizer to find your best lineup.
                """)
        
        # Opponent-specific lineup optimization
        st.subheader(f"Optimize Lineup vs {opponent}")
        
        if st.button(f"Find Best Lineup Against {opponent}", type="primary"):
            result = optimize_lineup(player_impact, max_rating=8.0, min_minutes=100)
            if result["status"] == "Optimal":
                display_players = [format_player_name(p) for p in result['players']]
                st.success(f"**Recommended lineup:** {' - '.join(display_players)}")
                st.write(f"Rating: {result['total_rating']:.1f} | Expected +/-: {result['avg_diff_per_10_min']:+.2f}/10min")
    
    # =========================================================================
    # TAB 6: Historical Lineups
    # =========================================================================
    with tab6:
        st.header("Historical Lineup Performance")
        
        min_lineup_minutes = st.slider(
            "Minimum Lineup Minutes",
            min_value=5,
            max_value=50,
            value=10,
            help="Filter lineups by total minutes played together"
        )
        
        filtered_lineups = lineup_perf[lineup_perf["minutes"] >= min_lineup_minutes].head(20)
        
        if len(filtered_lineups) > 0:
            # Top performing lineups
            st.subheader("Best Performing Lineups")
            
            for i, row in filtered_lineups.head(5).iterrows():
                players = row["lineup"].split("|")
                display_lineup = " | ".join([format_player_name(p) for p in players])
                with st.expander(f"#{i+1}: {display_lineup} (+{row['diff_per_10_min']:.2f}/10min)"):
                    cols = st.columns(4)
                    for j, p in enumerate(players):
                        with cols[j]:
                            p_data = player_impact[player_impact["player"] == p]
                            if len(p_data) > 0:
                                st.write(f"**{format_player_name(p)}**")
                                st.write(f"Rating: {p_data.iloc[0]['rating']:.1f}")
                            else:
                                st.write(f"**{format_player_name(p)}**")
                    st.write(f"**Minutes together:** {row['minutes']:.1f}")
                    st.write(f"**Goals For:** {row['goals_for']} | **Against:** {row['goals_against']}")
            
            # Chart
            st.subheader("Lineup Performance Distribution")
            fig = px.histogram(
                lineup_perf[lineup_perf["minutes"] >= min_lineup_minutes],
                x="diff_per_10_min",
                nbins=30,
                labels={"diff_per_10_min": "Goal Diff per 10 min"}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("No lineups found with sufficient minutes. Try lowering the threshold.")
    


if __name__ == "__main__":
    main()
