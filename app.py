"""
Football Prediction Platform
----------------------------

Streamlit + Football-Data.org API v4

Supported competitions:
    - Premier League (PL)
    - La Liga (PD)
    - Serie A (SA)

Model:
    - Recent home/away performance
    - League scoring baseline
    - Attack/defence strength
    - Poisson goal model
    - Scoreline probability matrix

Markets:
    - Home / Draw / Away
    - Double Chance
    - Draw No Bet
    - Over / Under 0.5, 1.5, 2.5, 3.5, 4.5
    - BTTS Yes / No
    - Home team goal totals
    - Away team goal totals
    - BTTS + Result
    - Correct Score

Backtesting:
    - 1X2 accuracy
    - Brier score
    - Log loss
    - Market accuracy
    - Historical prediction tracking

IMPORTANT:
    This application produces statistical estimates.
    It does not guarantee betting outcomes.
"""

import os
import math
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Prediction Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

API_BASE = "https://api.football-data.org/v4"

COMPETITIONS = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
}

HISTORY_FILE = Path("prediction_history.csv")

DEFAULT_HISTORY_MATCHES = 10
DEFAULT_FIXTURE_DAYS = 14
DEFAULT_MAX_GOALS = 7
DEFAULT_MIN_HISTORY = 5

REQUEST_TIMEOUT = 20


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(128,128,128,0.06);
        }

        .prediction-box {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 0.7rem;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 10px;
            background: rgba(255,193,7,0.10);
            border: 1px solid rgba(255,193,7,0.35);
        }

        .small-text {
            font-size: 0.85rem;
            color: #777;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    """
    Gets API key from Streamlit secrets first.
    Falls back to environment variable.
    Finally allows sidebar entry for local testing.
    """

    secret_key = None

    try:
        secret_key = st.secrets.get("FOOTBALL_DATA_API_KEY")
    except Exception:
        secret_key = None

    if secret_key:
        return secret_key

    env_key = os.getenv("FOOTBALL_DATA_API_KEY")

    if env_key:
        return env_key

    return st.sidebar.text_input(
        "Football-Data.org API Key",
        type="password",
        help="For GitHub/Streamlit Cloud, store this as FOOTBALL_DATA_API_KEY in Streamlit secrets.",
    )


# ============================================================
# API REQUEST HELPER
# ============================================================

def api_get(endpoint, params=None, api_key=None):
    """
    Central API request function.
    """

    if not api_key:
        raise ValueError("Football-Data.org API key is missing.")

    url = f"{API_BASE}{endpoint}"

    headers = {
        "X-Auth-Token": api_key,
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error while contacting Football-Data.org: {exc}")

    if response.status_code == 429:
        reset_seconds = response.headers.get(
            "X-RequestCounter-Reset",
            "unknown",
        )

        raise RuntimeError(
            f"Football-Data.org rate limit reached. "
            f"Try again later. Reset information: {reset_seconds}."
        )

    if response.status_code == 401:
        raise RuntimeError(
            "The Football-Data.org API key was rejected. "
            "Check that the token is correct."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "Football-Data.org restricted this resource for your API plan."
        )

    if response.status_code >= 400:
        try:
            error_data = response.json()
            error_message = error_data.get("message") or error_data.get("error")
        except Exception:
            error_message = response.text

        raise RuntimeError(
            f"Football-Data.org returned HTTP {response.status_code}: "
            f"{error_message}"
        )

    try:
        return response.json()
    except ValueError:
        raise RuntimeError("Football-Data.org returned an invalid JSON response.")


# ============================================================
# API FUNCTIONS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_competition_matches(
    competition_code,
    api_key,
    season=None,
    date_from=None,
    date_to=None,
    status=None,
):
    """
    Retrieve competition matches.

    Cached for 15 minutes to reduce API requests.
    """

    params = {}

    if season:
        params["season"] = season

    if date_from:
        params["dateFrom"] = date_from

    if date_to:
        params["dateTo"] = date_to

    if status:
        params["status"] = status

    data = api_get(
        f"/competitions/{competition_code}/matches",
        params=params,
        api_key=api_key,
    )

    matches = data.get("matches", [])

    return matches


@st.cache_data(ttl=1800, show_spinner=False)
def get_competition_info(competition_code, api_key):
    """
    Retrieve competition metadata.
    """

    return api_get(
        f"/competitions/{competition_code}",
        api_key=api_key,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def get_team_matches(
    team_id,
    api_key,
    season=None,
    competition_code=None,
    status="FINISHED",
):
    """
    Retrieve matches for a team.
    """

    params = {
        "status": status,
        "limit": 100,
    }

    if season:
        params["season"] = season

    if competition_code:
        params["competitions"] = competition_code

    data = api_get(
        f"/teams/{team_id}/matches/",
        params=params,
        api_key=api_key,
    )

    return data.get("matches", [])


@st.cache_data(ttl=600, show_spinner=False)
def get_match_by_id(match_id, api_key):
    """
    Retrieve a specific match.
    """

    return api_get(
        f"/matches/{match_id}",
        api_key=api_key,
    )


# ============================================================
# DATA HELPERS
# ============================================================

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def parse_match_date(match):
    """
    Convert UTC match date to pandas Timestamp.
    """

    value = match.get("utcDate")

    if not value:
        return pd.NaT

    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return pd.NaT


def get_full_time_score(match):
    """
    Return home and away full-time scores.

    Returns None if the match does not have a completed score.
    """

    score = match.get("score", {})
    full_time = score.get("fullTime", {})

    home = full_time.get("home")
    away = full_time.get("away")

    if home is None or away is None:
        return None

    return safe_int(home), safe_int(away)


def is_finished(match):
    """
    Determine whether a match has a usable final score.
    """

    status = match.get("status")

    if status == "FINISHED":
        return get_full_time_score(match) is not None

    return False


def match_team_names(match):
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})

    return (
        home.get("name", "Unknown"),
        away.get("name", "Unknown"),
    )


# ============================================================
# TEAM HISTORY
# ============================================================

def extract_team_history(
    matches,
    team_id,
    before_date=None,
    limit=10,
):
    """
    Extract only the team's completed matches occurring
    before the target fixture.

    This is important for avoiding future-data leakage.
    """

    records = []

    for match in matches:

        if not is_finished(match):
            continue

        match_date = parse_match_date(match)

        if pd.isna(match_date):
            continue

        if before_date is not None and match_date >= before_date:
            continue

        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})

        home_id = home_team.get("id")
        away_id = away_team.get("id")

        score = get_full_time_score(match)

        if score is None:
            continue

        home_goals, away_goals = score

        if home_id == team_id:

            records.append(
                {
                    "date": match_date,
                    "home_team": home_team.get("name"),
                    "away_team": away_team.get("name"),
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                    "venue": "HOME",
                    "result": (
                        "W"
                        if home_goals > away_goals
                        else "D"
                        if home_goals == away_goals
                        else "L"
                    ),
                }
            )

        elif away_id == team_id:

            records.append(
                {
                    "date": match_date,
                    "home_team": home_team.get("name"),
                    "away_team": away_team.get("name"),
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                    "venue": "AWAY",
                    "result": (
                        "W"
                        if away_goals > home_goals
                        else "D"
                        if home_goals == away_goals
                        else "L"
                    ),
                }
            )

    records = sorted(
        records,
        key=lambda x: x["date"],
        reverse=True,
    )

    return records[:limit]


# ============================================================
# STATISTICS
# ============================================================

def calculate_average(records, key):
    if not records:
        return 0.0

    values = [safe_float(record.get(key)) for record in records]

    return float(np.mean(values))


def calculate_weighted_average(records, key):
    """
    More recent matches receive greater weight.

    Most recent match has the highest weight.
    """

    if not records:
        return 0.0

    values = np.array(
        [safe_float(record.get(key)) for record in records],
        dtype=float,
    )

    weights = np.arange(
        len(values),
        0,
        -1,
        dtype=float,
    )

    return float(np.average(values, weights=weights))


def calculate_team_stats(history):
    """
    Calculate basic form statistics.
    """

    if not history:
        return {
            "matches": 0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
        }

    results = [r["result"] for r in history]

    return {
        "matches": len(history),
        "goals_for": calculate_weighted_average(
            history,
            "goals_for",
        ),
        "goals_against": calculate_weighted_average(
            history,
            "goals_against",
        ),
        "win_rate": results.count("W") / len(results),
        "draw_rate": results.count("D") / len(results),
        "loss_rate": results.count("L") / len(results),
    }


# ============================================================
# LEAGUE BASELINES
# ============================================================

def calculate_league_baseline(
    completed_matches,
):
    """
    Calculate league-wide average goals.

    Used as a baseline so the model isn't based only
    on two teams' raw averages.
    """

    rows = []

    for match in completed_matches:

        score = get_full_time_score(match)

        if score is None:
            continue

        home_goals, away_goals = score

        rows.append(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    if not rows:
        return {
            "home_goals": 1.40,
            "away_goals": 1.10,
            "total_goals": 2.50,
        }

    df = pd.DataFrame(rows)

    home_average = df["home_goals"].mean()
    away_average = df["away_goals"].mean()

    return {
        "home_goals": float(home_average),
        "away_goals": float(away_average),
        "total_goals": float(
            home_average + away_average
        ),
    }


# ============================================================
# EXPECTED GOALS MODEL
# ============================================================

def calculate_expected_goals(
    home_history,
    away_history,
    league_baseline,
):
    """
    Calculate expected goals.

    The model combines:

        - team's recent scoring
        - team's recent conceding
        - league baseline
        - venue-specific information
        - smoothing

    It is deliberately transparent rather than a black box.
    """

    league_home = league_baseline["home_goals"]
    league_away = league_baseline["away_goals"]

    if not home_history:
        home_attack = league_home
        home_defence = league_away
    else:
        home_home_games = [
            x for x in home_history
            if x["venue"] == "HOME"
        ]

        if not home_home_games:
            home_home_games = home_history

        home_attack = calculate_weighted_average(
            home_home_games,
            "goals_for",
        )

        home_defence = calculate_weighted_average(
            home_home_games,
            "goals_against",
        )

    if not away_history:
        away_attack = league_away
        away_defence = league_home
    else:
        away_away_games = [
            x for x in away_history
            if x["venue"] == "AWAY"
        ]

        if not away_away_games:
            away_away_games = away_history

        away_attack = calculate_weighted_average(
            away_away_games,
            "goals_for",
        )

        away_defence = calculate_weighted_average(
            away_away_games,
            "goals_against",
        )

    # --------------------------------------------------------
    # Attack / defence blend
    # --------------------------------------------------------

    raw_home_xg = (
        (home_attack * 0.60)
        + (away_defence * 0.40)
    )

    raw_away_xg = (
        (away_attack * 0.60)
        + (home_defence * 0.40)
    )

    # --------------------------------------------------------
    # League smoothing
    # --------------------------------------------------------

    home_xg = (
        raw_home_xg * 0.75
        + league_home * 0.25
    )

    away_xg = (
        raw_away_xg * 0.75
        + league_away * 0.25
    )

    # Prevent unrealistic values.
    home_xg = float(
        np.clip(home_xg, 0.15, 4.50)
    )

    away_xg = float(
        np.clip(away_xg, 0.15, 4.50)
    )

    return home_xg, away_xg


# ============================================================
# POISSON MODEL
# ============================================================

def poisson_score_matrix(
    home_xg,
    away_xg,
    max_goals=7,
):
    """
    Generate probability of every scoreline.
    """

    home_probs = poisson.pmf(
        np.arange(max_goals + 1),
        home_xg,
    )

    away_probs = poisson.pmf(
        np.arange(max_goals + 1),
        away_xg,
    )

    matrix = np.outer(
        home_probs,
        away_probs,
    )

    # Normalize because scorelines above max_goals
    # have been truncated.
    total = matrix.sum()

    if total > 0:
        matrix = matrix / total

    return matrix


# ============================================================
# MARKET CALCULATIONS
# ============================================================

def calculate_markets(
    matrix,
    max_goals=7,
):
    """
    Convert scoreline probabilities into market probabilities.
    """

    markets = {}

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    btts_yes = 0.0

    over_05 = 0.0
    over_15 = 0.0
    over_25 = 0.0
    over_35 = 0.0
    over_45 = 0.0

    home_over_05 = 0.0
    home_over_15 = 0.0
    home_over_25 = 0.0

    away_over_05 = 0.0
    away_over_15 = 0.0
    away_over_25 = 0.0

    correct_scores = []

    for home_goals in range(max_goals + 1):

        for away_goals in range(max_goals + 1):

            probability = float(
                matrix[home_goals, away_goals]
            )

            total_goals = home_goals + away_goals

            if home_goals > away_goals:
                home_win += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away_win += probability

            if home_goals > 0 and away_goals > 0:
                btts_yes += probability

            if total_goals > 0:
                over_05 += probability

            if total_goals > 1:
                over_15 += probability

            if total_goals > 2:
                over_25 += probability

            if total_goals > 3:
                over_35 += probability

            if total_goals > 4:
                over_45 += probability

            if home_goals > 0:
                home_over_05 += probability

            if home_goals > 1:
                home_over_15 += probability

            if home_goals > 2:
                home_over_25 += probability

            if away_goals > 0:
                away_over_05 += probability

            if away_goals > 1:
                away_over_15 += probability

            if away_goals > 2:
                away_over_25 += probability

            correct_scores.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "probability": probability,
                }
            )

    markets["Home Win"] = home_win
    markets["Draw"] = draw
    markets["Away Win"] = away_win

    markets["1X"] = home_win + draw
    markets["12"] = home_win + away_win
    markets["X2"] = draw + away_win

    # Draw No Bet probabilities
    # These represent the probability of the team winning
    # among non-draw outcomes.
    non_draw = home_win + away_win

    if non_draw > 0:
        markets["Home DNB"] = home_win / non_draw
        markets["Away DNB"] = away_win / non_draw
    else:
        markets["Home DNB"] = 0.0
        markets["Away DNB"] = 0.0

    markets["Over 0.5"] = over_05
    markets["Under 0.5"] = 1 - over_05

    markets["Over 1.5"] = over_15
    markets["Under 1.5"] = 1 - over_15

    markets["Over 2.5"] = over_25
    markets["Under 2.5"] = 1 - over_25

    markets["Over 3.5"] = over_35
    markets["Under 3.5"] = 1 - over_35

    markets["Over 4.5"] = over_45
    markets["Under 4.5"] = 1 - over_45

    markets["BTTS Yes"] = btts_yes
    markets["BTTS No"] = 1 - btts_yes

    markets["Home Over 0.5"] = home_over_05
    markets["Home Under 0.5"] = 1 - home_over_05

    markets["Home Over 1.5"] = home_over_15
    markets["Home Under 1.5"] = 1 - home_over_15

    markets["Home Over 2.5"] = home_over_25
    markets["Home Under 2.5"] = 1 - home_over_25

    markets["Away Over 0.5"] = away_over_05
    markets["Away Under 0.5"] = 1 - away_over_05

    markets["Away Over 1.5"] = away_over_15
    markets["Away Under 1.5"] = 1 - away_over_15

    markets["Away Over 2.5"] = away_over_25
    markets["Away Under 2.5"] = 1 - away_over_25

    # --------------------------------------------------------
    # BTTS + Result
    # --------------------------------------------------------

    home_btts = 0.0
    draw_btts = 0.0
    away_btts = 0.0

    for home_goals in range(max_goals + 1):

        for away_goals in range(max_goals + 1):

            probability = float(
                matrix[home_goals, away_goals]
            )

            if home_goals > 0 and away_goals > 0:

                if home_goals > away_goals:
                    home_btts += probability

                elif home_goals == away_goals:
                    draw_btts += probability

                else:
                    away_btts += probability

    markets["Home Win + BTTS"] = home_btts
    markets["Draw + BTTS"] = draw_btts
    markets["Away Win + BTTS"] = away_btts

    # --------------------------------------------------------
    # Correct scores
    # --------------------------------------------------------

    correct_scores = sorted(
        correct_scores,
        key=lambda x: x["probability"],
        reverse=True,
    )

    markets["_correct_scores"] = correct_scores

    return markets


# ============================================================
# PREDICTION SELECTION
# ============================================================

PREFERRED_MARKETS = [
    "Home Win",
    "Draw",
    "Away Win",
    "1X",
    "12",
    "X2",
    "Over 1.5",
    "Under 1.5",
    "Over 2.5",
    "Under 2.5",
    "Over 3.5",
    "Under 3.5",
    "BTTS Yes",
    "BTTS No",
    "Home DNB",
    "Away DNB",
]


def select_best_prediction(
    markets,
    minimum_probability=0.60,
):
    """
    Select the highest-probability supported market
    above the configured threshold.

    This is a model selection, not a recommendation
    or guarantee.
    """

    candidates = []

    for market in PREFERRED_MARKETS:

        probability = markets.get(market)

        if probability is None:
            continue

        candidates.append(
            (
                market,
                float(probability),
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    if not candidates:
        return None, 0.0

    best_market, best_probability = candidates[0]

    if best_probability >= minimum_probability:
        return best_market, best_probability

    return "No threshold selection", best_probability


def confidence_label(probability):
    if probability >= 0.80:
        return "Very High"
    elif probability >= 0.70:
        return "High"
    elif probability >= 0.60:
        return "Moderate"
    elif probability >= 0.50:
        return "Low"
    return "Very Low"


# ============================================================
# COMPLETE MATCH PREDICTION
# ============================================================

def predict_match(
    match,
    all_completed_matches,
    history_matches=10,
    max_goals=7,
    minimum_probability=0.60,
):
    """
    Generate a complete prediction for one fixture.
    """

    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})

    home_id = home_team.get("id")
    away_id = away_team.get("id")

    home_name = home_team.get(
        "name",
        "Unknown Home Team",
    )

    away_name = away_team.get(
        "name",
        "Unknown Away Team",
    )

    match_date = parse_match_date(match)

    # --------------------------------------------------------
    # Historical form
    # --------------------------------------------------------

    home_history = extract_team_history(
        all_completed_matches,
        home_id,
        before_date=match_date,
        limit=history_matches,
    )

    away_history = extract_team_history(
        all_completed_matches,
        away_id,
        before_date=match_date,
        limit=history_matches,
    )

    league_baseline = calculate_league_baseline(
        [
            m for m in all_completed_matches
            if parse_match_date(m) < match_date
            and is_finished(m)
        ]
    )

    # --------------------------------------------------------
    # Expected goals
    # --------------------------------------------------------

    home_xg, away_xg = calculate_expected_goals(
        home_history,
        away_history,
        league_baseline,
    )

    # --------------------------------------------------------
    # Poisson matrix
    # --------------------------------------------------------

    matrix = poisson_score_matrix(
        home_xg,
        away_xg,
        max_goals=max_goals,
    )

    # --------------------------------------------------------
    # Markets
    # --------------------------------------------------------

    markets = calculate_markets(
        matrix,
        max_goals=max_goals,
    )

    best_market, best_probability = select_best_prediction(
        markets,
        minimum_probability=minimum_probability,
    )

    correct_scores = markets.pop(
        "_correct_scores",
        [],
    )

    top_scores = correct_scores[:5]

    return {
        "match_id": match.get("id"),
        "competition": match.get(
            "competition",
            {},
        ).get(
            "name",
            "Unknown Competition",
        ),
        "date": match_date,
        "home_team": home_name,
        "away_team": away_name,
        "home_id": home_id,
        "away_id": away_id,
        "home_history": home_history,
        "away_history": away_history,
        "home_stats": calculate_team_stats(
            home_history
        ),
        "away_stats": calculate_team_stats(
            away_history
        ),
        "league_baseline": league_baseline,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "matrix": matrix,
        "markets": markets,
        "top_scores": top_scores,
        "best_market": best_market,
        "best_probability": best_probability,
        "confidence": confidence_label(
            best_probability
        ),
    }


# ============================================================
# RESULT EVALUATION
# ============================================================

def actual_market_result(
    home_goals,
    away_goals,
    market,
):
    """
    Determine whether a market was correct.

    Returns:
        True / False
    """

    total = home_goals + away_goals

    if market == "Home Win":
        return home_goals > away_goals

    if market == "Draw":
        return home_goals == away_goals

    if market == "Away Win":
        return away_goals > home_goals

    if market == "1X":
        return home_goals >= away_goals

    if market == "12":
        return home_goals != away_goals

    if market == "X2":
        return away_goals >= home_goals

    if market == "Over 0.5":
        return total > 0

    if market == "Under 0.5":
        return total == 0

    if market == "Over 1.5":
        return total > 1

    if market == "Under 1.5":
        return total <= 1

    if market == "Over 2.5":
        return total > 2

    if market == "Under 2.5":
        return total <= 2

    if market == "Over 3.5":
        return total > 3

    if market == "Under 3.5":
        return total <= 3

    if market == "Over 4.5":
        return total > 4

    if market == "Under 4.5":
        return total <= 4

    if market == "BTTS Yes":
        return (
            home_goals > 0
            and away_goals > 0
        )

    if market == "BTTS No":
        return (
            home_goals == 0
            or away_goals == 0
        )

    if market == "Home Over 0.5":
        return home_goals > 0

    if market == "Home Under 0.5":
        return home_goals == 0

    if market == "Home Over 1.5":
        return home_goals > 1

    if market == "Home Under 1.5":
        return home_goals <= 1

    if market == "Home Over 2.5":
        return home_goals > 2

    if market == "Home Under 2.5":
        return home_goals <= 2

    if market == "Away Over 0.5":
        return away_goals > 0

    if market == "Away Under 0.5":
        return away_goals == 0

    if market == "Away Over 1.5":
        return away_goals > 1

    if market == "Away Under 1.5":
        return away_goals <= 1

    if market == "Away Over 2.5":
        return away_goals > 2

    if market == "Away Under 2.5":
        return away_goals <= 2

    return None


# ============================================================
# BRIER SCORE
# ============================================================

def brier_score(predictions, outcomes):
    """
    Binary Brier score.
    Lower is better.

    BS = mean((probability - outcome)^2)
    """

    if not predictions:
        return None

    predictions = np.array(predictions)
    outcomes = np.array(outcomes)

    return float(
        np.mean(
            (predictions - outcomes) ** 2
        )
    )


# ============================================================
# LOG LOSS
# ============================================================

def log_loss_binary(
    probabilities,
    outcomes,
):
    """
    Binary logarithmic loss.
    Lower is better.
    """

    if not probabilities:
        return None

    eps = 1e-15

    probabilities = np.clip(
        np.array(probabilities),
        eps,
        1 - eps,
    )

    outcomes = np.array(outcomes)

    loss = -np.mean(
        outcomes * np.log(probabilities)
        + (
            1 - outcomes
        )
        * np.log(1 - probabilities)
    )

    return float(loss)


# ============================================================
# HISTORY FILE
# ============================================================

def load_prediction_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            HISTORY_FILE
        )
    except Exception:
        return pd.DataFrame()


def save_prediction_history(rows):
    """
    Save prediction records locally.
    """

    if not rows:
        return

    new_df = pd.DataFrame(rows)

    existing = load_prediction_history()

    if not existing.empty:
        combined = pd.concat(
            [
                existing,
                new_df,
            ],
            ignore_index=True,
        )
    else:
        combined = new_df

    if "match_id" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["match_id"],
            keep="last",
        )

    combined.to_csv(
        HISTORY_FILE,
        index=False,
    )


def create_prediction_history_row(
    prediction,
):
    return {
        "saved_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "match_id": prediction["match_id"],

        "competition": prediction[
            "competition"
        ],

        "match_date": (
            prediction["date"].isoformat()
            if not pd.isna(prediction["date"])
            else ""
        ),

        "home_team": prediction[
            "home_team"
        ],

        "away_team": prediction[
            "away_team"
        ],

        "home_xg": prediction[
            "home_xg"
        ],

        "away_xg": prediction[
            "away_xg"
        ],

        "best_market": prediction[
            "best_market"
        ],

        "best_probability": prediction[
            "best_probability"
        ],

        "confidence": prediction[
            "confidence"
        ],

        "actual_home_goals": "",

        "actual_away_goals": "",

        "result_correct": "",
    }


# ============================================================
# UPDATE SAVED PREDICTIONS
# ============================================================

def update_prediction_results(
    history_df,
    api_key,
):
    """
    Automatically check saved prediction matches
    that should now have completed.

    This uses the Football-Data.org match ID.
    """

    if history_df.empty:
        return history_df

    required_columns = {
        "match_id",
        "best_market",
        "actual_home_goals",
        "actual_away_goals",
        "result_correct",
    }

    if not required_columns.issubset(
        history_df.columns
    ):
        return history_df

    updated = history_df.copy()

    for index, row in updated.iterrows():

        actual_home = row.get(
            "actual_home_goals"
        )

        actual_away = row.get(
            "actual_away_goals"
        )

        # Already updated.
        if (
            pd.notna(actual_home)
            and str(actual_home).strip() != ""
            and pd.notna(actual_away)
            and str(actual_away).strip() != ""
        ):
            continue

        match_id = row.get(
            "match_id"
        )

        if pd.isna(match_id):
            continue

        try:
            match = get_match_by_id(
                int(match_id),
                api_key,
            )
        except Exception:
            continue

        if not is_finished(match):
            continue

        score = get_full_time_score(
            match
        )

        if score is None:
            continue

        home_goals, away_goals = score

        market = row.get(
            "best_market"
        )

        correct = actual_market_result(
            home_goals,
            away_goals,
            market,
        )

        updated.at[
            index,
            "actual_home_goals",
        ] = home_goals

        updated.at[
            index,
            "actual_away_goals",
        ] = away_goals

        updated.at[
            index,
            "result_correct",
        ] = (
            "WIN"
            if correct
            else "LOSS"
        )

    updated.to_csv(
        HISTORY_FILE,
        index=False,
    )

    return updated


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

api_key = get_api_key()

selected_leagues = st.sidebar.multiselect(
    "Leagues",
    list(COMPETITIONS.keys()),
    default=[
        "Premier League",
        "La Liga",
        "Serie A",
    ],
)

fixture_days = st.sidebar.slider(
    "Upcoming fixture window (days)",
    min_value=1,
    max_value=30,
    value=DEFAULT_FIXTURE_DAYS,
)

history_matches = st.sidebar.slider(
    "Recent matches used",
    min_value=3,
    max_value=20,
    value=DEFAULT_HISTORY_MATCHES,
)

minimum_probability = st.sidebar.slider(
    "Selection probability threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.01,
)

max_goals = st.sidebar.slider(
    "Maximum scoreline goals",
    min_value=5,
    max_value=10,
    value=DEFAULT_MAX_GOALS,
)

run_backtest_matches = st.sidebar.slider(
    "Backtest matches per league",
    min_value=10,
    max_value=100,
    value=30,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚽ Football Prediction Platform</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Poisson-based football prediction dashboard powered by
    Football-Data.org.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    """
    **Model note:** Predictions are statistical estimates based on
    historical league data. They are not guarantees of match outcomes
    or betting returns.
    """
)


# ============================================================
# API KEY CHECK
# ============================================================

if not api_key:

    st.warning(
        """
        Enter your Football-Data.org API key in the sidebar,
        or configure `FOOTBALL_DATA_API_KEY` in Streamlit secrets.
        """
    )

    st.stop()


# ============================================================
# DATE RANGE
# ============================================================

today = datetime.now(
    timezone.utc
).date()

future_date = today + timedelta(
    days=fixture_days
)

date_from = today.isoformat()
date_to = future_date.isoformat()


# ============================================================
# LOAD FIXTURES
# ============================================================

all_upcoming_predictions = []

errors = []

with st.spinner(
    "Loading fixtures and building predictions..."
):

    for league_name in selected_leagues:

        competition_code = COMPETITIONS[
            league_name
        ]

        try:

            # ------------------------------------------------
            # Upcoming fixtures
            # ------------------------------------------------

            upcoming_matches = (
                get_competition_matches(
                    competition_code,
                    api_key,
                    date_from=date_from,
                    date_to=date_to,
                )
            )

            # ------------------------------------------------
            # Full current-season competition data
            # ------------------------------------------------

            competition_info = (
                get_competition_info(
                    competition_code,
                    api_key,
                )
            )

            current_season = (
                competition_info
                .get("currentSeason", {})
                .get("startDate")
            )

            season_year = None

            if current_season:
                season_year = int(
                    str(current_season)[:4]
                )

            completed_matches = (
                get_competition_matches(
                    competition_code,
                    api_key,
                    season=season_year,
                    status="FINISHED",
                )
            )

            # ------------------------------------------------
            # Build prediction for every upcoming match
            # ------------------------------------------------

            for match in upcoming_matches:

                status = match.get(
                    "status"
                )

                if status not in [
                    "SCHEDULED",
                    "TIMED",
                    "POSTPONED",
                ]:
                    continue

                try:

                    prediction = predict_match(
                        match,
                        completed_matches,
                        history_matches=history_matches,
                        max_goals=max_goals,
                        minimum_probability=minimum_probability,
                    )

                    all_upcoming_predictions.append(
                        prediction
                    )

                except Exception as exc:

                    errors.append(
                        f"{league_name}: "
                        f"{match_team_names(match)} - "
                        f"{exc}"
                    )

        except Exception as exc:

            errors.append(
                f"{league_name}: {exc}"
            )


# ============================================================
# ERROR DISPLAY
# ============================================================

if errors:

    with st.expander(
        f"⚠️ Data/API messages ({len(errors)})"
    ):

        for error in errors:
            st.write(
                f"- {error}"
            )


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Fixtures",
        len(all_upcoming_predictions),
    )

with col2:
    st.metric(
        "Leagues",
        len(selected_leagues),
    )

with col3:
    if all_upcoming_predictions:
        avg_probability = np.mean(
            [
                p["best_probability"]
                for p in all_upcoming_predictions
            ]
        )
        st.metric(
            "Avg. model probability",
            f"{avg_probability:.1%}",
        )
    else:
        st.metric(
            "Avg. model probability",
            "—",
        )

with col4:
    high_confidence = sum(
        1
        for p in all_upcoming_predictions
        if p["best_probability"] >= 0.70
    )

    st.metric(
        "≥70% probability",
        high_confidence,
    )


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_details, tab_history, tab_backtest, tab_model = st.tabs(
    [
        "📊 Predictions",
        "🔎 Match Analysis",
        "📝 Prediction History",
        "🧪 Backtesting",
        "🧠 Model",
    ]
)


# ============================================================
# DASHBOARD TAB
# ============================================================

with tab_dashboard:

    st.subheader(
        "Upcoming Match Predictions"
    )

    if not all_upcoming_predictions:

        st.warning(
            "No upcoming fixtures were found for the selected leagues and date range."
        )

    else:

        dashboard_rows = []

        for prediction in all_upcoming_predictions:

            markets = prediction[
                "markets"
            ]

            dashboard_rows.append(
                {
                    "League": prediction[
                        "competition"
                    ],

                    "Date": prediction[
                        "date"
                    ].strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),

                    "Home": prediction[
                        "home_team"
                    ],

                    "Away": prediction[
                        "away_team"
                    ],

                    "Home xG": round(
                        prediction[
                            "home_xg"
                        ],
                        2,
                    ),

                    "Away xG": round(
                        prediction[
                            "away_xg"
                        ],
                        2,
                    ),

                    "Home Win": markets[
                        "Home Win"
                    ],

                    "Draw": markets[
                        "Draw"
                    ],

                    "Away Win": markets[
                        "Away Win"
                    ],

                    "Over 2.5": markets[
                        "Over 2.5"
                    ],

                    "BTTS": markets[
                        "BTTS Yes"
                    ],

                    "Model Selection": prediction[
                        "best_market"
                    ],

                    "Probability": prediction[
                        "best_probability"
                    ],

                    "Confidence": prediction[
                        "confidence"
                    ],
                }
            )

        dashboard_df = pd.DataFrame(
            dashboard_rows
        )

        percent_columns = [
            "Home Win",
            "Draw",
            "Away Win",
            "Over 2.5",
            "BTTS",
            "Probability",
        ]

        for column in percent_columns:
            dashboard_df[column] = (
                dashboard_df[column]
                .astype(float)
                .map(
                    lambda x: f"{x:.1%}"
                )
            )

        st.dataframe(
            dashboard_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "### Probability overview"
        )

        chart_df = pd.DataFrame(
            [
                {
                    "Match": (
                        f"{p['home_team']} "
                        f"vs "
                        f"{p['away_team']}"
                    ),
                    "Home Win": p[
                        "markets"
                    ]["Home Win"],
                    "Draw": p[
                        "markets"
                    ]["Draw"],
                    "Away Win": p[
                        "markets"
                    ]["Away Win"],
                }
                for p in all_upcoming_predictions
            ]
        )

        if not chart_df.empty:

            melted = chart_df.melt(
                id_vars="Match",
                var_name="Market",
                value_name="Probability",
            )

            fig = px.bar(
                melted,
                x="Match",
                y="Probability",
                color="Market",
                barmode="group",
                title="1X2 probability comparison",
            )

            fig.update_yaxes(
                tickformat=".0%",
                range=[0, 1],
            )

            fig.update_layout(
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        if st.button(
            "💾 Save current predictions",
            type="primary",
        ):

            rows = [
                create_prediction_history_row(
                    p
                )
                for p in all_upcoming_predictions
            ]

            save_prediction_history(
                rows
            )

            st.success(
                f"Saved {len(rows)} predictions to prediction_history.csv"
            )


# ============================================================
# MATCH DETAILS TAB
# ============================================================

with tab_details:

    st.subheader(
        "Detailed Match Analysis"
    )

    if not all_upcoming_predictions:

        st.info(
            "No matches are available for detailed analysis."
        )

    else:

        match_labels = [
            (
                f"{p['competition']} — "
                f"{p['home_team']} vs "
                f"{p['away_team']}"
            )
            for p in all_upcoming_predictions
        ]

        selected_index = st.selectbox(
            "Select a match",
            range(
                len(match_labels)
            ),
            format_func=lambda i: match_labels[i],
        )

        prediction = (
            all_upcoming_predictions[
                selected_index
            ]
        )

        st.markdown(
            f"## {prediction['home_team']} "
            f"vs "
            f"{prediction['away_team']}"
        )

        st.caption(
            f"{prediction['competition']} • "
            f"{prediction['date'].strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # ----------------------------------------------------
        # Expected goals
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Home xG",
                f"{prediction['home_xg']:.2f}",
            )

        with c2:
            st.metric(
                "Away xG",
                f"{prediction['away_xg']:.2f}",
            )

        with c3:
            st.metric(
                "Total xG",
                f"{prediction['home_xg'] + prediction['away_xg']:.2f}",
            )

        # ----------------------------------------------------
        # Main model selection
        # ----------------------------------------------------

        st.markdown(
            "### Model selection"
        )

        st.success(
            f"""
            **{prediction['best_market']}**

            Estimated probability:
            **{prediction['best_probability']:.1%}**

            Confidence category:
            **{prediction['confidence']}**
            """
        )

        st.caption(
            "This is the highest-probability selection produced by the configured statistical model. It is not a guaranteed outcome."
        )

        # ----------------------------------------------------
        # 1X2
        # ----------------------------------------------------

        st.markdown(
            "### 1X2 probabilities"
        )

        one_x_two = pd.DataFrame(
            {
                "Outcome": [
                    "Home Win",
                    "Draw",
                    "Away Win",
                ],
                "Probability": [
                    prediction["markets"][
                        "Home Win"
                    ],
                    prediction["markets"][
                        "Draw"
                    ],
                    prediction["markets"][
                        "Away Win"
                    ],
                ],
            }
        )

        one_x_two["Probability"] = (
            one_x_two["Probability"]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

        st.dataframe(
            one_x_two,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Scoreline heatmap
        # ----------------------------------------------------

        st.markdown(
            "### Correct-score probability matrix"
        )

        matrix = prediction[
            "matrix"
        ]

        score_labels = [
            str(i)
            for i in range(
                max_goals + 1
            )
        ]

        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=score_labels,
                y=score_labels,
                text=[
                    [
                        f"{value:.1%}"
                        for value in row
                    ]
                    for row in matrix
                ],
                texttemplate="%{text}",
                hovertemplate=(
                    "Home goals: %{y}"
                    "<br>Away goals: %{x}"
                    "<br>Probability: %{z:.2%}"
                    "<extra></extra>"
                ),
            )
        )

        heatmap.update_layout(
            xaxis_title="Away goals",
            yaxis_title="Home goals",
            title="Poisson scoreline matrix",
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Top correct scores
        # ----------------------------------------------------

        st.markdown(
            "### Most probable correct scores"
        )

        score_df = pd.DataFrame(
            [
                {
                    "Correct Score": x[
                        "score"
                    ],
                    "Probability": (
                        f"{x['probability']:.2%}"
                    ),
                }
                for x in prediction[
                    "top_scores"
                ]
            ]
        )

        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # All markets
        # ----------------------------------------------------

        st.markdown(
            "### Full market probabilities"
        )

        market_rows = []

        for market, probability in prediction[
            "markets"
        ].items():

            market_rows.append(
                {
                    "Market": market,
                    "Probability": probability,
                    "Probability Display": (
                        f"{probability:.2%}"
                    ),
                }
            )

        market_df = pd.DataFrame(
            market_rows
        )

        market_df = market_df.sort_values(
            "Probability",
            ascending=False,
        )

        st.dataframe(
            market_df[
                [
                    "Market",
                    "Probability Display",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Team form
        # ----------------------------------------------------

        st.markdown(
            "### Recent team form"
        )

        form1, form2 = st.columns(2)

        with form1:

            st.markdown(
                f"#### {prediction['home_team']}"
            )

            home_history_df = pd.DataFrame(
                prediction[
                    "home_history"
                ]
            )

            if not home_history_df.empty:

                display_cols = [
                    "date",
                    "home_team",
                    "away_team",
                    "goals_for",
                    "goals_against",
                    "venue",
                    "result",
                ]

                st.dataframe(
                    home_history_df[
                        display_cols
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        with form2:

            st.markdown(
                f"#### {prediction['away_team']}"
            )

            away_history_df = pd.DataFrame(
                prediction[
                    "away_history"
                ]
            )

            if not away_history_df.empty:

                display_cols = [
                    "date",
                    "home_team",
                    "away_team",
                    "goals_for",
                    "goals_against",
                    "venue",
                    "result",
                ]

                st.dataframe(
                    away_history_df[
                        display_cols
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# PREDICTION HISTORY TAB
# ============================================================

with tab_history:

    st.subheader(
        "Prediction History"
    )

    history_df = load_prediction_history()

    if history_df.empty:

        st.info(
            "No saved predictions yet. Use the Save Current Predictions button on the Predictions tab."
        )

    else:

        if st.button(
            "🔄 Update completed results"
        ):

            with st.spinner(
                "Checking completed matches..."
            ):

                history_df = (
                    update_prediction_results(
                        history_df,
                        api_key,
                    )
                )

            st.success(
                "Prediction results updated."
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        resolved = history_df[
            history_df[
                "result_correct"
            ].isin(
                ["WIN", "LOSS"]
            )
        ]

        if not resolved.empty:

            wins = (
                resolved[
                    "result_correct"
                ]
                == "WIN"
            ).sum()

            losses = (
                resolved[
                    "result_correct"
                ]
                == "LOSS"
            ).sum()

            total = wins + losses

            accuracy = (
                wins / total
                if total
                else 0
            )

            h1, h2, h3 = st.columns(3)

            with h1:
                st.metric(
                    "Resolved predictions",
                    total,
                )

            with h2:
                st.metric(
                    "Correct",
                    wins,
                )

            with h3:
                st.metric(
                    "Historical accuracy",
                    f"{accuracy:.1%}",
                )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download prediction history CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
        )


# ============================================================
# BACKTESTING
# ============================================================

with tab_backtest:

    st.subheader(
        "Historical Backtesting"
    )

    st.write(
        """
        Backtesting runs the same prediction logic against historical
        matches. For each historical fixture, only matches that occurred
        before that fixture are used to build the prediction. This helps
        reduce future-data leakage.
        """
    )

    st.warning(
        """
        Backtesting is computationally heavier and can consume more
        Football-Data.org API/data resources. Start with a smaller number
        of matches when testing.
        """
    )

    if st.button(
        "🧪 Run full backtest",
        type="primary",
    ):

        all_backtest_rows = []

        progress = st.progress(
            0
        )

        total_leagues = len(
            selected_leagues
        )

        for league_number, league_name in enumerate(
            selected_leagues,
            start=1,
        ):

            competition_code = COMPETITIONS[
                league_name
            ]

            try:

                competition_info = (
                    get_competition_info(
                        competition_code,
                        api_key,
                    )
                )

                current_season = (
                    competition_info
                    .get("currentSeason", {})
                    .get("startDate")
                )

                if current_season:

                    season_year = int(
                        str(
                            current_season
                        )[:4]
                    )

                else:

                    season_year = None

                all_matches = (
                    get_competition_matches(
                        competition_code,
                        api_key,
                        season=season_year,
                    )
                )

                historical_matches = [
                    m
                    for m in all_matches
                    if is_finished(m)
                ]

                historical_matches = sorted(
                    historical_matches,
                    key=lambda m: (
                        parse_match_date(m)
                    ),
                )

                # ------------------------------------------------
                # Use the most recent N matches.
                # ------------------------------------------------

                historical_matches = (
                    historical_matches[
                        -run_backtest_matches:
                    ]
                )

                for match in historical_matches:

                    try:

                        prediction = predict_match(
                            match,
                            all_matches,
                            history_matches=history_matches,
                            max_goals=max_goals,
                            minimum_probability=minimum_probability,
                        )

                        score = get_full_time_score(
                            match
                        )

                        if score is None:
                            continue

                        actual_home, actual_away = (
                            score
                        )

                        # ------------------------------------------------
                        # 1X2 actual outcome
                        # ------------------------------------------------

                        if actual_home > actual_away:
                            actual_1x2 = (
                                "Home Win"
                            )

                        elif actual_home == actual_away:
                            actual_1x2 = (
                                "Draw"
                            )

                        else:
                            actual_1x2 = (
                                "Away Win"
                            )

                        predicted_probs = {
                            "Home Win": prediction[
                                "markets"
                            ]["Home Win"],

                            "Draw": prediction[
                                "markets"
                            ]["Draw"],

                            "Away Win": prediction[
                                "markets"
                            ]["Away Win"],
                        }

                        predicted_1x2 = max(
                            predicted_probs,
                            key=predicted_probs.get,
                        )

                        all_backtest_rows.append(
                            {
                                "League": league_name,
                                "Date": parse_match_date(
                                    match
                                ),
                                "Home": prediction[
                                    "home_team"
                                ],
                                "Away": prediction[
                                    "away_team"
                                ],
                                "Actual Score": (
                                    f"{actual_home}-{actual_away}"
                                ),
                                "Predicted 1X2": predicted_1x2,
                                "Actual 1X2": actual_1x2,
                                "1X2 Correct": (
                                    predicted_1x2
                                    == actual_1x2
                                ),
                                "Home Win Prob": predicted_probs[
                                    "Home Win"
                                ],
                                "Draw Prob": predicted_probs[
                                    "Draw"
                                ],
                                "Away Win Prob": predicted_probs[
                                    "Away Win"
                                ],
                                "Best Market": prediction[
                                    "best_market"
                                ],
                                "Best Market Prob": prediction[
                                    "best_probability"
                                ],
                            }
                        )

                    except Exception as exc:

                        errors.append(
                            f"Backtest {league_name}: {exc}"
                        )

            except Exception as exc:

                st.error(
                    f"Could not backtest {league_name}: {exc}"
                )

            progress.progress(
                league_number
                / total_leagues
            )

        progress.empty()

        if all_backtest_rows:

            backtest_df = pd.DataFrame(
                all_backtest_rows
            )

            # ----------------------------------------------------
            # Accuracy
            # ----------------------------------------------------

            accuracy = (
                backtest_df[
                    "1X2 Correct"
                ].mean()
            )

            # ----------------------------------------------------
            # Multi-class Brier score
            # ----------------------------------------------------

            brier_values = []

            log_loss_values = []

            for _, row in backtest_df.iterrows():

                outcome = row[
                    "Actual 1X2"
                ]

                probs = [
                    row[
                        "Home Win Prob"
                    ],
                    row[
                        "Draw Prob"
                    ],
                    row[
                        "Away Win Prob"
                    ],
                ]

                classes = [
                    "Home Win",
                    "Draw",
                    "Away Win",
                ]

                one_hot = [
                    1 if outcome == c
                    else 0
                    for c in classes
                ]

                brier = np.mean(
                    [
                        (
                            probs[i]
                            - one_hot[i]
                        ) ** 2
                        for i in range(3)
                    ]
                )

                brier_values.append(
                    brier
                )

                actual_index = (
                    classes.index(
                        outcome
                    )
                )

                probability = np.clip(
                    probs[actual_index],
                    1e-15,
                    1 - 1e-15,
                )

                log_loss_values.append(
                    -math.log(
                        probability
                    )
                )

            brier = float(
                np.mean(
                    brier_values
                )
            )

            multiclass_log_loss = float(
                np.mean(
                    log_loss_values
                )
            )

            # ----------------------------------------------------
            # Metrics
            # ----------------------------------------------------

            b1, b2, b3 = st.columns(3)

            with b1:
                st.metric(
                    "1X2 Accuracy",
                    f"{accuracy:.1%}",
                )

            with b2:
                st.metric(
                    "Brier Score",
                    f"{brier:.4f}",
                )

            with b3:
                st.metric(
                    "Log Loss",
                    f"{multiclass_log_loss:.4f}",
                )

            st.caption(
                "For Brier score and log loss, lower values indicate better probabilistic calibration."
            )

            # ----------------------------------------------------
            # League performance
            # ----------------------------------------------------

            league_results = (
                backtest_df
                .groupby("League")
                ["1X2 Correct"]
                .mean()
                .reset_index()
            )

            league_results[
                "Accuracy"
            ] = league_results[
                "1X2 Correct"
            ].map(
                lambda x: f"{x:.1%}"
            )

            st.markdown(
                "### Accuracy by league"
            )

            st.dataframe(
                league_results[
                    [
                        "League",
                        "Accuracy",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------------
            # Results chart
            # ----------------------------------------------------

            chart = px.bar(
                league_results,
                x="League",
                y="1X2 Correct",
                title="Historical 1X2 accuracy by league",
            )

            chart.update_yaxes(
                tickformat=".0%",
                range=[0, 1],
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

            # ----------------------------------------------------
            # Detailed backtest
            # ----------------------------------------------------

            st.markdown(
                "### Backtest results"
            )

            display_backtest = backtest_df.copy()

            for col in [
                "Home Win Prob",
                "Draw Prob",
                "Away Win Prob",
                "Best Market Prob",
            ]:

                display_backtest[
                    col
                ] = display_backtest[
                    col
                ].map(
                    lambda x: f"{x:.1%}"
                )

            st.dataframe(
                display_backtest,
                use_container_width=True,
                hide_index=True,
            )

            csv = backtest_df.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download backtest results",
                data=csv,
                file_name="football_backtest.csv",
                mime="text/csv",
            )

        else:

            st.warning(
                "No backtest results were generated."
            )


# ============================================================
# MODEL TAB
# ============================================================

with tab_model:

    st.subheader(
        "🧠 How the prediction model works"
    )

    st.markdown(
        """
        ### 1. Data collection

        The application retrieves football fixtures and historical
        league results from Football-Data.org.

        The three configured competitions are:

        - Premier League — `PL`
        - La Liga — `PD`
        - Serie A — `SA`

        ### 2. Historical form

        For each upcoming match, the application examines the recent
        completed matches of both teams.

        It considers:

        - goals scored
        - goals conceded
        - home performance
        - away performance
        - recent results

        More recent matches receive greater weight.

        ### 3. Expected goals

        The model combines the attacking performance of one team with
        the defensive performance of its opponent.

        It also applies league-level smoothing so that a small number
        of recent matches does not completely dominate the calculation.

        ### 4. Poisson distribution

        Expected goals are converted into probabilities for possible
        scorelines.

        For example:

        `0-0`

        `1-0`

        `1-1`

        `2-1`

        `2-2`

        etc.

        The probability of each scoreline is then used to calculate
        the different markets.

        ### 5. Market probabilities

        The scoreline matrix is aggregated into:

        - Home Win
        - Draw
        - Away Win
        - Double Chance
        - Draw No Bet
        - Over/Under goals
        - BTTS
        - Team goal totals
        - BTTS + Result
        - Correct Score

        ### 6. Backtesting

        Historical matches can be tested using the same prediction
        methodology.

        The application reports:

        - Accuracy
        - Brier score
        - Log loss

        Importantly, the historical prediction process only uses
        matches that occurred before the target fixture.

        ### 7. Important limitations

        This is a statistical model.

        It does not currently directly model every factor that can
        influence a football match, such as:

        - injuries
        - confirmed starting lineups
        - suspensions
        - managerial changes
        - tactical changes
        - weather
        - travel
        - motivation
        - bookmaker odds
        - market movement

        Therefore, a high model probability should never be interpreted
        as certainty.
        """
    )

    st.markdown(
        "### API configuration"
    )

    st.code(
        """
# Streamlit secrets

FOOTBALL_DATA_API_KEY = "YOUR_API_KEY_HERE"
        """.strip(),
        language="toml",
    )

    st.markdown(
        "### Required Python packages"
    )

    st.code(
        """
streamlit
requests
pandas
numpy
scipy
plotly
        """.strip(),
        language="text",
    )

    st.markdown(
        "### Data source"
    )

    st.write(
        "Football-Data.org API v4"
    )

    st.markdown(
        "### Responsible-use note"
    )

    st.warning(
        """
        Football outcomes are inherently uncertain. Statistical
        probabilities are not guarantees. If this application is used
        in connection with betting, use appropriate limits and treat
        predictions as informational estimates rather than certainty.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Football Prediction Platform • Streamlit • Football-Data.org API • Poisson Statistical Model"
)
