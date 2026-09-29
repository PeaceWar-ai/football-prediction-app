
import math
import os
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Prediction Centre",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "https://api.football-data.org/v4"

LEAGUES = {
    "Premier League": "PL",
    "Championship": "ELC",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
    "Eredivisie": "DED",
    "Primeira Liga": "PPD",
    "UEFA Champions League": "CL",
    "Brasileirão Série A": "BSA",
    "Euro Championship": "EC",
    "FIFA World Cup": "WC",
}

# Maximum goals considered in the Poisson score matrix.
MAX_GOALS = 10

# Minimum historical matches required before a team-specific
# estimate is considered reasonably established.
MIN_TEAM_MATCHES = 3

# Prediction history is kept in Streamlit session state.
HISTORY_KEY = "prediction_history"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e7eaf0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    .small-note {
        color: #6b7280;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API FUNCTIONS
# ============================================================

def get_api_token():
    """Read the API token from Streamlit secrets or environment."""

    try:
        token = st.secrets.get("FOOTBALL_DATA_TOKEN", "")
    except Exception:
        token = ""

    if not token:
        token = os.getenv("FOOTBALL_DATA_TOKEN", "")

    return token.strip() if token else ""


def api_get(endpoint, token, params=None):
    """Send a request to Football-Data.org and return JSON."""

    if not token:
        raise ValueError("API token is missing.")

    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"

    headers = {
        "X-Auth-Token": token,
        "Accept": "application/json",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=25,
    )

    if response.status_code == 401:
        raise ValueError("Invalid API token. Check your Football-Data.org token.")

    if response.status_code == 403:
        raise ValueError(
            "Your current API plan does not provide access to this competition "
            "or endpoint."
        )

    if response.status_code == 404:
        raise ValueError(
            "The requested competition or data endpoint was not found."
        )

    if response.status_code == 429:
        raise ValueError(
            "API rate limit reached. Please wait before refreshing."
        )

    if response.status_code >= 400:
        raise ValueError(
            f"API request failed with status {response.status_code}: "
            f"{response.text[:250]}"
        )

    return response.json()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_competition_matches(competition_code, token, date_from=None, date_to=None):
    """Fetch matches for a competition."""

    params = {}

    if date_from:
        params["dateFrom"] = date_from

    if date_to:
        params["dateTo"] = date_to

    data = api_get(
        f"competitions/{competition_code}/matches",
        token,
        params=params,
    )

    return data.get("matches", [])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_competition_standings(competition_code, token):
    """Fetch standings when supported by the competition."""

    data = api_get(
        f"competitions/{competition_code}/standings",
        token,
    )

    return data.get("standings", [])


# ============================================================
# DATA PROCESSING
# ============================================================

def match_to_record(match, league_name):
    """Convert an API match object into a consistent dictionary."""

    home = match.get("homeTeam") or {}
    away = match.get("awayTeam") or {}
    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}

    return {
        "match_id": match.get("id"),
        "league": league_name,
        "competition_code": match.get("competition", {}).get("code"),
        "utc_date": match.get("utcDate"),
        "date": match.get("utcDate", "")[:10],
        "status": match.get("status"),
        "matchday": match.get("matchday"),
        "home_team": home.get("name", "Unknown"),
        "away_team": away.get("name", "Unknown"),
        "home_score": full_time.get("home"),
        "away_score": full_time.get("away"),
        "winner": score.get("winner"),
    }


def records_to_dataframe(matches, league_name):
    """Convert a list of API matches to a DataFrame."""

    records = [
        match_to_record(match, league_name)
        for match in matches
    ]

    return pd.DataFrame(records)


def get_finished_matches(df):
    """Return matches with valid full-time scores."""

    if df.empty:
        return df.copy()

    finished = df[
        (df["status"] == "FINISHED")
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    if not finished.empty:
        finished["home_score"] = finished["home_score"].astype(int)
        finished["away_score"] = finished["away_score"].astype(int)

    return finished


def get_upcoming_matches(df):
    """Return scheduled matches."""

    if df.empty:
        return df.copy()

    return df[
        df["status"].isin(["SCHEDULED", "TIMED"])
    ].copy()


# ============================================================
# TEAM STATISTICS
# ============================================================

def calculate_team_stats(matches, team_name, before_date=None):
    """
    Calculate a team's scoring and conceding rates.

    Only matches earlier than before_date are used when supplied.
    """

    if matches.empty:
        return None

    team_matches = matches[
        (matches["home_team"] == team_name)
        | (matches["away_team"] == team_name)
    ].copy()

    if before_date is not None:
        before_date = pd.to_datetime(before_date, utc=True)

        team_matches["_datetime"] = pd.to_datetime(
            team_matches["utc_date"],
            utc=True,
            errors="coerce",
        )

        team_matches = team_matches[
            team_matches["_datetime"] < before_date
        ]

    if team_matches.empty:
        return None

    goals_for = []
    goals_against = []
    home_games = 0
    away_games = 0
    wins = 0
    draws = 0
    losses = 0

    for _, match in team_matches.iterrows():
        if match["home_team"] == team_name:
            scored = int(match["home_score"])
            conceded = int(match["away_score"])
            home_games += 1
        else:
            scored = int(match["away_score"])
            conceded = int(match["home_score"])
            away_games += 1

        goals_for.append(scored)
        goals_against.append(conceded)

        if scored > conceded:
            wins += 1
        elif scored == conceded:
            draws += 1
        else:
            losses += 1

    games = len(goals_for)

    return {
        "team": team_name,
        "played": games,
        "home_games": home_games,
        "away_games": away_games,
        "goals_for": sum(goals_for),
        "goals_against": sum(goals_against),
        "goals_for_per_game": sum(goals_for) / games,
        "goals_against_per_game": sum(goals_against) / games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": wins * 3 + draws,
    }


def calculate_league_baseline(matches, before_date=None):
    """Calculate average home and away goals for a competition."""

    if matches.empty:
        return 1.35, 1.10

    data = matches.copy()

    if before_date is not None:
        before_date = pd.to_datetime(before_date, utc=True)

        data["_datetime"] = pd.to_datetime(
            data["utc_date"],
            utc=True,
            errors="coerce",
        )

        data = data[data["_datetime"] < before_date]

    if data.empty:
        return 1.35, 1.10

    home_average = data["home_score"].mean()
    away_average = data["away_score"].mean()

    if pd.isna(home_average) or home_average <= 0:
        home_average = 1.35

    if pd.isna(away_average) or away_average <= 0:
        away_average = 1.10

    return float(home_average), float(away_average)


# ============================================================
# POISSON PREDICTION ENGINE
# ============================================================

def poisson_probability(goals, expected_goals):
    """Calculate the probability of scoring exactly N goals."""

    expected_goals = max(0.01, float(expected_goals))

    return (
        math.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )


def calculate_expected_goals(
    home_stats,
    away_stats,
    league_home_average,
    league_away_average,
):
    """
    Estimate expected goals using attack and defence strengths.

    The calculation blends team-specific performance with league
    averages to reduce extreme estimates from small samples.
    """

    if home_stats is None or away_stats is None:
        return league_home_average, league_away_average

    # League averages are used as a stabilising baseline.
    home_attack = (
        home_stats["goals_for_per_game"] / max(0.1, league_home_average)
    )

    home_defence = (
        home_stats["goals_against_per_game"] / max(0.1, league_away_average)
    )

    away_attack = (
        away_stats["goals_for_per_game"] / max(0.1, league_away_average)
    )

    away_defence = (
        away_stats["goals_against_per_game"] / max(0.1, league_home_average)
    )

    # Keep estimates within a practical range.
    expected_home = league_home_average * home_attack * away_defence
    expected_away = league_away_average * away_attack * home_defence

    expected_home = max(0.15, min(expected_home, 4.5))
    expected_away = max(0.15, min(expected_away, 4.5))

    return expected_home, expected_away


def create_score_matrix(expected_home, expected_away, max_goals=MAX_GOALS):
    """Create a matrix of exact score probabilities."""

    home_probs = [
        poisson_probability(i, expected_home)
        for i in range(max_goals + 1)
    ]

    away_probs = [
        poisson_probability(i, expected_away)
        for i in range(max_goals + 1)
    ]

    matrix = []

    for home_goals in range(max_goals + 1):
        row = []

        for away_goals in range(max_goals + 1):
            row.append(
                home_probs[home_goals] * away_probs[away_goals]
            )

        matrix.append(row)

    # Normalise to account for probability beyond the matrix boundary.
    total = sum(sum(row) for row in matrix)

    if total > 0:
        matrix = [
            [probability / total for probability in row]
            for row in matrix
        ]

    return matrix


def analyse_score_matrix(matrix):
    """Calculate 1X2, goals, BTTS and most likely score markets."""

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over_15 = 0.0
    over_25 = 0.0
    over_35 = 0.0
    under_25 = 0.0
    btts_yes = 0.0
    btts_no = 0.0

    best_score = (0, 0)
    best_probability = 0.0

    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):

            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

            total_goals = home_goals + away_goals

            if total_goals >= 2:
                over_15 += probability

            if total_goals >= 3:
                over_25 += probability
            else:
                under_25 += probability

            if total_goals >= 4:
                over_35 += probability

            if home_goals > 0 and away_goals > 0:
                btts_yes += probability
            else:
                btts_no += probability

            if probability > best_probability:
                best_probability = probability
                best_score = (home_goals, away_goals)

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_15": over_15,
        "over_25": over_25,
        "over_35": over_35,
        "under_25": under_25,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "most_likely_score": best_score,
        "score_probability": best_probability,
    }


def predict_match(match, historical_matches):
    """Generate a prediction for one fixture."""

    home_team = match["home_team"]
    away_team = match["away_team"]
    kickoff = match.get("utc_date")

    home_stats = calculate_team_stats(
        historical_matches,
        home_team,
        before_date=kickoff,
    )

    away_stats = calculate_team_stats(
        historical_matches,
        away_team,
        before_date=kickoff,
    )

    league_home, league_away = calculate_league_baseline(
        historical_matches,
        before_date=kickoff,
    )

    expected_home, expected_away = calculate_expected_goals(
        home_stats,
        away_stats,
        league_home,
        league_away,
    )

    matrix = create_score_matrix(
        expected_home,
        expected_away,
    )

    markets = analyse_score_matrix(matrix)

    return {
        "match_id": match.get("match_id"),
        "league": match["league"],
        "date": match["date"],
        "utc_date": match["utc_date"],
        "home_team": home_team,
        "away_team": away_team,
        "home_expected_goals": expected_home,
        "away_expected_goals": expected_away,
        "home_history": home_stats["played"] if home_stats else 0,
        "away_history": away_stats["played"] if away_stats else 0,
        **markets,
    }


# ============================================================
# PREDICTION FORMATTING
# ============================================================

def percent(value):
    return f"{value * 100:.1f}%"


def prediction_to_row(prediction):
    """Convert prediction results to a display-friendly row."""

    home_score, away_score = prediction["most_likely_score"]

    return {
        "Date": prediction["date"],
        "League": prediction["league"],
        "Home Team": prediction["home_team"],
        "Away Team": prediction["away_team"],
        "Home Win": percent(prediction["home_win"]),
        "Draw": percent(prediction["draw"]),
        "Away Win": percent(prediction["away_win"]),
        "Over 2.5": percent(prediction["over_25"]),
        "Under 2.5": percent(prediction["under_25"]),
        "BTTS Yes": percent(prediction["btts_yes"]),
        "BTTS No": percent(prediction["btts_no"]),
        "Expected Goals": (
            f"{prediction['home_expected_goals']:.2f} - "
            f"{prediction['away_expected_goals']:.2f}"
        ),
        "Predicted Score": f"{home_score} - {away_score}",
        "Home History": prediction["home_history"],
        "Away History": prediction["away_history"],
    }


# ============================================================
# HISTORY AND RESULT TRACKING
# ============================================================

def initialise_history():
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []


def save_predictions(predictions):
    """Save predictions without duplicating the same fixture."""

    initialise_history()

    existing_ids = {
        item.get("match_id")
        for item in st.session_state[HISTORY_KEY]
    }

    for prediction in predictions:
        match_id = prediction.get("match_id")

        if match_id not in existing_ids:
            saved = prediction.copy()
            saved["saved_at"] = datetime.now().isoformat()
            saved["actual_home_score"] = None
            saved["actual_away_score"] = None
            saved["result_checked"] = False

            st.session_state[HISTORY_KEY].append(saved)
            existing_ids.add(match_id)


def update_saved_results(all_matches):
    """Update stored predictions when matches are finished."""

    initialise_history()

    score_lookup = {}

    for _, match in all_matches.iterrows():
        if (
            match["status"] == "FINISHED"
            and pd.notna(match["home_score"])
            and pd.notna(match["away_score"])
        ):
            score_lookup[match["match_id"]] = (
                int(match["home_score"]),
                int(match["away_score"]),
            )

    for prediction in st.session_state[HISTORY_KEY]:
        match_id = prediction.get("match_id")

        if match_id in score_lookup:
            home_score, away_score = score_lookup[match_id]

            prediction["actual_home_score"] = home_score
            prediction["actual_away_score"] = away_score
            prediction["result_checked"] = True


def get_prediction_history():
    initialise_history()
    return st.session_state[HISTORY_KEY]


def calculate_history_accuracy(history):
    """Calculate accuracy for settled 1X2 predictions."""

    settled = [
        item for item in history
        if item.get("result_checked")
    ]

    if not settled:
        return None

    correct = 0

    for item in settled:
        actual_home = item["actual_home_score"]
        actual_away = item["actual_away_score"]

        if actual_home > actual_away:
            actual_result = "home"
        elif actual_home == actual_away:
            actual_result = "draw"
        else:
            actual_result = "away"

        predicted_result = max(
            [
                ("home", item["home_win"]),
                ("draw", item["draw"]),
                ("away", item["away_win"]),
            ],
            key=lambda x: x[1],
        )[0]

        if predicted_result == actual_result:
            correct += 1

    return {
        "settled": len(settled),
        "correct": correct,
        "accuracy": correct / len(settled),
    }


# ============================================================
# BACKTESTING
# ============================================================

def backtest_matches(matches, minimum_history=5):
    """
    Backtest historical matches in chronological order.

    Each prediction uses only matches played before the target match.
    """

    if matches.empty:
        return pd.DataFrame()

    data = matches.copy()

    data["_datetime"] = pd.to_datetime(
        data["utc_date"],
        utc=True,
        errors="coerce",
    )

    data = data.sort_values("_datetime")

    results = []

    for index, match in data.iterrows():
        kickoff = match["utc_date"]

        prior_matches = data[
            data["_datetime"] < pd.to_datetime(kickoff, utc=True)
        ].copy()

        home_history = calculate_team_stats(
            prior_matches,
            match["home_team"],
        )

        away_history = calculate_team_stats(
            prior_matches,
            match["away_team"],
        )

        if (
            home_history is None
            or away_history is None
            or home_history["played"] < minimum_history
            or away_history["played"] < minimum_history
        ):
            continue

        record = match.to_dict()

        prediction = predict_match(
            record,
            prior_matches,
        )

        actual_home = int(match["home_score"])
        actual_away = int(match["away_score"])

        if actual_home > actual_away:
            actual_result = "home"
        elif actual_home == actual_away:
            actual_result = "draw"
        else:
            actual_result = "away"

        predicted_result = max(
            [
                ("home", prediction["home_win"]),
                ("draw", prediction["draw"]),
                ("away", prediction["away_win"]),
            ],
            key=lambda x: x[1],
        )[0]

        results.append({
            "Date": match["date"],
            "League": match["league"],
            "Home Team": match["home_team"],
            "Away Team": match["away_team"],
            "Actual Score": f"{actual_home} - {actual_away}",
            "Predicted Score": (
                f"{prediction['most_likely_score'][0]} - "
                f"{prediction['most_likely_score'][1]}"
            ),
            "Actual Result": actual_result,
            "Predicted Result": predicted_result,
            "Correct": actual_result == predicted_result,
            "Home Probability": prediction["home_win"],
            "Draw Probability": prediction["draw"],
            "Away Probability": prediction["away_win"],
        })

    return pd.DataFrame(results)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚽ Football Prediction Centre")
st.sidebar.caption("Data-powered match analysis")

token = get_api_token()

if not token:
    token = st.sidebar.text_input(
        "Football-Data.org API Token",
        type="password",
        help="Your token is used to retrieve football data.",
    )

st.sidebar.markdown("---")

selected_leagues = st.sidebar.multiselect(
    "Leagues",
    list(LEAGUES.keys()),
    default=list(LEAGUES.keys()),
    help="Select the competitions you want to include in your predictions.",
)

st.sidebar.markdown("---")

today = date.today()

date_from = st.sidebar.date_input(
    "Start date",
    value=today,
)

date_to = st.sidebar.date_input(
    "End date",
    value=today + timedelta(days=14),
)

if date_to < date_from:
    st.sidebar.error("End date must be after the start date.")

st.sidebar.markdown("---")

minimum_history = st.sidebar.slider(
    "Minimum historical matches for backtesting",
    min_value=1,
    max_value=15,
    value=5,
)

refresh = st.sidebar.button(
    "🔄 Refresh data",
    use_container_width=True,
)

if refresh:
    st.cache_data.clear()
    st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title("⚽ Football Prediction Centre")
st.markdown(
    """
    Analyse upcoming football fixtures using historical team
    performance and a Poisson probability model.
    """
)

st.info(
    "Predictions are statistical estimates, not guarantees. "
    "Football outcomes are uncertain, and this tool should not be "
    "treated as financial or betting advice."
)


# ============================================================
# VALIDATION
# ============================================================

if not token:
    st.warning(
        "Enter your Football-Data.org API token in the sidebar "
        "or configure it in Streamlit secrets to begin."
    )
    st.stop()

if not selected_leagues:
    st.warning("Select at least one competition.")
    st.stop()

if date_to < date_from:
    st.error("Please correct the selected date range.")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

all_league_data = {}
all_match_frames = []
league_errors = {}

with st.spinner("Loading football data..."):

    for league_name in selected_leagues:
        competition_code = LEAGUES[league_name]

        try:
            matches = fetch_competition_matches(
                competition_code,
                token,
                date_from=date_from.isoformat(),
                date_to=date_to.isoformat(),
            )

            # Fetch a wider range for historical team form.
            historical_matches = fetch_competition_matches(
                competition_code,
                token,
            )

            current_df = records_to_dataframe(
                matches,
                league_name,
            )

            history_df = records_to_dataframe(
                historical_matches,
                league_name,
            )

            all_league_data[league_name] = {
                "current": current_df,
                "history": history_df,
            }

            if not history_df.empty:
                all_match_frames.append(history_df)

        except Exception as error:
            league_errors[league_name] = str(error)


if league_errors:
    with st.expander(
        f"Data access notices ({len(league_errors)})",
        expanded=True,
    ):
        for league_name, error in league_errors.items():
            st.warning(f"**{league_name}:** {error}")


if not all_match_frames:
    st.error(
        "No competition data could be loaded. Check your API token, "
        "subscription access, and internet connection."
    )
    st.stop()


all_matches = pd.concat(
    all_match_frames,
    ignore_index=True,
)

all_finished = get_finished_matches(all_matches)


# ============================================================
# UPDATE HISTORY
# ============================================================

initialise_history()
update_saved_results(all_matches)


# ============================================================
# BUILD UPCOMING PREDICTIONS
# ============================================================

upcoming_frames = []

for league_name, league_data in all_league_data.items():
    current_df = league_data["current"]

    if not current_df.empty:
        upcoming = get_upcoming_matches(current_df)

        if not upcoming.empty:
            upcoming_frames.append(upcoming)


if upcoming_frames:
    upcoming_matches = pd.concat(
        upcoming_frames,
        ignore_index=True,
    )
else:
    upcoming_matches = pd.DataFrame()


predictions = []

if not upcoming_matches.empty:
    with st.spinner("Generating match predictions..."):

        for _, match in upcoming_matches.iterrows():
            league_name = match["league"]

            league_history = all_finished[
                all_finished["league"] == league_name
            ].copy()

            try:
                prediction = predict_match(
                    match.to_dict(),
                    league_history,
                )

                predictions.append(prediction)

            except Exception as error:
                st.warning(
                    f"Could not predict "
                    f"{match['home_team']} vs {match['away_team']}: {error}"
                )


# ============================================================
# NAVIGATION
# ============================================================

tab_dashboard, tab_fixtures, tab_history, tab_backtest, tab_standings = (
    st.tabs([
        "📊 Dashboard",
        "⚽ Fixtures & Predictions",
        "🗂️ Prediction History",
        "🧪 Backtesting",
        "🏆 Standings",
    ])
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    st.subheader("Overview")

    total_fixtures = len(upcoming_matches)
    total_predictions = len(predictions)
    total_leagues = len(all_league_data)

    history = get_prediction_history()
    accuracy = calculate_history_accuracy(history)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Selected Leagues", total_leagues)
    col2.metric("Upcoming Fixtures", total_fixtures)
    col3.metric("Predictions Generated", total_predictions)

    if accuracy:
        col4.metric(
            "Settled 1X2 Accuracy",
            percent(accuracy["accuracy"]),
        )
    else:
        col4.metric("Settled 1X2 Accuracy", "N/A")

    st.markdown("---")

    st.subheader("Competition Data Status")

    status_rows = []

    for league_name, league_data in all_league_data.items():
        history_df = league_data["history"]
        current_df = league_data["current"]

        status_rows.append({
            "Competition": league_name,
            "Historical Matches": len(get_finished_matches(history_df)),
            "Fixtures in Selected Window": len(current_df),
            "Access": "Available",
        })

    for league_name, error in league_errors.items():
        status_rows.append({
            "Competition": league_name,
            "Historical Matches": 0,
            "Fixtures in Selected Window": 0,
            "Access": "Unavailable",
        })

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )

    if predictions:
        st.markdown("---")
        st.subheader("Upcoming Fixtures")

        overview_df = pd.DataFrame(
            [prediction_to_row(p) for p in predictions]
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No upcoming fixtures were found in the selected date range."
        )


# ============================================================
# FIXTURES AND PREDICTIONS
# ============================================================

with tab_fixtures:

    st.subheader("Fixtures & Predictions")

    if not predictions:
        st.info(
            "No predictions are available for the selected date range."
        )
    else:
        league_filter = st.selectbox(
            "Filter by competition",
            ["All Competitions"] + selected_leagues,
        )

        filtered_predictions = predictions

        if league_filter != "All Competitions":
            filtered_predictions = [
                p for p in predictions
                if p["league"] == league_filter
            ]

        prediction_df = pd.DataFrame(
            [prediction_to_row(p) for p in filtered_predictions]
        )

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download predictions as CSV",
            data=prediction_df.to_csv(index=False).encode("utf-8"),
            file_name="football_predictions.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Detailed Match Analysis")

        match_options = {
            (
                f"{p['date']} | {p['home_team']} vs {p['away_team']} "
                f"({p['league']})"
            ): p
            for p in filtered_predictions
        }

        selected_match_label = st.selectbox(
            "Select a match",
            list(match_options.keys()),
        )

        selected_prediction = match_options[selected_match_label]

        home = selected_prediction["home_team"]
        away = selected_prediction["away_team"]

        st.markdown(f"### {home} vs {away}")

        home_col, draw_col, away_col = st.columns(3)

        home_col.metric(
            f"{home} Win",
            percent(selected_prediction["home_win"]),
        )

        draw_col.metric(
            "Draw",
            percent(selected_prediction["draw"]),
        )

        away_col.metric(
            f"{away} Win",
            percent(selected_prediction["away_win"]),
        )

        st.markdown("---")

        goal_col1, goal_col2 = st.columns(2)

        goal_col1.metric(
            "Expected Home Goals",
            f"{selected_prediction['home_expected_goals']:.2f}",
        )

        goal_col2.metric(
            "Expected Away Goals",
            f"{selected_prediction['away_expected_goals']:.2f}",
        )

        st.markdown(
            f"**Most likely exact score:** "
            f"{selected_prediction['most_likely_score'][0]} - "
            f"{selected_prediction['most_likely_score'][1]}"
        )

        st.markdown("---")
        st.subheader("Goal Markets")

        market_data = pd.DataFrame({
            "Market": [
                "Over 1.5 Goals",
                "Over 2.5 Goals",
                "Under 2.5 Goals",
                "Over 3.5 Goals",
                "Both Teams to Score — Yes",
                "Both Teams to Score — No",
            ],
            "Estimated Probability": [
                selected_prediction["over_15"],
                selected_prediction["over_25"],
                selected_prediction["under_25"],
                selected_prediction["over_35"],
                selected_prediction["btts_yes"],
                selected_prediction["btts_no"],
            ],
        })

        market_data["Estimated Probability"] = (
            market_data["Estimated Probability"].apply(percent)
        )

        st.dataframe(
            market_data,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The model uses league scoring averages and historical team "
            "performance. It does not account for every factor, such as "
            "injuries, suspensions, lineups, tactical changes, or weather."
        )

        if st.button(
            "💾 Save these predictions to history",
            use_container_width=True,
        ):
            save_predictions(filtered_predictions)
            st.success("Predictions saved to your session history.")


# ============================================================
# PREDICTION HISTORY
# ============================================================

with tab_history:

    st.subheader("Prediction History")

    history = get_prediction_history()

    if not history:
        st.info(
            "No predictions have been saved yet. "
            "Use the Fixtures & Predictions tab to save predictions."
        )
    else:
        history_rows = []

        for item in history:
            home_score = item.get("actual_home_score")
            away_score = item.get("actual_away_score")

            if item.get("result_checked"):
                actual_score = f"{home_score} - {away_score}"
            else:
                actual_score = "Pending"

            history_rows.append({
                "Date": item["date"],
                "League": item["league"],
                "Home Team": item["home_team"],
                "Away Team": item["away_team"],
                "Predicted Score": (
                    f"{item['most_likely_score'][0]} - "
                    f"{item['most_likely_score'][1]}"
                ),
                "Actual Score": actual_score,
                "Home Win": percent(item["home_win"]),
                "Draw": percent(item["draw"]),
                "Away Win": percent(item["away_win"]),
            })

        history_df = pd.DataFrame(history_rows)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        accuracy = calculate_history_accuracy(history)

        if accuracy:
            st.markdown("---")
            st.subheader("Settled Prediction Performance")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric("Settled Matches", accuracy["settled"])
            metric2.metric("Correct 1X2 Predictions", accuracy["correct"])
            metric3.metric(
                "1X2 Accuracy",
                percent(accuracy["accuracy"]),
            )

        if st.button("Clear session prediction history"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()

        st.caption(
            "History is stored in the current Streamlit session. "
            "It may be lost when the session resets or the app restarts."
        )


# ============================================================
# BACKTESTING
# ============================================================

with tab_backtest:

    st.subheader("Historical Backtesting")

    st.write(
        "Backtesting evaluates how the model would have performed on "
        "past matches. Each historical prediction is generated using "
        "only matches played before the target fixture."
    )

    backtest_league = st.selectbox(
        "Competition to backtest",
        selected_leagues,
        key="backtest_league",
    )

    backtest_history = all_league_data[backtest_league]["history"]
    backtest_finished = get_finished_matches(backtest_history)

    if backtest_finished.empty:
        st.info("No finished matches are available for this competition.")
    else:
        st.caption(
            f"{len(backtest_finished)} finished matches are available "
            "in the retrieved dataset."
        )

        if st.button(
            "Run backtest",
            use_container_width=True,
        ):
            with st.spinner(
                "Running historical simulations. This may take a while..."
            ):
                backtest_results = backtest_matches(
                    backtest_finished,
                    minimum_history=minimum_history,
                )

            st.session_state["backtest_results"] = backtest_results

        if "backtest_results" in st.session_state:
            results = st.session_state["backtest_results"]

            if results.empty:
                st.warning(
                    "Not enough historical matches to run the backtest. "
                    "Try reducing the minimum historical matches setting."
                )
            else:
                total = len(results)
                correct = int(results["Correct"].sum())
                accuracy_value = correct / total

                b1, b2, b3 = st.columns(3)

                b1.metric("Matches Evaluated", total)
                b2.metric("Correct 1X2 Predictions", correct)
                b3.metric("Historical 1X2 Accuracy", percent(accuracy_value))

                st.markdown("---")

                st.dataframe(
                    results.drop(
                        columns=[
                            "Home Probability",
                            "Draw Probability",
                            "Away Probability",
                        ],
                        errors="ignore",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇️ Download backtest results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="football_backtest.csv",
                    mime="text/csv",
                )

                st.caption(
                    "Backtest results describe performance on the retrieved "
                    "historical sample. They do not guarantee future accuracy."
                )


# ============================================================
# STANDINGS
# ============================================================

with tab_standings:

    st.subheader("League Standings")

    standings_league = st.selectbox(
        "Select competition",
        selected_leagues,
        key="standings_league",
    )

    standings_code = LEAGUES[standings_league]

    try:
        standings = fetch_competition_standings(
            standings_code,
            token,
        )

        if not standings:
            st.info(
                "Standings are not available for this competition or "
                "are not supported by the current API response."
            )
        else:
            for table in standings:
                table_type = table.get("type", "TABLE")
                group = table.get("group")

                title = table_type

                if group:
                    title = f"{title} — {group}"

                st.markdown(f"### {title}")

                rows = []

                for entry in table.get("table", []):
                    team = entry.get("team", {})

                    rows.append({
                        "Position": entry.get("position"),
                        "Team": team.get("name"),
                        "Played": entry.get("playedGames"),
                        "Won": entry.get("won"),
                        "Drawn": entry.get("draw"),
                        "Lost": entry.get("lost"),
                        "Goals For": entry.get("goalsFor"),
                        "Goals Against": entry.get("goalsAgainst"),
                        "Goal Difference": entry.get("goalDifference"),
                        "Points": entry.get("points"),
                    })

                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    except Exception as error:
        st.warning(
            f"Standings could not be loaded for {standings_league}: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small-note">
        Football Prediction Centre · Powered by Streamlit and
        Football-Data.org API · Predictions are estimates only.
    </div>
    """,
    unsafe_allow_html=True,
)
