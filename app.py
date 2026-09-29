
import os
import math
from datetime import date, datetime

import pandas as pd
import requests
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Football Prediction Centre",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# COMPETITIONS
# =========================================================

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

API_URL = "https://api.football-data.org/v4"
HISTORY_KEY = "prediction_history"
MAX_GOALS = 7


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
    }

    .small-text {
        color: #777777;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# API TOKEN AND REQUESTS
# =========================================================

def get_token():
    try:
        token = st.secrets.get("FOOTBALL_DATA_TOKEN", "")
    except Exception:
        token = ""

    return token or os.getenv("FOOTBALL_DATA_TOKEN", "")


def api_get(endpoint, token, params=None):
    response = requests.get(
        f"{API_URL}/{endpoint}",
        headers={"X-Auth-Token": token},
        params=params,
        timeout=30,
    )

    if response.status_code == 401:
        raise ValueError("Invalid API token.")

    if response.status_code == 403:
        raise ValueError(
            "Your Football-Data.org plan does not provide access "
            "to this competition or endpoint."
        )

    if response.status_code == 429:
        raise ValueError("API rate limit reached. Try again later.")

    if response.status_code >= 400:
        raise ValueError(
            f"API error {response.status_code}: {response.text[:200]}"
        )

    return response.json()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_matches(competition_code, token):
    data = api_get(
        f"competitions/{competition_code}/matches",
        token,
    )
    return data.get("matches", [])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_standings(competition_code, token):
    data = api_get(
        f"competitions/{competition_code}/standings",
        token,
    )
    return data.get("standings", [])


# =========================================================
# DATA PROCESSING
# =========================================================

def matches_to_df(matches, league_name):
    rows = []

    for match in matches:
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}
        score = match.get("score", {}).get("fullTime") or {}

        rows.append({
            "match_id": match.get("id"),
            "league": league_name,
            "date": match.get("utcDate", "")[:10],
            "utc_date": match.get("utcDate"),
            "status": match.get("status"),
            "matchday": match.get("matchday"),
            "home_team": home.get("name", "Unknown"),
            "away_team": away.get("name", "Unknown"),
            "home_score": score.get("home"),
            "away_score": score.get("away"),
        })

    return pd.DataFrame(rows)


def finished_matches(df):
    if df.empty:
        return df.copy()

    result = df[
        (df["status"] == "FINISHED")
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    if not result.empty:
        result["home_score"] = result["home_score"].astype(int)
        result["away_score"] = result["away_score"].astype(int)

    return result


def upcoming_matches(df):
    if df.empty:
        return df.copy()

    return df[
        df["status"].isin(["SCHEDULED", "TIMED"])
    ].copy()


# =========================================================
# TEAM FORM
# =========================================================

def get_team_form(matches, team, recent_matches=10, before_date=None):
    if matches.empty:
        return None

    team_data = matches[
        (matches["home_team"] == team)
        | (matches["away_team"] == team)
    ].copy()

    if before_date:
        team_data["_datetime"] = pd.to_datetime(
            team_data["utc_date"],
            utc=True,
            errors="coerce",
        )
        cutoff = pd.to_datetime(before_date, utc=True)
        team_data = team_data[team_data["_datetime"] < cutoff]

    team_data = team_data.sort_values(
        "utc_date",
        ascending=False,
    ).head(recent_matches)

    if team_data.empty:
        return None

    goals_for = []
    goals_against = []
    wins = draws = losses = 0

    for _, match in team_data.iterrows():
        if match["home_team"] == team:
            scored = int(match["home_score"])
            conceded = int(match["away_score"])
        else:
            scored = int(match["away_score"])
            conceded = int(match["home_score"])

        goals_for.append(scored)
        goals_against.append(conceded)

        if scored > conceded:
            wins += 1
        elif scored == conceded:
            draws += 1
        else:
            losses += 1

    played = len(goals_for)

    return {
        "played": played,
        "goals_for": sum(goals_for),
        "goals_against": sum(goals_against),
        "scored_avg": sum(goals_for) / played,
        "conceded_avg": sum(goals_against) / played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": wins * 3 + draws,
        "form": f"{wins}W {draws}D {losses}L",
    }


def league_averages(matches, before_date=None):
    data = matches.copy()

    if before_date:
        data["_datetime"] = pd.to_datetime(
            data["utc_date"],
            utc=True,
            errors="coerce",
        )
        cutoff = pd.to_datetime(before_date, utc=True)
        data = data[data["_datetime"] < cutoff]

    if data.empty:
        return 1.35, 1.10

    home_avg = data["home_score"].mean()
    away_avg = data["away_score"].mean()

    if pd.isna(home_avg) or home_avg <= 0:
        home_avg = 1.35

    if pd.isna(away_avg) or away_avg <= 0:
        away_avg = 1.10

    return float(home_avg), float(away_avg)


# =========================================================
# POISSON MODEL
# =========================================================

def poisson_probability(goals, expected):
    expected = max(0.01, expected)
    return (
        math.exp(-expected)
        * expected ** goals
        / math.factorial(goals)
    )


def expected_goals(home_form, away_form, home_avg, away_avg):
    if not home_form or not away_form:
        return home_avg, away_avg

    home_attack = home_form["scored_avg"] / max(home_avg, 0.1)
    home_defence = home_form["conceded_avg"] / max(away_avg, 0.1)

    away_attack = away_form["scored_avg"] / max(away_avg, 0.1)
    away_defence = away_form["conceded_avg"] / max(home_avg, 0.1)

    home_xg = home_avg * home_attack * away_defence
    away_xg = away_avg * away_attack * home_defence

    return (
        max(0.15, min(home_xg, 5.0)),
        max(0.15, min(away_xg, 5.0)),
    )


def score_matrix(home_xg, away_xg, max_goals=7):
    home_probs = [
        poisson_probability(i, home_xg)
        for i in range(max_goals + 1)
    ]

    away_probs = [
        poisson_probability(i, away_xg)
        for i in range(max_goals + 1)
    ]

    matrix = [
        [
            home_probs[h] * away_probs[a]
            for a in range(max_goals + 1)
        ]
        for h in range(max_goals + 1)
    ]

    total = sum(sum(row) for row in matrix)

    if total:
        matrix = [
            [value / total for value in row]
            for row in matrix
        ]

    return matrix


def analyse_matrix(matrix):
    home_win = draw = away_win = 0.0
    over_15 = over_25 = over_35 = 0.0
    under_25 = 0.0
    btts_yes = btts_no = 0.0

    best_score = (0, 0)
    best_probability = 0

    score_probabilities = {}

    for h, row in enumerate(matrix):
        for a, probability in enumerate(row):
            score_probabilities[(h, a)] = probability

            if h > a:
                home_win += probability
            elif h == a:
                draw += probability
            else:
                away_win += probability

            total_goals = h + a

            if total_goals >= 2:
                over_15 += probability

            if total_goals >= 3:
                over_25 += probability
            else:
                under_25 += probability

            if total_goals >= 4:
                over_35 += probability

            if h > 0 and a > 0:
                btts_yes += probability
            else:
                btts_no += probability

            if probability > best_probability:
                best_probability = probability
                best_score = (h, a)

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_15": over_15,
        "over_25": over_25,
        "under_25": under_25,
        "over_35": over_35,
        "btts_yes": btts_yes,
        "btts_no": btts_no,
        "best_score": best_score,
        "best_score_probability": best_probability,
        "score_probabilities": score_probabilities,
    }


def predict_match(
    match,
    history,
    recent_matches=10,
    max_goals=7,
):
    kickoff = match.get("utc_date")

    home = match["home_team"]
    away = match["away_team"]

    home_form = get_team_form(
        history,
        home,
        recent_matches,
        before_date=kickoff,
    )

    away_form = get_team_form(
        history,
        away,
        recent_matches,
        before_date=kickoff,
    )

    home_avg, away_avg = league_averages(
        history,
        before_date=kickoff,
    )

    home_xg, away_xg = expected_goals(
        home_form,
        away_form,
        home_avg,
        away_avg,
    )

    matrix = score_matrix(
        home_xg,
        away_xg,
        max_goals,
    )

    markets = analyse_matrix(matrix)

    return {
        "match_id": match.get("match_id"),
        "league": match["league"],
        "date": match["date"],
        "utc_date": kickoff,
        "home_team": home,
        "away_team": away,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_form": home_form,
        "away_form": away_form,
        **markets,
    }


# =========================================================
# HELPERS
# =========================================================

def pct(value):
    return f"{value * 100:.1f}%"


def result_label(home_score, away_score):
    if home_score > away_score:
        return "home"
    if home_score == away_score:
        return "draw"
    return "away"


def predicted_result(prediction):
    outcomes = {
        "home": prediction["home_win"],
        "draw": prediction["draw"],
        "away": prediction["away_win"],
    }
    return max(outcomes, key=outcomes.get)


def prediction_row(p):
    h, a = p["best_score"]

    return {
        "Date": p["date"],
        "League": p["league"],
        "Home Team": p["home_team"],
        "Away Team": p["away_team"],
        "Home Win": pct(p["home_win"]),
        "Draw": pct(p["draw"]),
        "Away Win": pct(p["away_win"]),
        "Over 2.5": pct(p["over_25"]),
        "Under 2.5": pct(p["under_25"]),
        "BTTS Yes": pct(p["btts_yes"]),
        "BTTS No": pct(p["btts_no"]),
        "Expected Goals": f"{p['home_xg']:.2f} - {p['away_xg']:.2f}",
        "Predicted Score": f"{h} - {a}",
    }


# =========================================================
# SESSION HISTORY
# =========================================================

if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []


def save_predictions(predictions):
    history = st.session_state[HISTORY_KEY]

    existing = {
        item["match_id"]
        for item in history
    }

    for p in predictions:
        if p["match_id"] in existing:
            continue

        item = p.copy()
        item["saved_at"] = datetime.now().isoformat()
        item["actual_home"] = None
        item["actual_away"] = None
        item["settled"] = False

        history.append(item)
        existing.add(p["match_id"])


def update_history(all_matches):
    scores = {}

    for _, match in all_matches.iterrows():
        if (
            match["status"] == "FINISHED"
            and pd.notna(match["home_score"])
            and pd.notna(match["away_score"])
        ):
            scores[match["match_id"]] = (
                int(match["home_score"]),
                int(match["away_score"]),
            )

    for item in st.session_state[HISTORY_KEY]:
        if item["match_id"] in scores:
            h, a = scores[item["match_id"]]
            item["actual_home"] = h
            item["actual_away"] = a
            item["settled"] = True


def history_accuracy():
    settled = [
        p for p in st.session_state[HISTORY_KEY]
        if p["settled"]
    ]

    if not settled:
        return None

    correct = 0

    for p in settled:
        actual = result_label(
            p["actual_home"],
            p["actual_away"],
        )

        if predicted_result(p) == actual:
            correct += 1

    return correct, len(settled), correct / len(settled)


# =========================================================
# BACKTESTING
# =========================================================

def run_backtest(
    matches,
    recent_matches,
    max_goals,
    matches_limit,
    minimum_history=5,
):
    if matches.empty:
        return pd.DataFrame()

    data = matches.copy()

    data["_datetime"] = pd.to_datetime(
        data["utc_date"],
        utc=True,
        errors="coerce",
    )

    data = data.sort_values("_datetime").tail(matches_limit)

    results = []

    for _, match in data.iterrows():
        kickoff = pd.to_datetime(
            match["utc_date"],
            utc=True,
        )

        prior = matches.copy()
        prior["_datetime"] = pd.to_datetime(
            prior["utc_date"],
            utc=True,
            errors="coerce",
        )
        prior = prior[prior["_datetime"] < kickoff]

        home_form = get_team_form(
            prior,
            match["home_team"],
            recent_matches,
        )

        away_form = get_team_form(
            prior,
            match["away_team"],
            recent_matches,
        )

        if (
            not home_form
            or not away_form
            or home_form["played"] < minimum_history
            or away_form["played"] < minimum_history
        ):
            continue

        prediction = predict_match(
            match.to_dict(),
            prior,
            recent_matches,
            max_goals,
        )

        actual_home = int(match["home_score"])
        actual_away = int(match["away_score"])

        actual = result_label(actual_home, actual_away)
        predicted = predicted_result(prediction)

        results.append({
            "Date": match["date"],
            "Home Team": match["home_team"],
            "Away Team": match["away_team"],
            "Actual Score": f"{actual_home} - {actual_away}",
            "Predicted Score": (
                f"{prediction['best_score'][0]} - "
                f"{prediction['best_score'][1]}"
            ),
            "Actual Result": actual,
            "Predicted Result": predicted,
            "Correct": actual == predicted,
        })

    return pd.DataFrame(results)


# =========================================================
# SIDEBAR — ORIGINAL SETTINGS RESTORED
# =========================================================

st.sidebar.title("⚙️ Settings")

token = get_token()

if not token:
    token = st.sidebar.text_input(
        "Football-Data.org API Token",
        type="password",
    )

st.sidebar.markdown("---")

selected_leagues = st.sidebar.multiselect(
    "Leagues",
    options=list(LEAGUES.keys()),
    default=[
        "Premier League",
        "La Liga",
        "Serie A",
    ],
)

st.sidebar.markdown("---")

recent_matches = st.sidebar.slider(
    "Recent matches used",
    min_value=3,
    max_value=20,
    value=10,
)

probability_threshold = st.sidebar.slider(
    "Selection probability threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.60,
    step=0.05,
    format="%.2f",
)

max_goals = st.sidebar.slider(
    "Maximum scoreline goals",
    min_value=3,
    max_value=12,
    value=7,
)

backtest_limit = st.sidebar.slider(
    "Backtest matches per league",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
)

st.sidebar.markdown("---")

refresh = st.sidebar.button(
    "🔄 Refresh data",
    use_container_width=True,
)

if refresh:
    st.cache_data.clear()
    st.rerun()


# =========================================================
# MAIN APPLICATION
# =========================================================

st.title("⚽ Football Prediction Centre")

st.markdown(
    """
    A football analysis dashboard using historical match data,
    team form and Poisson probability modelling.
    """
)

st.info(
    "All probabilities are statistical estimates, not guaranteed "
    "outcomes. The model does not account for every factor affecting "
    "a match."
)

if not token:
    st.warning(
        "Please enter your Football-Data.org API token in the sidebar."
    )
    st.stop()

if not selected_leagues:
    st.warning("Select at least one league.")
    st.stop()


# =========================================================
# LOAD SELECTED COMPETITIONS
# =========================================================

league_data = {}
all_frames = []
errors = {}

with st.spinner("Loading selected competitions..."):

    for league in selected_leagues:
        code = LEAGUES[league]

        try:
            raw = fetch_matches(code, token)
            df = matches_to_df(raw, league)

            league_data[league] = df

            if not df.empty:
                all_frames.append(df)

        except Exception as error:
            errors[league] = str(error)


if errors:
    with st.expander("Competition access notices", expanded=True):
        for league, error in errors.items():
            st.warning(f"{league}: {error}")


if not all_frames:
    st.error(
        "No data was retrieved. Check your API token and competition access."
    )
    st.stop()


all_matches = pd.concat(
    all_frames,
    ignore_index=True,
)

finished = finished_matches(all_matches)
upcoming = upcoming_matches(all_matches)

update_history(all_matches)


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

predictions = []

with st.spinner("Calculating match probabilities..."):

    for _, match in upcoming.iterrows():
        league_history = finished[
            finished["league"] == match["league"]
        ].copy()

        try:
            prediction = predict_match(
                match.to_dict(),
                league_history,
                recent_matches,
                max_goals,
            )

            predictions.append(prediction)

        except Exception as error:
            st.warning(
                f"Prediction error for {match['home_team']} "
                f"vs {match['away_team']}: {error}"
            )


# =========================================================
# TABS
# =========================================================

dashboard, fixtures_tab, history_tab, backtest_tab, standings_tab = st.tabs(
    [
        "📊 Dashboard",
        "⚽ Fixtures & Predictions",
        "🗂️ Prediction History",
        "🧪 Backtesting",
        "🏆 Standings",
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

with dashboard:

    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Selected Leagues", len(selected_leagues))
    c2.metric("Upcoming Fixtures", len(upcoming))
    c3.metric("Predictions Generated", len(predictions))

    accuracy = history_accuracy()

    if accuracy:
        c4.metric("Settled 1X2 Accuracy", pct(accuracy[2]))
    else:
        c4.metric("Settled 1X2 Accuracy", "N/A")

    st.markdown("---")
    st.subheader("Competition Status")

    status_rows = []

    for league, df in league_data.items():
        status_rows.append({
            "League": league,
            "Finished Matches": len(finished_matches(df)),
            "Upcoming Matches": len(upcoming_matches(df)),
            "Status": "Available",
        })

    for league in errors:
        status_rows.append({
            "League": league,
            "Finished Matches": 0,
            "Upcoming Matches": 0,
            "Status": "Unavailable",
        })

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )

    if predictions:
        st.subheader("Upcoming Predictions")

        overview = pd.DataFrame(
            [prediction_row(p) for p in predictions]
        )

        st.dataframe(
            overview,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# FIXTURES AND PREDICTIONS
# =========================================================

with fixtures_tab:

    st.subheader("Fixtures & Predictions")

    if not predictions:
        st.info("No upcoming fixtures were found.")
    else:
        league_filter = st.selectbox(
            "Filter league",
            ["All Leagues"] + selected_leagues,
        )

        filtered = predictions

        if league_filter != "All Leagues":
            filtered = [
                p for p in predictions
                if p["league"] == league_filter
            ]

        display_df = pd.DataFrame(
            [prediction_row(p) for p in filtered]
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download predictions",
            display_df.to_csv(index=False).encode("utf-8"),
            "football_predictions.csv",
            "text/csv",
        )

        st.markdown("---")
        st.subheader("Detailed Match Analysis")

        options = {
            (
                f"{p['date']} | {p['home_team']} vs "
                f"{p['away_team']} ({p['league']})"
            ): p
            for p in filtered
        }

        selected_label = st.selectbox(
            "Select a fixture",
            list(options.keys()),
        )

        p = options[selected_label]

        st.markdown(
            f"### {p['home_team']} vs {p['away_team']}"
        )

        a, b, c = st.columns(3)

        a.metric("Home Win", pct(p["home_win"]))
        b.metric("Draw", pct(p["draw"]))
        c.metric("Away Win", pct(p["away_win"]))

        st.markdown("---")

        x, y = st.columns(2)

        x.metric("Expected Home Goals", f"{p['home_xg']:.2f}")
        y.metric("Expected Away Goals", f"{p['away_xg']:.2f}")

        st.markdown(
            f"**Most likely scoreline:** "
            f"{p['best_score'][0]} - {p['best_score'][1]}"
        )

        st.markdown("---")
        st.subheader("Goal Markets")

        market_rows = [
            ("Over 1.5 Goals", p["over_15"]),
            ("Over 2.5 Goals", p["over_25"]),
            ("Under 2.5 Goals", p["under_25"]),
            ("Over 3.5 Goals", p["over_35"]),
            ("Both Teams to Score — Yes", p["btts_yes"]),
            ("Both Teams to Score — No", p["btts_no"]),
        ]

        market_df = pd.DataFrame(
            [
                {
                    "Market": name,
                    "Probability": pct(value),
                    "Meets Threshold": (
                        "Yes" if value >= probability_threshold else "No"
                    ),
                }
                for name, value in market_rows
            ]
        )

        st.dataframe(
            market_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"The threshold is set to {probability_threshold:.2f}. "
            "It is a filtering preference, not a guarantee of accuracy."
        )

        # Correct-score probability table
        st.subheader("Correct Score Probabilities")

        score_rows = []

        for (home_goals, away_goals), probability in (
            p["score_probabilities"].items()
        ):
            score_rows.append({
                "Score": f"{home_goals} - {away_goals}",
                "Probability": probability,
            })

        score_df = pd.DataFrame(score_rows)
        score_df = score_df.sort_values(
            "Probability",
            ascending=False,
        ).head(15)

        score_df["Probability"] = (
            score_df["Probability"].apply(pct)
        )

        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
        )

        if st.button("💾 Save predictions to history"):
            save_predictions(filtered)
            st.success("Predictions saved to session history.")


# =========================================================
# PREDICTION HISTORY
# =========================================================

with history_tab:

    st.subheader("Prediction History")

    history = st.session_state[HISTORY_KEY]

    if not history:
        st.info(
            "No saved predictions yet. Save predictions from the "
            "Fixtures & Predictions tab."
        )
    else:
        rows = []

        for p in history:
            actual = (
                f"{p['actual_home']} - {p['actual_away']}"
                if p["settled"]
                else "Pending"
            )

            rows.append({
                "Date": p["date"],
                "League": p["league"],
                "Home Team": p["home_team"],
                "Away Team": p["away_team"],
                "Predicted Score": (
                    f"{p['best_score'][0]} - {p['best_score'][1]}"
                ),
                "Actual Score": actual,
                "Predicted Result": predicted_result(p),
            })

        history_df = pd.DataFrame(rows)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download prediction history",
            history_df.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv",
        )

        accuracy = history_accuracy()

        if accuracy:
            a, b, c = st.columns(3)
            a.metric("Settled Matches", accuracy[1])
            b.metric("Correct Predictions", accuracy[0])
            c.metric("1X2 Accuracy", pct(accuracy[2]))

        if st.button("Clear prediction history"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()

        st.caption(
            "History is stored in the current Streamlit session and "
            "may be cleared when the session restarts."
        )


# =========================================================
# BACKTESTING
# =========================================================

with backtest_tab:

    st.subheader("Historical Backtesting")

    st.write(
        "Test the model against previous matches. Each historical "
        "prediction uses only results available before that fixture."
    )

    backtest_league = st.selectbox(
        "Select league to backtest",
        selected_leagues,
        key="backtest_league",
    )

    league_history = finished[
        finished["league"] == backtest_league
    ].copy()

    st.caption(
        f"Finished matches available: {len(league_history)}"
    )

    if st.button("Run backtest"):
        with st.spinner("Running backtest..."):

            results = run_backtest(
                league_history,
                recent_matches,
                max_goals,
                backtest_limit,
            )

            st.session_state["backtest_results"] = results

    if "backtest_results" in st.session_state:

        results = st.session_state["backtest_results"]

        if results.empty:
            st.warning(
                "Not enough historical matches to complete the backtest. "
                "Try a lower minimum history requirement or another league."
            )
        else:
            total = len(results)
            correct = int(results["Correct"].sum())
            accuracy_value = correct / total

            a, b, c = st.columns(3)

            a.metric("Matches Evaluated", total)
            b.metric("Correct 1X2 Predictions", correct)
            c.metric("Historical Accuracy", pct(accuracy_value))

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download backtest results",
                results.to_csv(index=False).encode("utf-8"),
                "backtest_results.csv",
                "text/csv",
            )


# =========================================================
# STANDINGS
# =========================================================

with standings_tab:

    st.subheader("League Standings")

    standings_league = st.selectbox(
        "Choose competition",
        selected_leagues,
        key="standings_league",
    )

    try:
        standings = fetch_standings(
            LEAGUES[standings_league],
            token,
        )

        if not standings:
            st.info("Standings are not available for this competition.")
        else:
            for table in standings:

                group = table.get("group")
                title = table.get("type", "Standings")

                if group:
                    title += f" — {group}"

                st.markdown(f"### {title}")

                rows = []

                for entry in table.get("table", []):
                    team = entry.get("team", {})

                    rows.append({
                        "Position": entry.get("position"),
                        "Team": team.get("name"),
                        "Played": entry.get("playedGames"),
                        "Won": entry.get("won"),
                        "Draw": entry.get("draw"),
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
            f"Could not load standings: {error}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Football Prediction Centre | Powered by Streamlit and "
    "Football-Data.org | Statistical estimates only."
)
