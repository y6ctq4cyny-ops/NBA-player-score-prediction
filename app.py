#Local URL: http://localhost:8501 Network URL: http://192.168.0.39:8501
#!/usr/bin/env python3
import streamlit as st
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, leaguedashteamstats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# =======================================================================
# SETTINGS
# =======================================================================
SEASONS = ["2024-25", "2023-24"]
LATEST_SEASON = "2024-25"
SIMULATIONS = 30000

# =======================================================================
# SUPERSTAR TIERS
# =======================================================================
TIER_1 = {
    "Luka Doncic", "Nikola Jokic", "Joel Embiid", "Shai Gilgeous-Alexander",
    "Giannis Antetokounmpo", "Kevin Durant", "Jayson Tatum", "Steph Curry",
    "Anthony Edwards", "Devin Booker"
}

TIER_2 = {
    "Donovan Mitchell", "Kawhi Leonard", "LeBron James",
    "Anthony Davis", "Kyrie Irving", "Trae Young",
    "Zion Williamson", "Jimmy Butler", "Jalen Brunson",
    "Domantas Sabonis", "Bam Adebayo", "Jaylen Brown"
}


# =======================================================================
# DATA LOADING FUNCTIONS
# =======================================================================
def load_player(player_name):
    plist = players.get_players()
    match = next((p for p in plist if p["full_name"].lower() == player_name.lower()), None)
    if not match:
        st.error("Player not found.")
        st.stop()
    pid = match["id"]
    info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
    return pid, match["full_name"], int(info["TEAM_ID"].iloc[0]), info["TEAM_NAME"].iloc[0]


def load_logs(player_id):
    frames = []
    for season in SEASONS:
        try:
            df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            if not df.empty:
                df["SEASON"] = season
                frames.append(df)
        except:
            continue

    if not frames:
        st.error("No logs available.")
        st.stop()

    gl = pd.concat(frames, ignore_index=True)

    numerics = ["PTS", "FGA", "FG3A", "FTA", "AST"]
    for col in numerics:
        gl[col] = pd.to_numeric(gl[col], errors="coerce")

    gl["MIN"] = gl["MIN"].astype(str).replace(":", ".", regex=True)
    gl["MIN"] = pd.to_numeric(gl["MIN"], errors="coerce")

    return gl


def load_opponent_metrics(opponent_name):
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=LATEST_SEASON,
        per_mode_detailed="PerGame"
    ).get_data_frames()[0]

    row = df[df["TEAM_NAME"] == opponent_name]

    if row.empty:
        st.error("Opponent not found in NBA data.")
        st.stop()

    opp_pts_allowed = float(row["PTS"].iloc[0])
    lg_pts_allowed = df["PTS"].mean()

    if "PACE" in df.columns:
        opp_pace = float(row["PACE"].iloc[0])
        lg_pace = df["PACE"].mean()
    else:
        opp_pace = 98.5
        lg_pace = 98.5
        st.warning("⚠ NBA API returned no PACE data — using league average.")

    return opp_pts_allowed, lg_pts_allowed, opp_pace, lg_pace


# =======================================================================
# CONTEXTUAL EXPECTED POINTS
# =======================================================================
def contextual_expected_points(gl, df_vs, metrics,
                               homeaway_factor, fatigue_factor, altitude_factor,
                               player_name):

    opp_def, league_def, opp_pace, league_pace = metrics

    recent10 = gl.head(10)["PTS"].mean()
    recent5 = gl.head(5)["PTS"].mean()
    season = gl["PTS"].mean()

    cold_gap = max(0, season - recent5)
    bounce_boost = cold_gap * 0.50
    hot = (recent5 - season) * 0.50 + bounce_boost

    gl["USG"] = gl["FGA"] + 0.44 * gl["FTA"] + gl["AST"]
    usage_recent = gl.head(7)["USG"].mean()
    usage_avg = gl["USG"].mean()
    usage_boost = (usage_recent - usage_avg) * 0.12

    fga_boost = (gl.head(7)["FGA"].mean() - gl["FGA"].mean()) * 0.60

    def_boost = (league_def - opp_def) * 0.20
    pace_boost = (opp_pace - league_pace) * 0.20

    if len(df_vs) >= 2:
        matchup_adj = 0.15 * (df_vs["PTS"].mean() - season)
    else:
        matchup_adj = 0

    expected = (
        0.50 * recent10 +
        0.20 * recent5 +
        0.20 * season +
        0.10 * df_vs["PTS"].mean()
    )

    expected += hot + usage_boost + fga_boost + def_boost + pace_boost + matchup_adj

    if player_name in TIER_1:
        expected += 2.2
        expected = max(expected, 26)
    elif player_name in TIER_2:
        expected += 1.0
        expected = max(expected, 22)

    expected += homeaway_factor + fatigue_factor + altitude_factor

    return max(expected, 8)


# =======================================================================
# BASE SIMULATION
# =======================================================================
def run_base_sim(gl, metrics):

    opp_def, league_def, opp_pace, league_pace = metrics
    last10 = gl.head(10)

    min_mean = last10["MIN"].mean()
    min_std = max(1.5, last10["MIN"].std())

    usage = (last10["FGA"] / last10["MIN"]).mean()
    usage_std = max(0.02, (last10["FGA"] / last10["MIN"]).std())

    three_rate = np.nan_to_num((last10["FG3A"] / last10["FGA"].replace(0, np.nan)).mean(), nan=0.25)
    ft_rate = np.nan_to_num((last10["FTA"] / last10["FGA"].replace(0, np.nan)).mean(), nan=0.10)

    foul_mean = last10["PF"].mean()
    foul_std = max(0.7, last10["PF"].std())

    results = []

    for _ in range(SIMULATIONS):

        mins = np.clip(np.random.normal(min_mean, min_std), 18, 44)

        pf = np.random.normal(foul_mean, foul_std)
        if pf >= 5.5:
            mins *= np.random.uniform(0.45, 0.75)

        usage_t = np.random.normal(usage, usage_std)

        pace_adj = opp_pace / league_pace
        def_adj = (league_def / opp_def)

        fga = np.clip(usage_t * mins * pace_adj * def_adj, 3, 35)

        fg3a = fga * three_rate
        fta = fga * ft_rate
        fg2a = fga - fg3a

        fg2_pct = np.random.normal(0.52, 0.05) * def_adj
        fg3_pct = np.random.normal(0.35, 0.05) * def_adj
        ft_pct = np.random.normal(0.78, 0.03)

        fg2_m = np.random.binomial(int(fg2a), np.clip(fg2_pct, 0.05, 0.75))
        fg3_m = np.random.binomial(int(fg3a), np.clip(fg3_pct, 0.05, 0.60))
        ft_m = np.random.binomial(int(fta), np.clip(ft_pct, 0.50, 0.95))

        points = fg2_m * 2 + fg3_m * 3 + ft_m
        results.append(points)

    return np.array(results)


# =======================================================================
# CALIBRATION LAYER
# =======================================================================
def calibrate_sim(base, target_ep, gl, metrics):

    sim_mean = np.mean(base)
    error = target_ep - sim_mean

    usage_scale = 1 + (error / max(10, sim_mean)) * 0.60
    eff_scale = 1 + (error / max(10, sim_mean)) * 0.40

    opp_def, league_def, opp_pace, league_pace = metrics
    last10 = gl.head(10)

    min_mean = last10["MIN"].mean()
    min_std = max(1.5, last10["MIN"].std())

    usage = (last10["FGA"] / last10["MIN"]).mean() * usage_scale
    usage_std = max(0.02, (last10["FGA"] / last10["MIN"]).std())

    three_rate = np.nan_to_num((last10["FG3A"] / last10["FGA"].replace(0, np.nan)).mean(), nan=0.25)
    ft_rate = np.nan_to_num((last10["FTA"] / last10["FGA"].replace(0, np.nan)).mean(), nan=0.10)

    foul_mean = last10["PF"].mean()
    foul_std = max(0.7, last10["PF"].std())

    results = []

    for _ in range(SIMULATIONS):

        mins = np.clip(np.random.normal(min_mean, min_std), 18, 44)

        pf = np.random.normal(foul_mean, foul_std)
        if pf >= 5.5:
            mins *= np.random.uniform(0.45, 0.75)

        usage_t = np.random.normal(usage, usage_std)

        pace_adj = opp_pace / league_pace
        def_adj = (league_def / opp_def)

        fga = np.clip(usage_t * mins * pace_adj * def_adj, 3, 35)

        fg3a = fga * three_rate
        fta = fga * ft_rate
        fg2a = fga - fg3a

        fg2_pct = np.clip(np.random.normal(0.52, 0.05) * eff_scale * def_adj, 0.05, 0.75)
        fg3_pct = np.clip(np.random.normal(0.35, 0.05) * eff_scale * def_adj, 0.05, 0.65)
        ft_pct = np.clip(np.random.normal(0.78, 0.03) * eff_scale, 0.50, 0.97)

        fg2_m = np.random.binomial(int(fg2a), fg2_pct)
        fg3_m = np.random.binomial(int(fg3a), fg3_pct)
        ft_m = np.random.binomial(int(fta), ft_pct)

        results.append(fg2_m * 2 + fg3_m * 3 + ft_m)

    return np.array(results)


# =======================================================================
# KDE + TAIL CORRECTION
# =======================================================================
def kde_tail(samples):
    kde = gaussian_kde(samples)
    xs = np.linspace(samples.min(), samples.max(), 400)
    density = kde(xs)
    density[xs > np.percentile(samples, 95)] *= 1.15
    density /= np.trapz(density, xs)
    return xs, density


# =======================================================================
# STREAMLIT APP
# =======================================================================

st.title("🏀 Advanced NBA Player Points Predictor")
st.write("Monte Carlo Simulation + Contextual Engine + KDE Tail Modeling")

player_name = st.text_input("Enter Player Name", "Nikola Jokic")

team_list = teams.get_teams()
TEAM_MAP = {t["abbreviation"]: t["full_name"] for t in team_list}
opp = st.selectbox("Opponent Team", list(TEAM_MAP.keys()))
loc = st.radio("Location", ("Home", "Away"))
b2b = st.radio("Back-to-back?", ("No", "Yes"))

run_button = st.button("Run Prediction")

if run_button:

    PID, NAME, TEAM_ID, TEAM_NAME = load_player(player_name)
    gl = load_logs(PID)

    df_vs = gl[gl["MATCHUP"].str.contains(opp, case=False, na=False)]
    if df_vs.empty:
        df_vs = gl.head(5)

    homeaway_factor = 1.5 if loc == "Home" else -1.5
    fatigue_factor = -1.4 if b2b == "Yes" else 0
    altitude_factor = 1.0 if TEAM_NAME == "Nuggets" and loc == "Home" else 0

    metrics = load_opponent_metrics(TEAM_MAP[opp])

    ep = contextual_expected_points(
        gl, df_vs, metrics,
        homeaway_factor, fatigue_factor, altitude_factor,
        NAME
    )

    st.subheader(f"🎯 Contextual Expected Points: **{ep:.2f}**")

    base_sim = run_base_sim(gl, metrics)
    calibrated_sim = calibrate_sim(base_sim, ep, gl, metrics)

    xs, density = kde_tail(calibrated_sim)

    st.subheader("📊 Simulation Results")
    st.write(f"Monte Carlo Mean: **{np.mean(calibrated_sim):.2f}**")
    st.write(f"Monte Carlo Median: **{np.median(calibrated_sim):.2f}**")

    st.write("---")
    st.write("### Probability Table")

    for line in [15, 18, 20, 22.5, 25, 30, 35, 40]:
        st.write(f"**P > {line}:** {np.mean(calibrated_sim > line):.2%}")

    st.write("---")
    st.write("### Distribution Plot")

    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(calibrated_sim, bins=40, density=True, alpha=0.6, label="Simulation")
    ax.plot(xs, density, color="red", linewidth=2, label="KDE Tail")
    ax.set_title(f"{NAME} vs {TEAM_MAP[opp]} — Points Distribution")
    ax.set_xlabel("Points")
    ax.set_ylabel("Density")
    ax.legend()

    st.pyplot(fig)
