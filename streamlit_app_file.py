import requests
import pandas as pd
import streamlit as st
import glob
import os


@st.cache_data
def get_all_players():
    """Pull all active players with primary position and bye week."""
    bye_week_data = {
        5: ['CHI', 'ATL', 'GB', 'PIT'], 6: ['HOU', 'MIN'], 7: ['BUF', 'BAL'],
        8: ['ARI', 'DET', 'JAX', 'LV', 'LAR', 'SEA'], 9: ['CLE', 'NYJ', 'PHI', 'TB'],
        10: ['CIN', 'DAL', 'KC', 'TEN'], 11: ['IND', 'NO'], 12: ['DEN', 'LAC', 'MIA', 'WAS'],
        14: ['CAR', 'NE', 'NYG', 'SF']
    }
    bye_week_df = pd.DataFrame([
        {"bye_week": week, "team": team} for week, teams in bye_week_data.items() for team in teams
    ])

    url = "https://api.sleeper.app/v1/players/nfl"
    data = requests.get(url).json()
    df = pd.DataFrame(data).T.reset_index()
    df = df[
        df['fantasy_positions'].apply(lambda x: isinstance(x, list) and any(p in x for p in ['QB', 'RB', 'WR', 'TE']))]
    df = df[df['team'].notna() & (df['team'] != "")]

    df['primary_fantasy_position'] = df['fantasy_positions'].apply(
        lambda pos_list: next((p for p in pos_list if p in ['QB', 'WR', 'RB', 'TE']), None)
    )
    df = df.merge(bye_week_df, on='team', how='left')
    cols = [
        'player_id', 'first_name', 'last_name', 'years_exp', 'primary_fantasy_position',
        'team', 'bye_week', 'depth_chart_position', 'depth_chart_order'
    ]
    df['first_name'] = df['first_name'].str.replace(" ", "", regex=False)
    df['last_name'] = df['last_name'].str.replace(" ", "", regex=False)
    return df[cols].reset_index(drop=True)


def process_sleeper_stats_or_projections(df: pd.DataFrame, stats_prefix: str, scoring_dict: dict):
    """Compute half-PPR fantasy points for stats or projections dataframe."""
    df = df.copy()

    if 'stats' in df.columns:
        stats_df = pd.json_normalize(df['stats']).add_prefix(f"{stats_prefix}_")
        df = pd.concat([df.drop(columns=['stats', 'player']), stats_df], axis=1)

    df[f'{stats_prefix}_calc_pts_half_ppr'] = 0
    for stat, pts in scoring_dict.items():
        col = f"{stats_prefix}_{stat}"
        if col in df.columns:
            df[f'{stats_prefix}_calc_pts_half_ppr'] += df[col].fillna(0) * pts
    return df.reset_index(drop=True)


@st.cache_data
def get_season_stats(season: int):
    url = f"https://api.sleeper.app/stats/nfl/{season}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE"
    df = pd.json_normalize(requests.get(url).json(), sep="_")

    scoring_dict = {
        'pass_int': -2.0, 'pass_2pt': 2.0, 'rec_td': 6.0, 'rush_td': 6.0, 'rec_2pt': 2.0,
        'rec': 0.5, 'fum_lost': -2.0, 'rush_2pt': 2.0, 'pass_yd': 0.04, 'pass_td': 4.0,
        'rush_yd': 0.1, 'rec_yd': 0.1
    }

    df = process_sleeper_stats_or_projections(df, stats_prefix="stats", scoring_dict=scoring_dict)

    keep_cols = [c for c in [
        "player_id", "season", "stats_calc_pts_half_ppr",
        "pass_att", "stats_pass_yd", "stats_pass_td", "stats_pass_int",
        "stats_rush_att", "stats_rush_yd", "stats_rush_td",
        "stats_rec", "stats_rec_tgt", "stats_rec_yd", "stats_rec_td",
        "stats_gp", "stats_fum_lost"
    ] if c in df.columns]

    return df[keep_cols].sort_values(by='stats_calc_pts_half_ppr', ascending=False).reset_index(drop=True)


@st.cache_data
def get_season_projections(season: int):
    url = f"https://api.sleeper.app/projections/nfl/{season}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE"
    df = pd.DataFrame(requests.get(url).json())

    scoring_dict = {
        'pass_int': -2.0, 'pass_2pt': 2.0, 'rec_td': 6.0, 'rush_td': 6.0, 'rec_2pt': 2.0,
        'rec': 0.5, 'fum_lost': -2.0, 'rush_2pt': 2.0, 'pass_yd': 0.04, 'pass_td': 4.0,
        'rush_yd': 0.1, 'rec_yd': 0.1
    }

    df = process_sleeper_stats_or_projections(df, stats_prefix="proj", scoring_dict=scoring_dict)

    keep_cols = [c for c in [
        "player_id", "season", "proj_calc_pts_half_ppr", "proj_adp_half_ppr", "proj_adp_2qb",
        "proj_pass_att", "proj_pass_td", "proj_pass_int", "proj_pass_yd", "proj_pass_2pt",
        "proj_rush_att", "proj_rush_yd", "proj_rush_td", "proj_rush_2pt",
        "proj_rec", "proj_rec_yd", "proj_rec_td", "proj_rec_2pt", "proj_fum_lost"
    ] if c in df.columns]

    return df[keep_cols].sort_values(by='proj_calc_pts_half_ppr', ascending=False).reset_index(drop=True)


def build_full_player_df(player_info, season_stats, season_projections):
    df = pd.merge(player_info, season_stats, on='player_id', how='inner')
    df = pd.merge(df, season_projections, on='player_id', how='inner', suffixes=('_stats', '_proj'))

    df = df.sort_values(by='proj_calc_pts_half_ppr', ascending=False)
    df['proj_overall_2025_rank'] = df['proj_calc_pts_half_ppr'].rank(ascending=False)
    df['stats_overall_2024_rank'] = df['stats_calc_pts_half_ppr'].rank(ascending=False)

    df['proj_position_2025_rank'] = df.groupby('primary_fantasy_position')['proj_calc_pts_half_ppr'].rank(
        ascending=False)
    df['stats_position_2024_rank'] = df.groupby('primary_fantasy_position')['stats_calc_pts_half_ppr'].rank(
        ascending=False)

    return df.reset_index(drop=True)


def pull_live_draft(draft_id: str):
    draft_url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
    draft_picks_response = requests.get(draft_url).json()
    draft_picks_df = pd.json_normalize(draft_picks_response)
    draft_picks_df.columns = [col.replace("metadata.", "") for col in draft_picks_df.columns]
    draft_picks_df = draft_picks_df.loc[:, ~draft_picks_df.columns.duplicated()]

    draft_pick_selected_columns = ['round', 'draft_slot', 'pick_no', 'player_id']
    draft_picks_df_clean = draft_picks_df[draft_pick_selected_columns]

    return draft_picks_df_clean


@st.cache_data
def league_rosters_scoring(league_id):
    league_url = f"https://api.sleeper.app/v1/league/{league_id}"
    league_settings_response = requests.get(league_url).json()

    roster_limits = league_settings_response['settings']
    roster_limit_dict = {k: v for k, v in roster_limits.items() if k.startswith('position_limit')}

    roster_structure = league_settings_response['roster_positions']
    scoring_settings = league_settings_response['scoring_settings']
    relevant_scoring_keys = {
        # Passing
        "pass_yd", "pass_td", "pass_int", "pass_2pt",
        # Rushing
        "rush_yd", "rush_td", "rush_2pt",
        # Receiving
        "rec", "rec_yd", "rec_td", "rec_2pt",
        # Turnovers
        "fum_lost"
    }
    scoring_structure_dict = {k: v for k, v in scoring_settings.items() if k in relevant_scoring_keys}
    return scoring_structure_dict, roster_limit_dict, roster_structure


def clean_column_names(df):
    """Cleans column names to be lowercase and replaces spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def process_fantasypros_df(file_path):
    """Reads a FantasyPros CSV, cleans column names, processes SOS SEASON, adds position, and splits player name."""
    df = pd.read_csv(file_path, header=0)
    df = clean_column_names(df)

    # Extract position from filename
    filename = os.path.basename(file_path)
    position = '_'.join(filename.split('_')[:4]).split('_')[-1]
    df['position'] = position

    # Extract the first character of 'sos_season'
    if 'sos_season' in df.columns:
        df['sos_season'] = df['sos_season'].astype(str).str[0]

    df['player_name'] = df['player_name'].str.replace("St. Brown", "St.Brown", regex=False).str.replace(
        "Marquise Brown", "Hollywood Brown", regex=False)
    # Split player_name into first_name and last_name
    if 'player_name' in df.columns:
        name_parts = df['player_name'].str.split(' ')
        df['first_name'] = name_parts.str[0].str.replace(" ", "", regex=False)
        # Handle names with more than two parts, taking the last part as the last name
        df['last_name'] = name_parts.str[1].str.replace(" ", "", regex=False)
    df = df.drop(columns=['rk', 'bye_week', 'ecr_vs._adp'], errors='ignore')
    df['team'] = df['team'].replace('JAC', 'JAX')
    return df[['first_name', 'last_name', 'position', 'sos_season', 'tiers']]


st.title("Fantasy Football Projections")
scoring_structure_dict, roster_limit_dict, roster_structure = league_rosters_scoring('1182045780030189568')
player_info = get_all_players()
season_stats = get_season_stats(2024)
season_projections = get_season_projections(2025)
full_player_data = build_full_player_df(player_info, season_stats, season_projections)
file_paths = glob.glob("FantasyPros_2025_Draft_*.csv")
all_fantasypros_df = pd.concat([process_fantasypros_df(f) for f in file_paths], ignore_index=True)
draft_picks_live = pull_live_draft('1232794131110055936')

### Create and Clean Final Master Table

full_player_data_test_tiers = full_player_data.copy()
full_player_data_test_tiers["name"] = full_player_data_test_tiers["first_name"] + " " + full_player_data_test_tiers[
    "last_name"]

final_base_data = pd.merge(
    full_player_data_test_tiers,
    all_fantasypros_df,
    left_on=['first_name', 'last_name', 'primary_fantasy_position'],
    right_on=['first_name', 'last_name', 'position'],
    how='left'
)[[
    'player_id', 'name', 'position', 'stats_calc_pts_half_ppr', 'proj_calc_pts_half_ppr',
    'stats_position_2024_rank', 'proj_position_2025_rank', 'tiers', 'sos_season', 'stats_gp',
    'proj_adp_half_ppr', 'proj_adp_2qb', 'team', 'bye_week', 'depth_chart_order', 'years_exp',
    'stats_pass_yd', 'stats_pass_td', 'stats_pass_int',
    'proj_pass_yd', 'proj_pass_td', 'proj_pass_int',
    'stats_rush_att', 'stats_rush_yd', 'stats_rush_td',
    'proj_rush_att', 'proj_rush_yd', 'proj_rush_td',
    'stats_rec_tgt', 'stats_rec', 'stats_rec_yd', 'stats_rec_td',
    'proj_rec', 'proj_rec_yd', 'proj_rec_td'
]]

col_rename_map = {
    # Player Basics and Full Season Data
    # 'player_id': 'ID',
    'name': 'Name',
    'position': 'Pos',
    'stats_calc_pts_half_ppr': 'Pts 24',
    'proj_calc_pts_half_ppr': 'Pts 25',

    # Player Rankings
    'stats_position_2024_rank': 'Rank 24',
    'proj_position_2025_rank': 'Rank 25',
    'tiers': 'Tier',
    'sos_season': 'SOS',
    'stats_gp': 'GP 24',

    # Additional Player Details
    'proj_adp_half_ppr': 'ADP HPPR',
    'proj_adp_2qb': 'ADP 2QB',
    'team': 'Team',
    'bye_week': 'Bye',
    'depth_chart_order': 'Depth',
    'years_exp': 'Exp',

    # Passing
    'stats_pass_yd': 'Pass YD 24',
    'stats_pass_td': 'Pass TD 24',
    'stats_pass_int': 'Int 24',
    'proj_pass_yd': 'Pass YD 25',
    'proj_pass_td': 'Pass TD 25',
    'proj_pass_int': 'Int 25',

    # Rushing
    'stats_rush_att': 'Rush Att 24',
    'stats_rush_yd': 'Rush YD 24',
    'stats_rush_td': 'Rush TD 24',
    'proj_rush_att': 'Rush Att 25',
    'proj_rush_yd': 'Rush YD 25',
    'proj_rush_td': 'Rush TD 25',

    # Receiving
    'stats_rec_tgt': 'Tgt 24',
    'stats_rec': 'Rec 24',
    'stats_rec_yd': 'Rec YD 24',
    'stats_rec_td': 'Rec TD 24',
    'proj_rec': 'Rec 25',
    'proj_rec_yd': 'Rec YD 25',
    'proj_rec_td': 'Rec TD 25',

    # Live Draft
    'round': 'Draft Round',
    'draft_slot': 'Draft Team',
    'pick_no': 'Draft Pick #'
}

final_base_data_draft_flag = pd.merge(
    final_base_data,
    draft_picks_live, on='player_id', how='left').rename(columns=col_rename_map)

col_types = {
    # Decimal columns (round to 1 decimal)
    'Pts 2024': 'float',
    'Pts 2025': 'float',
    'ADP HPPR': 'float',
    'ADP 2QB': 'float',

    # Integer columns
    'Pass YD 24': 'int',
    'Pass YD 25': 'int',
    'Rush YD 24': 'int',
    'Rush YD 25': 'int',
    'Rec YD 24': 'int',
    'Rec YD 25': 'int',
    'Rank 2024': 'int',
    'Rank 2025': 'int',
    'GP 24': 'int',
    'Tgt 24': 'int',
    'Rec 24': 'int',
    'Rec 25': 'int',
    'Rush Att 24': 'int',
    'Rush Att 25': 'int',
    'Rush TD 24': 'int',
    'Rush TD 25': 'int',
    'Pass TD 24': 'int',
    'Pass TD 25': 'int',
    'Int 24': 'int',
    'Int 25': 'int',
    'Rec TD 24': 'int',
    'Rec TD 25': 'int',
    'Depth': 'int',
    'Exp': 'int',
    'Bye': 'int',
    'Tier': 'int',
    'SOS': 'int',
    'Draft Round': 'int',
    'Draft Team': 'int',
    'Draft Pick #': 'int',

    # String columns
    'Name': 'string',
    'Pos': 'string',
    'Team': 'string'
}

for col, typ in col_types.items():
    if col in final_base_data_draft_flag.columns:
        if typ == 'int':
            final_base_data_draft_flag[col] = final_base_data_draft_flag[col].fillna(0).astype(int)
        elif typ == 'float':
            final_base_data_draft_flag[col] = final_base_data_draft_flag[col].astype(float).round(1)
        elif typ == 'string':
            final_base_data_draft_flag[col] = final_base_data_draft_flag[col].astype(str)

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==========================
# CONFIG & ASSUMPTIONS
# ==========================
# Assumes a DataFrame named `final_base_data_draft_flag` is already present in memory with the
# columns you described (stats, projections, ADP, tiers, draft metadata, etc.).
# This app is built to be refreshed live as your draft progresses.

st.set_page_config(page_title="Fantasy Draft War Room", layout="wide")

# ==========================
# SIDEBAR: GLOBAL CONTROLS
# ==========================
st.sidebar.header("⚙️ Settings")

# Draft configuration
league_size = st.sidebar.number_input("League Size (teams)", min_value=4, max_value=16, value=12, step=1)
total_rounds = st.sidebar.number_input("Total Rounds", min_value=8, max_value=24, value=16, step=1)
your_slot = st.sidebar.number_input("Your Draft Slot (1-based)", min_value=1, max_value=league_size, value=1, step=1)
snake = st.sidebar.checkbox("Snake Draft", value=True)

# ADP source
adp_source = st.sidebar.selectbox("ADP Source", ["ADP HPPR", "ADP 2QB"], index=0)

# Visibility / Sorting
show_drafted = st.sidebar.checkbox("Show Drafted Players on Board", value=False)
sort_option = st.sidebar.selectbox("Sort Players By", [adp_source, "Pts 25"])

# Best Value Weights
st.sidebar.subheader("Best Value Weights")
w_pts = st.sidebar.slider("Weight: Projected Points", 0.0, 2.0, 1.0, 0.1)
w_adp = st.sidebar.slider("Weight: ADP Gap", 0.0, 2.0, 0.7, 0.1)
w_scar = st.sidebar.slider("Weight: Positional Scarcity", 0.0, 2.0, 0.6, 0.1)

# Starter assumptions
st.sidebar.subheader("Starter Slots Per Team")
starter_qb = st.sidebar.number_input("QB starters", 0, 2, 1)
starter_rb = st.sidebar.number_input("RB starters", 0, 4, 2)
starter_wr = st.sidebar.number_input("WR starters", 0, 5, 2)
starter_te = st.sidebar.number_input("TE starters", 0, 3, 1)

# ==========================
# HELPER FUNCTIONS
# ==========================

def get_current_pick(df: pd.DataFrame) -> int:
    if df["Draft Pick #"].notna().any():
        return int(df["Draft Pick #"].dropna().astype(int).max())
    return 0


def compute_your_future_picks(league_size: int, total_rounds: int, your_slot: int, snake: bool) -> list:
    picks = []
    for rnd in range(1, total_rounds + 1):
        if snake and (rnd % 2 == 0):
            pick_in_round = league_size - your_slot + 1
        else:
            pick_in_round = your_slot
        overall = (rnd - 1) * league_size + pick_in_round
        picks.append(overall)
    return picks


def next_pick_info(df: pd.DataFrame, league_size: int, total_rounds: int, your_slot: int, snake: bool):
    current_pick = get_current_pick(df)
    your_picks = compute_your_future_picks(league_size, total_rounds, your_slot, snake)
    upcoming = [p for p in your_picks if p > current_pick]
    if not upcoming:
        return current_pick, None, 0
    next_pick = min(upcoming)
    picks_until = max(0, next_pick - current_pick - 1)
    return current_pick, next_pick, picks_until


def compute_scarcity(df: pd.DataFrame) -> pd.Series:
    available = df[df["Draft Team"].isna()]
    if available.empty:
        return pd.Series(dtype=float)
    total_by_pos = df.groupby("Pos").size()
    rem_by_pos = available.groupby("Pos").size()
    scarcity = (rem_by_pos / total_by_pos).reindex(total_by_pos.index).fillna(0.0)
    return scarcity


def best_value_scores(df: pd.DataFrame, next_pick: int, weights: tuple[float, float, float]) -> pd.DataFrame:
    w_pts, w_adp, w_scar = weights
    data = df.copy()
    data = data[data["Draft Team"].isna()].copy()
    if data.empty:
        return data

    data["pts_z"] = data.groupby("Pos")["Pts 25"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    adp = data[adp_source].astype(float)
    data["adp_gap"] = (adp - next_pick) * -1
    data["adp_gap_z"] = (data["adp_gap"] - data["adp_gap"].mean()) / (data["adp_gap"].std(ddof=0) + 1e-9)
    scarcity = compute_scarcity(df)
    data["scarcity_pos"] = data["Pos"].map(lambda p: 1.0 - float(scarcity.get(p, 0.0)))
    data["scarcity_z"] = (data["scarcity_pos"] - data["scarcity_pos"].mean()) / (data["scarcity_pos"].std(ddof=0) + 1e-9)
    data["bv_score"] = w_pts * data["pts_z"] + w_adp * data["adp_gap_z"] + w_scar * data["scarcity_z"]
    return data


def likely_gone_flag(row: pd.Series, next_pick: int, teams_need_factor: float) -> bool:
    try:
        adp_val = float(row.get(adp_source, np.nan))
    except Exception:
        return False
    if np.isnan(adp_val) or next_pick is None:
        return False
    return adp_val <= (next_pick + max(0, int(2 * teams_need_factor)))

# ==========================
# DEDUCE YOUR TEAM FROM DRAFT SLOT
# ==========================
my_slot_picks = final_base_data_draft_flag[final_base_data_draft_flag["Draft Pick #"] == your_slot]
if not my_slot_picks.empty:
    my_team = my_slot_picks.iloc[0]["Draft Team"]
else:
    my_team = None  # will populate after first round

# ==========================
# COMMAND CENTER
# ==========================
current_pick, next_pick, picks_until = next_pick_info(final_base_data_draft_flag, league_size, total_rounds, your_slot, snake)

colA, colB, colC, colD = st.columns(4)
colA.metric("Current Pick #", current_pick)
colB.metric("Your Next Pick #", next_pick if next_pick else "—")
colC.metric("Picks Until You", picks_until)

# Lowest Tier Remaining by Position
lowest_tier_counts = {}
for pos, dfp in final_base_data_draft_flag[final_base_data_draft_flag["Draft Team"].isna()].groupby("Pos"):
    if dfp["Tier"].notna().any():
        min_tier = dfp["Tier"].min()
        lowest_tier_counts[pos] = int((dfp[dfp["Tier"] == min_tier]).shape[0])
    else:
        lowest_tier_counts[pos] = 0
colD.write("**Lowest Tier Remaining (by Pos)**")
colD.write(pd.DataFrame([lowest_tier_counts]))

# ==========================
# MY PICKS TRACKER
# ==========================
st.markdown("### My Picks")
my_picks = final_base_data_draft_flag[final_base_data_draft_flag["Draft Team"] == my_team]
if not my_picks.empty:
    st.dataframe(my_picks[["Name", "Pos", "Bye", "Pts 25"]].reset_index(drop=True))
else:
    st.info("No picks yet — will populate as draft progresses.")

# ==========================
# BYE WEEK OVERLAPS
# ==========================
st.markdown("### 🧭 My Team Snapshot — Bye Week Overlaps")
if my_picks.empty or my_picks["Bye"].isna().all():
    st.write("(Will populate as soon as you make picks.)")
else:
    bye_counts = my_picks.groupby("Bye").size().rename("Count").reset_index()
    chart = alt.Chart(bye_counts).mark_bar().encode(
        x=alt.X("Bye:O", title="Bye Week"),
        y=alt.Y("Count:Q", title="# of Your Players"),
        tooltip=["Bye", "Count"]
    )
    st.altair_chart(chart, use_container_width=True)

# ==========================
# VISUAL TIER BOARD & BEST VALUE
# ==========================
st.subheader("Visual Tier Board — All Positions")
positions = ["QB", "RB", "WR", "TE"]
cols = st.columns(len(positions))
team_drafted_pos = final_base_data_draft_flag.dropna(subset=["Draft Team"]).groupby(["Draft Team", "Pos"]).size().unstack(fill_value=0)
need_factor = float((team_drafted_pos == 0).sum().sum()) / max(1, league_size)

for i, pos in enumerate(positions):
    with cols[i]:
        st.markdown(f"### {pos}")
        pos_df = final_base_data_draft_flag[final_base_data_draft_flag["Pos"] == pos].copy()
        if not show_drafted:
            pos_df = pos_df[pos_df["Draft Team"].isna()]
        if sort_option == adp_source:
            pos_df = pos_df.sort_values(adp_source)
        else:
            pos_df = pos_df.sort_values("Pts 25", ascending=False)
        for tier, tier_df in pos_df.groupby("Tier"):
            available_tier = tier_df[tier_df["Draft Team"].isna()]
            st.markdown(f"**Tier {tier} — {available_tier.shape[0]} left**")
            cards_html = []
            for _, r in tier_df.iterrows():
                is_risky = False
                if next_pick is not None:
                    is_risky = likely_gone_flag(r, next_pick, need_factor)
                adp_val = r.get(adp_source, np.nan)
                pts_val = r.get("Pts 25", np.nan)
                bye_val = r.get("Bye", "-")
                team_val = r.get("Team", "-")
                name = r.get("Name", "-")
                drafted_flag = pd.notna(r.get("Draft Team"))
                card_color = "#f0f0f0" if drafted_flag else "white"
                border = {
                    "QB": "#4a90e2",
                    "RB": "#50e3c2",
                    "WR": "#e94e77",
                    "TE": "#f5a623",
                }.get(pos, "#cccccc")
                risk_glow = "box-shadow: 0 0 10px 2px rgba(255,0,0,0.4);" if is_risky else ""
                cards_html.append(f"""
                <div style='border: 2px solid {border}; border-radius: 10px; padding: 8px; margin: 6px; {risk_glow} background-color:{card_color}; opacity:{'0.4' if drafted_flag else '1'};'>
                    <div style='font-weight:700'>{name}</div>
                    <div style='font-size:12px; color:#666'>{team_val} • Bye {bye_val}</div>
                    <div style='margin-top:4px; font-size:13px'>Pts 25: <b>{pts_val}</b> &nbsp; | &nbsp; ADP: <b>{adp_val}</b></div>
                </div>
                """)
            st.markdown("".join(cards_html), unsafe_allow_html=True)

# Compute Best Value
st.subheader("🎯 Best Value Recommendations")
bv = best_value_scores(final_base_data_draft_flag, next_pick if next_pick else 9999, (w_pts, w_adp, w_scar))
priority_pos = None
if not bv.empty:
    top_by_pos = bv.sort_values("bv_score", ascending=False).groupby("Pos").head(1).set_index("Pos")
    if not top_by_pos.empty:
        priority_pos = top_by_pos["bv_score"].idxmax()
st.write(f"**Position Priority Now:** {priority_pos if priority_pos else 'N/A'}")

for pos in positions:
    st.markdown(f"**Top 5 {pos}s**")
    sub = bv[bv["Pos"] == pos].copy()
    if sub.empty:
        st.write("(No players available)")
        continue
    sub["ADP Gap vs Next Pick"] = (sub[adp_source].astype(float) - (next_pick if next_pick else np.nan))
    cols_show = ["Name", "Team", "Pts 25", adp_source, "Tier", "ADP Gap vs Next Pick", "bv_score"]
    st.dataframe(sub.sort_values("bv_score", ascending=False)[cols_show].head(5).reset_index(drop=True))

# ==========================
# LEAGUE-WIDE INSIGHTS
# ==========================
st.subheader("🌐 League-Wide Insights")
drafted = final_base_data_draft_flag.dropna(subset=["Draft Team"]).copy()
if drafted.empty:
    st.info("Once draft picks populate, this section will show opponent needs and trends.")
else:
    team_pos_counts = drafted.groupby(["Draft Team", "Pos"]).size().unstack(fill_value=0)
    for p in positions:
        if p not in team_pos_counts.columns:
            team_pos_counts[p] = 0
    need_matrix = pd.DataFrame(index=team_pos_counts.index)
    for p in positions:
        need_matrix[p] = (team_pos_counts[p] < {"QB": starter_qb, "RB": starter_rb, "WR": starter_wr, "TE": starter_te}[p]).astype(int)
    st.markdown("**Opponent Starter Needs (1 = still needs starters)**")
    st.dataframe(need_matrix)

    st.markdown("**Run Detection (last 12 picks)**")
    lastN = drafted.sort_values("Draft Pick #", ascending=False).head(12)
    run_counts = lastN["Pos"].value_counts()
    st.write(run_counts.to_frame("Count"))
    hot_pos = run_counts.index[0] if not run_counts.empty else None
    if hot_pos and run_counts.iloc[0] >= 6:
        st.warning(f"{hot_pos} run in progress: {int(run_counts.iloc[0])} selected in last 12 picks")

    st.markdown("**ADP vs Draft Trends**")
    if {"Draft Pick #", adp_source, "Pos", "Name"}.issubset(drafted.columns):
        trend = drafted.dropna(subset=["Draft Pick #", adp_source]).copy()
        trend["Draft Pick #"] = trend["Draft Pick #"].astype(int)
        chart = alt.Chart(trend).mark_circle(size=70).encode(
            x=alt.X("Draft Pick #", title="Overall Draft Pick"),
            y=alt.Y(f"{adp_source}:Q", title=f"{adp_source}"),
            color=alt.Color("Pos:N", scale=alt.Scale(domain=["QB","RB","WR","TE"], range=["#4a90e2", "#50e3c2", "#e94e77", "#f5a623"])),
            tooltip=["Name", "Team", "Pos", "Draft Pick #", adp_source]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# ==========================
# ADVANCED ANALYTICS — POSITIONAL DROP-OFFS
# ==========================
st.subheader("📉 Positional Drop-Offs (ADP vs Projected Points)")
for pos in positions:
    st.markdown(f"**{pos}**")
    sub = final_base_data_draft_flag[final_base_data_draft_flag["Pos"] == pos].dropna(subset=[adp_source, "Pts 25"]).copy()
    if sub.empty:
        st.write("No data available for chart.")
        continue
    chart = alt.Chart(sub).mark_line(point=True).encode(
        x=alt.X(f"{adp_source}:Q", title=f"{adp_source} (lower = earlier)"),
        y=alt.Y("Pts 25:Q", title="Projected Points 2025"),
        tooltip=["Name", "Team", adp_source, "Pts 25", "Tier"]
    )
    st.altair_chart(chart, use_container_width=True)
