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

drafted = final_base_data_draft_flag.dropna(subset=["Draft Team"]).copy()
positions = ["QB", "RB", "WR", "TE"]
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
st.set_page_config(page_title="Fantasy Draft War Room", layout="wide")

# ==========================
# SIDEBAR: GLOBAL CONTROLS
# ==========================
st.sidebar.header("⚙️ Settings")

# Draft configuration
league_size = st.sidebar.number_input("League Size (teams)", min_value=4, max_value=16, value=14, step=1)
total_rounds = st.sidebar.number_input("Total Rounds", min_value=8, max_value=24, value=15, step=1)
your_slot = st.sidebar.number_input("Your Draft Slot", min_value=1, max_value=league_size, value=1, step=1)
snake = st.sidebar.checkbox("Snake Draft", value=True)

# ADP source
adp_source = st.sidebar.selectbox("ADP Source", ["ADP HPPR", "ADP 2QB"], index=0)

# Scoring Actual or Projection
score_source = st.sidebar.selectbox("Stats or Projections", ['Pts 2024', 'Pts 2025'], index=0)

# Visibility / Sorting
show_drafted = st.sidebar.checkbox("Show Drafted Players on Board", value=False)
sort_option = st.sidebar.selectbox("Sort Players By", [adp_source, score_source])

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
colA.metric("Picks Made #", current_pick)
colB.metric("Your Next Pick #", next_pick if next_pick else "—")
colC.metric("Picks Until You", picks_until)

# Run detection for colD
lastN = drafted.sort_values("Draft Pick #", ascending=False).head(14)
run_counts = lastN["Pos"].value_counts()
run_text = ", ".join([f"{pos}: {count}" for pos, count in run_counts.items()]) if not run_counts.empty else "No picks yet"
colD.metric("Run Detection (last 14 picks)", run_text)


# Optional: existing warning if a run is happening
hot_pos = run_counts.index[0] if not run_counts.empty else None
if hot_pos and run_counts.iloc[0] >= 6:
  st.warning(f"{hot_pos} run in progress: {int(run_counts.iloc[0])} selected in last 14 picks")

# ==========================
# MY PICKS TRACKER
# ==========================
st.markdown("### My Picks")
my_picks = final_base_data_draft_flag[final_base_data_draft_flag["Draft Team"] == my_team]

position_colors = {"QB": "#4a90e2", "RB": "#50e3c2", "WR": "#e94e77", "TE": "#f5a623"}

if not my_picks.empty:
  positions = ["QB", "RB", "WR", "TE"]
  cols = st.columns(4)
  for i, pos in enumerate(positions):
      with cols[i]:
          st.markdown(f"**{pos}s**")
          df_pos = my_picks[my_picks["Pos"] == pos][["Name", "Team", "Bye", "Pts 25"]].reset_index(drop=True)
          if not df_pos.empty:
              for _, row in df_pos.iterrows():
                  proj_val = round(row['Pts 25'], 1) if pd.notna(row['Pts 25']) else "-"
                  # st.markdown(f"""
                  # <div style='border: 2px solid {position_colors.get(pos, "#cccccc")}; border-radius: 10px; padding: 8px; margin: 6px; background-color:#2f2f2f; color:#f0f0f0;'>
                  #     <div style='font-weight:700'>{row['Name']}</div>
                  #     <div style='font-size:12px'>{row['Team']} • Bye {row['Bye']}</div>
                  #     <div style='margin-top:4px; font-size:13px'>Proj: <b>{proj_val}</b></div>
                  # </div>
                  # """, unsafe_allow_html=True)
                  st.markdown(f"""
                  <div style='border: 2px solid {position_colors.get(pos, "#cccccc")};
                              border-radius: 10px; padding: 8px; margin: 6px;
                              background-color:#2f2f2f; color:#f0f0f0;
                              display: grid; grid-template-columns: 1fr 1fr 1fr; align-items:center;'>

                      <div style='font-weight:700'>{row['Name']}</div>
                      <div style='font-size:12px; text-align:center;'>{row['Team']} • Bye {row['Bye']}</div>
                      <div style='font-size:13px; text-align:right;'>Proj: <b>{proj_val}</b></div>

                  </div>
                  """, unsafe_allow_html=True)
          else:
              st.write("—")
else:
  st.info("No picks yet — will populate as draft progresses.")

# ==========================
# LEAGUE-WIDE INSIGHTS
# ==========================
st.subheader("🌐 League-Wide Insights")
drafted = final_base_data_draft_flag.dropna(subset=["Draft Team"]).copy()
if drafted.empty:
    st.info("Once draft picks populate, this section will show opponent needs and trends.")
else:
    positions = ["QB", "RB", "WR", "TE"]
    drafted = drafted[(drafted["Pos"].isin(positions)) & (drafted["Draft Team"] != 0)]

    # Pivot: rows = Pos, cols = Draft Team
    team_pos_counts = drafted.groupby(["Pos", "Draft Team"]).size().unstack(fill_value=0)
    # Custom color function
    def color_counts(val):
        if val >= 2:
            return 'background-color: #50e3c2; color: black; text-align: center;'  # green
        elif val == 1:
            return 'background-color: #f5a623; color: black; text-align: center;'  # orange
        elif val == 0:
            return 'background-color: #e94e77; color: black; text-align: center;'  # red
        else:
            return 'text-align: center;'


    # Apply styling
    styled_counts = (
        team_pos_counts
        .style
        .applymap(color_counts)
        .set_properties(**{'text-align': 'center'})
    )

    st.markdown("**Team Positional Counts (filtered, centered, and color-coded)**")
    st.table(styled_counts)

    st.markdown("**ADP vs Draft Trends**")
    if {"Draft Pick #", adp_source, "Pos", "Name"}.issubset(drafted.columns):
        trend = drafted.dropna(subset=["Draft Pick #", adp_source])
        trend = trend[trend['Draft Pick #'] > 0].copy()
        trend["Draft Pick #"] = trend["Draft Pick #"].astype(int)

        base = alt.Chart(trend)
        # x=y reference line
        identity_line = alt.Chart(
            pd.DataFrame({'x': [0, trend['Draft Pick #'].max()], 'y': [0, trend['Draft Pick #'].max()]})).mark_line(
            color='lightgray', strokeDash=[5, 5]).encode(
            x='x:Q',
            y='y:Q'
        )

        # vertical lines from point to x=y line
        vlines = alt.Chart(trend).mark_rule(color='lightgray', strokeWidth=1, opacity=0.5).encode(
            x='Draft Pick #:Q',
            y='Draft Pick #:Q',
            y2=f'{adp_source}:Q'
        )

        points = base.mark_circle(size=70).encode(
            x=alt.X("Draft Pick #", title="Overall Draft Pick"),
            y=alt.Y(f"{adp_source}:Q", title=f"{adp_source}"),
            color=alt.Color("Pos:N", scale=alt.Scale(domain=["QB", "RB", "WR", "TE"],
                                                     range=["#4a90e2", "#50e3c2", "#e94e77", "#f5a623"])),
            tooltip=["Name", "Team", "Pos", "Draft Pick #", adp_source]
        ).interactive()

        chart = identity_line + vlines + points
        st.altair_chart(chart, use_container_width=True)