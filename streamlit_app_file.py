import requests
import pandas as pd
import streamlit as st


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


def pull_live_draft(draft_id: str, full_player_stat_proj):
    draft_url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
    draft_picks_response = requests.get(draft_url).json()
    draft_picks_df = pd.json_normalize(draft_picks_response)
    draft_picks_df.columns = [col.replace("metadata.", "") for col in draft_picks_df.columns]
    draft_picks_df = draft_picks_df.loc[:, ~draft_picks_df.columns.duplicated()]

    draft_pick_selected_columns = ['round', 'draft_slot', 'pick_no', 'player_id']
    draft_picks_df_clean = draft_picks_df[draft_pick_selected_columns]
    draft_picks_live = pd.merge(draft_picks_df_clean, full_player_stat_proj, on='player_id', how='left')

    return draft_picks_live.reset_index(drop=True)


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

st.title("Fantasy Football Projections")
scoring_structure_dict, roster_limit_dict, roster_structure = league_rosters_scoring('1182045780030189568')
player_info = get_all_players()
season_stats = get_season_stats(2024)
season_projections = get_season_projections(2025)
full_player_data = build_full_player_df(player_info, season_stats, season_projections)
draft_picks_live = pull_live_draft('1232794131110055936', full_player_data)

import streamlit as st
import pandas as pd

# --- Assume full_player_data & draft_picks_live are already loaded ---
df = full_player_data.copy()
df["name"] = df["first_name"] + " " + df["last_name"]

# Exclude already drafted players
drafted_ids = set(draft_picks_live["player_id"])
available_df = df[~df["player_id"].isin(drafted_ids)]

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("🏈 Fantasy Draft Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Player Pool", "📋 Draft Board + Scarcity", "🏈 My Roster"])

# --- TAB 1: Player Pool (keeps QB/RB/WR/TE tables) ---
with tab1:
    st.subheader("Available Players by Position")
    top_left, top_right = st.columns(2)

    with top_left:
        st.markdown("### Quarterbacks")
        st.dataframe(
            available_df[available_df["primary_fantasy_position"] == "QB"]
            .sort_values("proj_overall_2025_rank")
            .head(20),
            use_container_width=True
        )

    with top_right:
        st.markdown("### Running Backs")
        st.dataframe(
            available_df[available_df["primary_fantasy_position"] == "RB"]
            .sort_values("proj_overall_2025_rank")
            .head(20),
            use_container_width=True
        )

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.markdown("### Wide Receivers")
        st.dataframe(
            available_df[available_df["primary_fantasy_position"] == "WR"]
            .sort_values("proj_overall_2025_rank")
            .head(20),
            use_container_width=True
        )

    with bottom_right:
        st.markdown("### Tight Ends")
        st.dataframe(
            available_df[available_df["primary_fantasy_position"] == "TE"]
            .sort_values("proj_overall_2025_rank")
            .head(20),
            use_container_width=True
        )

# --- TAB 2: Draft Board + Scarcity ---
with tab2:
    st.subheader("📋 Draft Board & Positional Scarcity")

    # ADP filter
    max_adp = st.slider("Max ADP to consider for scarcity (lower = earlier picks, higher = deeper pool)",
                        min_value=1, max_value=400, value=200)

    filtered_available = available_df[available_df["proj_adp_half_ppr"] <= max_adp]

    col1, col2 = st.columns([2, 1])  # wider for draft board

    with col1:
        st.markdown("### Draft Board (Live Picks)")
        draft_view = draft_picks_live[["round", "pick_no", "draft_slot", "first_name", "last_name",
                                       "primary_fantasy_position", "team"]].sort_values(["round", "pick_no"])
        st.dataframe(draft_view, use_container_width=True)

    with col2:
        st.markdown("### Remaining Positional Depth")
        scarcity = (
            filtered_available.groupby("primary_fantasy_position")
            .agg(
                available_players=("player_id", "count"),
                avg_proj_points=("proj_calc_pts_half_ppr", "mean")
            )
            .reset_index()
            .sort_values("primary_fantasy_position")
        )
        st.dataframe(scarcity, use_container_width=True)

# --- TAB 3: Roster Grid View ---
with tab3:
    st.subheader("🏈 League Rosters")

    # all teams from draft slots
    teams = draft_picks_live["draft_slot"].unique()
    teams = sorted(teams)

    # Create an empty grid with roster spots as rows, teams as columns
    roster_grid = pd.DataFrame(index=roster_structure, columns=teams)

    # Fill roster grid
    for team in teams:
        team_picks = draft_picks_live[draft_picks_live["draft_slot"] == team]\
                        .sort_values("pick_no")  # order in which players were picked
        for i, slot in enumerate(roster_structure):
            if i < len(team_picks):
                player = team_picks.iloc[i]
                # show Player Name (Pos, Team)
                roster_grid.loc[slot, team] = f"{player['first_name']} {player['last_name']} ({player['primary_fantasy_position']}-{player['team']})"
            else:
                roster_grid.loc[slot, team] = ""  # empty slot

    # Display
    st.dataframe(roster_grid, use_container_width=True)
