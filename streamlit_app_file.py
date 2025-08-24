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
    return df[cols]


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
    return df


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

    return df[keep_cols].sort_values(by='stats_calc_pts_half_ppr', ascending=False)


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

    return df[keep_cols].sort_values(by='proj_calc_pts_half_ppr', ascending=False)


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

    return df


st.title("Fantasy Football Projections")
player_info = get_all_players()
season_stats = get_season_stats(2024)
season_projections = get_season_projections(2025)
full_player_data = build_full_player_df(player_info, season_stats, season_projections)