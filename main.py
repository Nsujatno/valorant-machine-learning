import pandas as pd
import numpy as np
import re

maps_df = pd.read_csv("detailed_matches_maps.csv")
player_stats_df = pd.read_csv("detailed_matches_player_stats.csv")
performance_df = pd.read_csv("performance_data.csv")
economy_df = pd.read_csv("economy_data.csv")

team_name_mapping = {
    'PRX': 'Paper Rex',
    'XLG': 'Xi Lai Gaming',
    'GX': 'GIANTX',
    'SEN': 'Sentinels',
    'NRG': 'NRG',
    'EDG': 'EDward Gaming',
    'TL': 'Team Liquid',
    'DRX': 'DRX',
    'DRG': 'Dragon Ranger Gaming',
    'T1': 'T1',
    'G2': 'G2 Esports',
    'TH': 'Team Heretics',
    'BLG': 'Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)',
    'MIBR': 'MIBR',
    'RRQ': 'Rex Regum Qeon',
    'FNC': 'FNATIC'
}
economy_df['Team'] = economy_df['Team'].map(team_name_mapping)
performance_df['Team'] = performance_df['Team'].map(team_name_mapping)

# Initial data exploration
print(maps_df.info())
print(maps_df.head())
print(maps_df.shape)
print(maps_df.columns)
print(maps_df.isnull().sum())

print(maps_df["winner"].value_counts())
print(maps_df["map_name"].value_counts())

print(maps_df.loc[0])
print(maps_df.loc[1])
# 88 maps played in the dataset, but 34 matches played
print(maps_df["match_id"].nunique())
# average duration of maps
duration = maps_df["duration"]
is_ms_format = (duration.str.count(":") == 1)
standardized_duration = np.where(is_ms_format, "00:" + duration, duration) 
maps_df["duration"] = pd.to_timedelta(standardized_duration)
print(maps_df.info())
maps_df["duration_in_seconds"] = maps_df["duration"].dt.total_seconds()
print(maps_df.loc[10])
print(maps_df.loc[0])
average_duration = maps_df["duration_in_seconds"].mean()
print(f"average duration in seconds: {average_duration}")

# 5 features most predictive of map outcome
# kda, fk/fd differential (the team with more fk more likely to win), meta agent comp, clutch rate (idk how to get this per map basis though), acs

def calculate_team_stats_for_map(player_stats_df, match_id, map_name):
    map_data = player_stats_df[
        (player_stats_df["stat_type"] == "map") &
        (player_stats_df["match_id"] == match_id) &
        (player_stats_df["map_name"] == map_name)
    ]
    
    team_acs = map_data.groupby("player_team")["acs"].mean()
    
    team_stats = map_data.groupby("player_team").agg({
        "k": "sum",
        "d": "sum",
        "a": "sum",
        "fk": "sum",
        "fd": "sum",
        "acs": "mean"
    })
    
    # calculate KDA ratio: (K + A) / D
    team_stats["kda"] = np.where(
        team_stats["d"] != 0,
        (team_stats["k"] + team_stats["a"])/team_stats["d"],
        team_stats["k"] + team_stats["a"]
    )
    
    # calculate FK/FD differential
    team_stats["fkfd_diff"] = team_stats["fk"] - team_stats["fd"]
    
    return team_stats

print(calculate_team_stats_for_map(player_stats_df, 542269, "Corrode"))

def parse_rounds_won(round_string):
    # Parses the round string from 5(3) into just 3
    result = re.split("[()]", round_string)
    filtered_string = []
    for s in result:
        if s:
            filtered_string.append(s)
    return filtered_string[len(filtered_string)-1]

# Apply to all economy columns
economy_df["eco_won"] = pd.to_numeric(economy_df["Eco (won)"].apply(parse_rounds_won))
economy_df["semi_eco_won"] = pd.to_numeric(economy_df["Semi-eco (won)"].apply(parse_rounds_won))
economy_df["total_won"] = economy_df["eco_won"] + economy_df["semi_eco_won"]

def build_features_for_map(match_id, map_name):
    """
    Build ALL features for a single map
    Returns: DataFrame with one row per team
    """
    features = pd.DataFrame()
    
    # 1. Basic team stats (ACS, KDA, FK/FD) - YOU ALREADY HAVE THIS!
    team_stats = calculate_team_stats_for_map(player_stats_df, match_id, map_name)

    # 2. Economy features
    economy_map_data = economy_df[
        (economy_df["match_id"] == match_id) &
        (economy_df["map"] == map_name)
    ]

    economy_features = economy_map_data.set_index('Team')[['eco_won', 'semi_eco_won']]
    performance_map_data = performance_df[
        (performance_df["Match ID"] == match_id) &
        (performance_df["Map"] == map_name)
    ].copy()

    performance_map_data["individual_clutch_score"] = (
        1 * performance_map_data["1v1"] +
        2 * performance_map_data["1v2"] +
        3 * performance_map_data["1v3"] +
        4 * performance_map_data["1v4"] +
        5 * performance_map_data["1v5"]
    )

    clutch_features = performance_map_data.groupby("Team")["individual_clutch_score"].sum()

    features = team_stats.copy()
    features = features.join(economy_features, how='left')
    features = features.join(clutch_features.rename('clutch_score'), how='left')
    
    # 4. Get the actual winner (our TARGET variable!)
    winner = maps_df[
        (maps_df['match_id'] == match_id) & 
        (maps_df['map_name'] == map_name)
    ]['winner'].values[0]
    
    features['won'] = features.index == winner
    features['won'] = features['won'].astype(int)
    return features

# Test on one map
test_features = build_features_for_map(542195, 'Bind')
print(test_features)

# test on all maps
all_features = []

for idx, row in maps_df.iterrows():
    match_id = row['match_id']
    map_name = row['map_name']
    
    try:
        features = build_features_for_map(match_id, map_name)
        all_features.append(features)
    except Exception as e:
        print(f"Error on {match_id}, {map_name}: {e}")
        continue

# Combine all maps into one dataset
full_dataset = pd.concat(all_features, ignore_index=False)

# save csv
full_dataset.to_csv('training_data.csv', index=True)
print("Saved to training_data.csv")

print("Average stats for winners vs losers:")
print(full_dataset.groupby('won')[['acs', 'kda', 'fkfd_diff', 'clutch_score']].mean())