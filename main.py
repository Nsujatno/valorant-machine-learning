import pandas as pd
import numpy as np

# Initial data exploration
maps_df = pd.read_csv("detailed_matches_maps.csv")
player_stats_df = pd.read_csv("detailed_matches_player_stats.csv")
performance_df = pd.read_csv("performance_data.csv")
print(maps_df.info())
print(maps_df.head())
print(maps_df.shape)
print(maps_df.columns)
print(maps_df.isnull().sum())

print(maps_df['winner'].value_counts())
print(maps_df['map_name'].value_counts())

print(maps_df.loc[0])
print(maps_df.loc[1])
# 88 maps played in the dataset, but 34 matches played
print(maps_df["match_id"].nunique())
# average duration of maps
duration = maps_df['duration']
is_ms_format = (duration.str.count(':') == 1)
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
    
    team_stats = map_data.groupby('player_team').agg({
        'k': 'sum',
        'd': 'sum',
        'a': 'sum',
        'fk': 'sum',
        'fd': 'sum',
        'acs': 'mean'
    })
    
    # calculate KDA ratio: (K + A) / D
    team_stats['kda'] = np.where(
        team_stats["d"] != 0,
        (team_stats["k"] + team_stats["a"])/team_stats["d"],
        team_stats["k"] + team_stats["a"]
    )
    
    # calculate FK/FD differential
    team_stats['fkfd_diff'] = team_stats["fk"] - team_stats["fd"]
    
    return team_stats

print(calculate_team_stats_for_map(player_stats_df, 542269, "Corrode"))

