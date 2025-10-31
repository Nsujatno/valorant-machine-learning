import pandas as pd
import numpy as np

# Initial data exploration
df_maps = pd.read_csv("detailed_matches_maps.csv")
print(df_maps.info())
print(df_maps.head())
print(df_maps.shape)
print(df_maps.columns)
print(df_maps.isnull().sum())

print(df_maps['winner'].value_counts())

print(df_maps.loc[0])
print(df_maps.loc[1])
# 88 maps played in the dataset, but 34 matches played
print(df_maps["match_id"].nunique())
# average duration of maps
duration = df_maps['duration']
is_ms_format = (duration.str.count(':') == 1)
standardized_duration = np.where(is_ms_format, "00:" + duration, duration) 
df_maps["duration"] = pd.to_timedelta(standardized_duration)
print(df_maps.info())
df_maps["duration_in_seconds"] = df_maps["duration"].dt.total_seconds()
print(df_maps.loc[10])
print(df_maps.loc[0])
average_duration = df_maps["duration_in_seconds"].mean()
print(f"average duration in seconds: {average_duration}")

# 5 features most predictive of map outcome
# kda, fk/fd differential (the team with more fk more likely to win), meta agent comp, clutch rate (idk how to get this per map basis though), acs