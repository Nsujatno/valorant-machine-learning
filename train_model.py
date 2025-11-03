import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

maps_df = pd.read_csv("detailed_matches_maps.csv")
player_stats_df = pd.read_csv("detailed_matches_player_stats.csv")
performance_df = pd.read_csv("performance_data.csv")\

team_name_mapping = {
    "PRX": "Paper Rex",
    "XLG": "Xi Lai Gaming",
    "GX": "GIANTX",
    "SEN": "Sentinels",
    "NRG": "NRG",
    "EDG": "EDward Gaming",
    "TL": "Team Liquid",
    "DRX": "DRX",
    "DRG": "Dragon Ranger Gaming",
    "T1": "T1",
    "G2": "G2 Esports",
    "TH": "Team Heretics",
    "BLG": "Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)",
    "MIBR": "MIBR",
    "RRQ": "Rex Regum Qeon",
    "FNC": "FNATIC"
}
performance_df["Team"] = performance_df["Team"].map(team_name_mapping)

def calculate_player_historical_stats(player_stats_df):
    player_averages = player_stats_df[player_stats_df['stat_type'] == 'map'].groupby(['player_name', 'player_id']).agg({
        'acs': 'mean',
        'kd_diff': 'mean',
        'fk_fd_diff': 'mean',
        'rating': 'mean',
        'k': 'mean',
        'd': 'mean',
        'a': 'mean'
    }).reset_index()

    player_averages['kd_ratio'] = np.where(
        player_averages['d'] != 0,
        player_averages['k'] / player_averages['d'],
        player_averages['k']
    )
    
    return player_averages

# Calculate historical averages
player_history = calculate_player_historical_stats(player_stats_df)
print(player_history.head(10))

def build_prematch_features_for_map(match_id, map_name, player_history):
    match_players = player_stats_df[
        (player_stats_df['match_id'] == match_id) &
        (player_stats_df['map_name'] == map_name) &
        (player_stats_df['stat_type'] == 'map')
    ][['player_id', 'player_name', 'player_team']].drop_duplicates()

    match_with_history = match_players.merge(
        player_history, 
        on=['player_id', 'player_name'], 
        how='left'
    )

    team_features = match_with_history.groupby('player_team').agg({
        'acs': 'mean',
        'kd_diff': 'mean',
        'fk_fd_diff': 'mean',
        'rating': 'mean',
        'k': 'mean',
        'd': 'mean',
        'a': 'mean'
    })

    winner = maps_df[
        (maps_df['match_id'] == match_id) &
        (maps_df['map_name'] == map_name)
    ]['winner'].values[0]
    
    team_features['won'] = (team_features.index == winner).astype(int)
    
    return team_features

all_prematch_features = []
failed_maps = []

for idx, row in maps_df.iterrows():
    match_id = row['match_id']
    map_name = row['map_name']
    
    try:
        features = build_prematch_features_for_map(match_id, map_name, player_history)
        all_prematch_features.append(features)
    except Exception as e:
        print(f"❌ Error on {match_id}, {map_name}: {e}")
        failed_maps.append((match_id, map_name))
        continue

# Combine into dataset
prematch_dataset = pd.concat(all_prematch_features, ignore_index=False)

# Fill any NaN with column mean
prematch_dataset = prematch_dataset.fillna(prematch_dataset.mean())

prematch_dataset.to_csv('prematch_training_data.csv')

prematch_dataset = pd.read_csv("prematch_training_data.csv")
X = prematch_dataset.drop(columns=["won", "player_team"])
y = prematch_dataset['won']

# train, validate, test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_train_pred = lr_model.predict(X_train_scaled)
y_val_pred = lr_model.predict(X_val_scaled)

print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

print("Confusion Matrix (Validation):")
print(confusion_matrix(y_val, y_val_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', ascending=False)

print("Feature Importance:")
print(feature_importance)