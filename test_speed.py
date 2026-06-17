import pandas as pd
import time
start = time.time()
matches_df = pd.read_csv("data/matches.csv").dropna(subset=['home_score', 'away_score'])
matches_df['date'] = pd.to_datetime(matches_df['date'])
matches_df = matches_df.tail(3000)
for i in range(56):
    tm1 = matches_df[(matches_df['home_team'] == 'Mexico') | (matches_df['away_team'] == 'Mexico')].tail(5)
    tm2 = matches_df[(matches_df['home_team'] == 'Brazil') | (matches_df['away_team'] == 'Brazil')].tail(5)
    h2h_df = matches_df[((matches_df['home_team'] == 'Mexico') & (matches_df['away_team'] == 'Brazil')) | ((matches_df['home_team'] == 'Brazil') & (matches_df['away_team'] == 'Mexico'))].tail(5)
print(f"Time: {time.time() - start}")
