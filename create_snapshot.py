import pandas as pd
from model import train_model

def create_snapshot():
    matches = pd.read_csv("data/matches.csv")
    
    # Lógica robusta: Excluir TODO partido que sea del Mundial 2026
    matches['date_dt'] = pd.to_datetime(matches['date'], errors='coerce')
    
    mask = ~((matches['date_dt'].dt.year >= 2026) & (matches['tournament'] == 'FIFA World Cup'))
    snapshot_df = matches[mask].copy()
    
    snapshot_df = snapshot_df.drop(columns=['date_dt'])
    
    snapshot_path = "data/matches_pretournament_snapshot.csv"
    snapshot_df.to_csv(snapshot_path, index=False)
    
    print(f"Snapshot data saved: {snapshot_path} ({len(snapshot_df)} rows)")
    
    train_model(data_path=snapshot_path, output_prefix="snapshot_")

if __name__ == '__main__':
    create_snapshot()
