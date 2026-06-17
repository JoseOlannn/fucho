import pandas as pd
import numpy as np
import joblib
import os

AMERICAS = ["Mexico", "USA", "United States", "Canada", "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru", "Ecuador", "Costa Rica", "Panama", "Paraguay", "Venezuela", "Bolivia", "Jamaica", "Haiti"]

# Refactored to use centralized predict_match

def run_monte_carlo(n_simulations=10000):
    df_cal = pd.read_csv("data/schedule.csv")
    
    # 1. Puntos Finalizados
    points_dict = {}
    for idx, row in df_cal[df_cal['Estado'] == 'Fin'].iterrows():
        t1, t2 = row['Local'], row['Visitante']
        points_dict[t1] = points_dict.get(t1, 0)
        points_dict[t2] = points_dict.get(t2, 0)
        
        g1, g2 = row['GolesLocal'], row['GolesVisitante']
        if g1 > g2: points_dict[t1] += 3
        elif g1 == g2:
            points_dict[t1] += 1
            points_dict[t2] += 1
        else: points_dict[t2] += 3
        
    # 2. Mapeo de grupos
    groups_mapping = {}
    for group in df_cal['Grupo'].unique():
        group_rows = df_cal[df_cal['Grupo'] == group]
        teams = set(group_rows['Local']).union(set(group_rows['Visitante']))
        groups_mapping[group] = list(teams)
        
    # 3. Predicción BATCH ultra-rápida (Aquí estaba el cuello de botella)
    pending = df_cal[df_cal['Estado'] == 'Pendiente'].copy()
    pending_probs = []
    
    if len(pending) > 0:
        from model import predict_match
        for idx, row in pending.iterrows():
            t1, t2 = row['Local'], row['Visitante']
            try:
                res = predict_match(t1, t2)
                # Convertir porcentajes (ej. 60.5) a probabilidades decimales (0.605)
                p1 = res['team1_win'] / 100.0
                pd_prob = res['draw'] / 100.0
                pending_probs.append({'t1': t1, 't2': t2, 'p1': p1, 'pd': pd_prob})
            except Exception as e:
                # Si falla, asignar 33% a cada uno
                pending_probs.append({'t1': t1, 't2': t2, 'p1': 0.33, 'pd': 0.33})
            
    # 4. Monte Carlo Loop
    group_winners = {team: 0 for team in set(df_cal['Local']).union(set(df_cal['Visitante']))}
    
    for i in range(n_simulations):
        sim_points = points_dict.copy()
        
        for p in pending_probs:
            roll = np.random.rand()
            t1, t2 = p['t1'], p['t2']
            if roll < p['p1']: 
                sim_points[t1] = sim_points.get(t1, 0) + 3
            elif roll < p['p1'] + p['pd']:
                sim_points[t1] = sim_points.get(t1, 0) + 1
                sim_points[t2] = sim_points.get(t2, 0) + 1
            else:
                sim_points[t2] = sim_points.get(t2, 0) + 3
                
        for group, teams in groups_mapping.items():
            best_team = None
            max_p = -1
            for t in teams:
                pts = sim_points.get(t, 0)
                if pts > max_p:
                    max_p = pts
                    best_team = t
            if best_team in group_winners:
                group_winners[best_team] += 1
                
    res = []
    for team, wins in group_winners.items():
        if wins > 0:
            res.append({"Equipo": team, "Probabilidad 1er Lugar Grupo": (wins / n_simulations) * 100})
            
    return pd.DataFrame(res).sort_values("Probabilidad 1er Lugar Grupo", ascending=False).reset_index(drop=True)
