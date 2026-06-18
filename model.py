import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier, LGBMRegressor
import sqlite3
import joblib
import os

# ============================================================
# VERSIÓN 2.0: MEJORAS IMPLEMENTADAS
# - XG Variance Calibrated: 1.73 (vs 1.40 old)
# - Advanced Features: 10-match weighted streaks
# - Defensive Consistency Features
# ============================================================

# Cargar perfiles tácticos de StatsBomb (Mundiales 2018+2022)
def load_team_profiles():
    path = 'data/team_profiles.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return {row['team']: row for _, row in df.iterrows()}
    return {}

TEAM_PROFILES = load_team_profiles()

def get_team_profile(team_name):
    """Devuelve stats tácticos de StatsBomb para un equipo"""
    profile = TEAM_PROFILES.get(team_name, None)
    if profile is not None:
        return float(profile['shots_per_match']), float(profile['shot_accuracy']), float(profile['conversion_rate']), float(profile['pressures_per_match'])
    # Valores promedio para equipos sin datos
    return 11.0, 0.35, 0.10, 155.0

# Cargar FIFA Rankings
def load_fifa_rankings():
    path = 'data/fifa_rankings.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.sort_values('date', ascending=False).drop_duplicates('team')
        return {row['team']: float(row['total_points']) for _, row in df.iterrows()}
    return {}

FIFA_RANKINGS = load_fifa_rankings()

AMERICAS = ["Mexico", "USA", "United States", "Canada", "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru", "Ecuador", "Costa Rica", "Panama", "Paraguay", "Venezuela", "Bolivia", "Jamaica", "Haiti"]

COMPETITIVE = ["FIFA World Cup", "FIFA World Cup qualification", "Copa América", "UEFA Euro", 
               "UEFA Euro qualification", "African Cup of Nations", "AFC Asian Cup",
               "CONCACAF Gold Cup", "UEFA Nations League", "CONCACAF Nations League",
               "African Cup of Nations qualification", "AFC Asian Cup qualification",
               "Copa América qualification"]

def get_market_value(team_name, elo):
    values = {
        "England": 1400, "France": 1200, "Brazil": 1100, "Portugal": 1000, 
        "Spain": 950, "Argentina": 900, "Germany": 850, "Italy": 800,
        "Netherlands": 750, "Belgium": 600, "Uruguay": 500, "Colombia": 400,
        "Senegal": 350, "Croatia": 350, "Morocco": 300, "USA": 250, "Mexico": 250,
        "Japan": 200, "South Korea": 180, "Switzerland": 250, "Denmark": 300,
        "Norway": 450, "Ecuador": 150, "Chile": 100, "Peru": 80, "Saudi Arabia": 50, 
        "Qatar": 40, "Iraq": 15
    }
    if team_name in values: return values[team_name]
    return max(5, (elo - 1400) * 0.5)

def calculate_dynamic_elo(df):
    elo_dict = {}
    elo_off_dict = {}
    elo_def_dict = {}
    home_elos, away_elos = [], []
    home_off, away_off = [], []
    home_def, away_def = [], []
    
    for idx, row in df.iterrows():
        t1, t2 = row['home_team'], row['away_team']
        elo1 = elo_dict.get(t1, 1500)
        elo2 = elo_dict.get(t2, 1500)
        
        off1 = elo_off_dict.get(t1, 1500)
        def1 = elo_def_dict.get(t1, 1500)
        off2 = elo_off_dict.get(t2, 1500)
        def2 = elo_def_dict.get(t2, 1500)
        
        home_elos.append(elo1)
        away_elos.append(elo2)
        home_off.append(off1)
        home_def.append(def1)
        away_off.append(off2)
        away_def.append(def2)
        
        if pd.isna(row['home_score']): continue
        
        hg, ag = row['home_score'], row['away_score']
        
        # Win/Loss for base ELO
        w1 = 1 if hg > ag else (0.5 if hg == ag else 0)
        w2 = 1 - w1
        K = 40 if row.get('tournament', '') in COMPETITIVE else 20
        dr = elo1 + 50 - elo2
        we1 = 1 / (10**(-dr/400) + 1)
        we2 = 1 - we1
        elo_dict[t1] = elo1 + K * (w1 - we1)
        elo_dict[t2] = elo2 + K * (w2 - we2)
        
        # Goals for Off/Def ELO
        # Expected goals based on Off vs Def
        dr_off1 = off1 - def2
        we_hg = max(0.1, 1.0 + (dr_off1 / 400.0))
        dr_off2 = off2 - def1
        we_ag = max(0.1, 1.0 + (dr_off2 / 400.0))
        
        K_goal = 20 if row.get('tournament', '') in COMPETITIVE else 10
        
        elo_off_dict[t1] = off1 + K_goal * (hg - we_hg)
        elo_def_dict[t2] = def2 + K_goal * (we_hg - hg)
        
        elo_off_dict[t2] = off2 + K_goal * (ag - we_ag)
        elo_def_dict[t1] = def1 + K_goal * (we_ag - ag)
        
    df['home_elo_dyn'] = home_elos
    df['away_elo_dyn'] = away_elos
    df['home_elo_off'] = home_off
    df['away_elo_off'] = away_off
    df['home_elo_def'] = home_def
    df['away_elo_def'] = away_def
    
    return df, elo_dict, elo_off_dict, elo_def_dict

def add_streaks(df):
    """Calcula streaks mejorados y features avanzados"""
    home = df[['date', 'home_team', 'home_score', 'away_score']].copy()
    home.columns = ['date', 'team', 'gf', 'ga']
    home['points'] = np.where(home['gf'] > home['ga'], 3, np.where(home['gf'] == home['ga'], 1, 0))
    home['gd'] = home['gf'] - home['ga']
    
    away = df[['date', 'away_team', 'away_score', 'home_score']].copy()
    away.columns = ['date', 'team', 'gf', 'ga']
    away['points'] = np.where(away['gf'] > away['ga'], 3, np.where(away['gf'] == away['ga'], 1, 0))
    away['gd'] = away['gf'] - away['ga']
    
    long_df = pd.concat([home, away]).sort_values(['team', 'date'])
    
    # Features 5-match (mantenido por compatibilidad)
    long_df['streak'] = long_df.groupby('team')['points'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long_df['streak'] = long_df['streak'].fillna(0)
    
    long_df['gd_streak'] = long_df.groupby('team')['gd'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long_df['gd_streak'] = long_df['gd_streak'].fillna(0)
    
    long_df['avg_goals'] = long_df.groupby('team')['gf'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    long_df['avg_goals'] = long_df['avg_goals'].fillna(1.0)
    
    long_df['avg_conceded'] = long_df.groupby('team')['ga'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    long_df['avg_conceded'] = long_df['avg_conceded'].fillna(1.0)
    
    # NUEVO: Features avanzados (10-match weighted)
    # Exponential weights: más recientes pesan más
    def weighted_streak_10(series):
        result = []
        for i in range(len(series)):
            if i == 0:
                result.append(0)
            else:
                window = series.iloc[max(0, i-10):i].values
                if len(window) > 0:
                    weights = np.exp(np.linspace(-1, 0, len(window)))
                    weighted = (window * weights).sum()
                    result.append(weighted)
                else:
                    result.append(0)
        return pd.Series(result, index=series.index)
    
    long_df['streak_10_weighted'] = long_df.groupby('team')['points'].transform(weighted_streak_10)
    
    # Defensive consistency (1 / (1 + avg_ga) - values between 0-1)
    long_df['defensive_consistency'] = 1.0 / (1.0 + long_df['avg_conceded'])
    
    # Offensive efficiency (goals per shot chance - historical)
    long_df['offensive_efficiency'] = long_df['avg_goals'] / (long_df['avg_goals'] + long_df['avg_conceded'] + 1)
    
    df = df.merge(long_df[['date', 'team', 'streak', 'gd_streak', 'avg_goals', 'avg_conceded', 'streak_10_weighted', 'defensive_consistency', 'offensive_efficiency']], 
                  left_on=['date', 'home_team'], right_on=['date', 'team'], how='left')
    df.rename(columns={'streak': 'home_streak', 'gd_streak': 'home_gd', 
                        'avg_goals': 'home_avg_gf', 'avg_conceded': 'home_avg_ga',
                        'streak_10_weighted': 'home_streak_10',
                        'defensive_consistency': 'home_def_consistency',
                        'offensive_efficiency': 'home_off_efficiency'}, inplace=True)
    df.drop('team', axis=1, inplace=True)
    
    df = df.merge(long_df[['date', 'team', 'streak', 'gd_streak', 'avg_goals', 'avg_conceded', 'streak_10_weighted', 'defensive_consistency', 'offensive_efficiency']], 
                  left_on=['date', 'away_team'], right_on=['date', 'team'], how='left')
    df.rename(columns={'streak': 'away_streak', 'gd_streak': 'away_gd',
                        'avg_goals': 'away_avg_gf', 'avg_conceded': 'away_avg_ga',
                        'streak_10_weighted': 'away_streak_10',
                        'defensive_consistency': 'away_def_consistency',
                        'offensive_efficiency': 'away_off_efficiency'}, inplace=True)
    df.drop('team', axis=1, inplace=True)
    
    return df.drop_duplicates(subset=['date', 'home_team', 'away_team'])

def add_h2h(df):
    df = df.sort_values('date')
    matchup_history = {}
    h2h_home_wins, h2h_away_wins = [], []
    
    for idx, row in df.iterrows():
        t1, t2 = row['home_team'], row['away_team']
        key = tuple(sorted([t1, t2]))
        history = matchup_history.get(key, [])
        hwins, awins = 0, 0
        for pm in history[-5:]:
            if pm['home'] == t1:
                if pm['hg'] > pm['ag']: hwins += 1
                elif pm['ag'] > pm['hg']: awins += 1
            else:
                if pm['ag'] > pm['hg']: hwins += 1
                elif pm['hg'] > pm['ag']: awins += 1
        h2h_home_wins.append(hwins)
        h2h_away_wins.append(awins)
        if key not in matchup_history: matchup_history[key] = []
        matchup_history[key].append({'home': t1, 'hg': row['home_score'], 'ag': row['away_score']})
        
    df['h2h_home_wins'] = h2h_home_wins
    df['h2h_away_wins'] = h2h_away_wins
    return df

FEATURES = ['home_elo_dyn', 'away_elo_dyn', 'home_elo_off', 'away_elo_off', 'home_elo_def', 'away_elo_def',
            'home_streak', 'away_streak', 'home_mv', 'away_mv', 
            'home_gd', 'away_gd', 'h2h_home_wins', 'h2h_away_wins', 'home_americas', 'away_americas',
            'home_avg_gf', 'home_avg_ga', 'away_avg_gf', 'away_avg_ga', 'is_competitive',
            'home_shots', 'home_accuracy', 'home_conversion', 'home_pressure',
            'away_shots', 'away_accuracy', 'away_conversion', 'away_pressure',
            'home_fifa_pts', 'away_fifa_pts', 'elo_diff', 'mv_ratio',
            'home_streak_10', 'away_streak_10', 'home_def_consistency', 'away_def_consistency',
            'home_off_efficiency', 'away_off_efficiency']

def train_model():
    print("Calculando ELO Dinámico...")
    df = pd.read_csv("data/matches.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df, elo_dict, elo_off_dict, elo_def_dict = calculate_dynamic_elo(df)
    joblib.dump(elo_dict, "database/elo_dict.pkl")
    joblib.dump(elo_off_dict, "database/elo_off_dict.pkl")
    joblib.dump(elo_def_dict, "database/elo_def_dict.pkl")
    
    rankings_df = pd.DataFrame(list(elo_dict.items()), columns=['Team', 'ELO']).sort_values('ELO', ascending=False)
    conn = sqlite3.connect("database/worldcup.db")
    rankings_df.to_sql('Rankings', conn, if_exists='replace', index=False)
    conn.close()
    
    df = df[df['date'].dt.year >= 2010].dropna(subset=['home_score', 'away_score'])
    df = add_streaks(df)
    df = add_h2h(df)
    
    df['home_americas'] = df['home_team'].isin(AMERICAS).astype(int)
    df['away_americas'] = df['away_team'].isin(AMERICAS).astype(int)
    df['is_competitive'] = df['tournament'].isin(COMPETITIVE).astype(int)
    
    df['home_mv'] = df.apply(lambda r: get_market_value(r['home_team'], r['home_elo_dyn']), axis=1)
    df['away_mv'] = df.apply(lambda r: get_market_value(r['away_team'], r['away_elo_dyn']), axis=1)
    
    # StatsBomb tactical features
    df['home_shots'] = df['home_team'].apply(lambda t: get_team_profile(t)[0])
    df['home_accuracy'] = df['home_team'].apply(lambda t: get_team_profile(t)[1])
    df['home_conversion'] = df['home_team'].apply(lambda t: get_team_profile(t)[2])
    df['home_pressure'] = df['home_team'].apply(lambda t: get_team_profile(t)[3])
    df['away_shots'] = df['away_team'].apply(lambda t: get_team_profile(t)[0])
    df['away_accuracy'] = df['away_team'].apply(lambda t: get_team_profile(t)[1])
    df['away_conversion'] = df['away_team'].apply(lambda t: get_team_profile(t)[2])
    df['away_pressure'] = df['away_team'].apply(lambda t: get_team_profile(t)[3])
    
    # FIFA ranking points
    df['home_fifa_pts'] = df['home_team'].apply(lambda t: FIFA_RANKINGS.get(t, 1400))
    df['away_fifa_pts'] = df['away_team'].apply(lambda t: FIFA_RANKINGS.get(t, 1400))
    
    # Derived features
    df['elo_diff'] = df['home_elo_dyn'] - df['away_elo_dyn']
    df['mv_ratio'] = df['home_mv'] / (df['away_mv'] + 1)
    
    X_data = df[FEATURES].fillna(0)
    y_class = np.where(df['home_score'] > df['away_score'], 2, np.where(df['home_score'] == df['away_score'], 1, 0))
    
    # Pesos: recientes + competitivos pesan más
    years_ago = (df['date'].max() - df['date']).dt.days / 365.25
    recency_weight = np.exp(-years_ago * 0.15)
    competitive_weight = np.where(df['tournament'].isin(COMPETITIVE), 1.5, 0.7)
    sample_weights = recency_weight * competitive_weight
              
    # ============================================================
    # ENSEMBLE: 4 CEREBROS (XGBoost + RandomForest + GradientBoosting + LightGBM)
    # + CALIBRACIÓN ISOTÓNICA PARA MEJORAR BRIER SCORE
    # ============================================================
    print("Entrenando Cerebro 1/4: XGBoost (Calibrado)...")
    xgb_base = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric='mlogloss',
        subsample=1.0, n_estimators=200, max_depth=4, learning_rate=0.01, colsample_bytree=0.8, n_jobs=2
    )
    xgb_rs = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=3)
    xgb_rs.fit(X_data, y_class, sample_weight=sample_weights)
    
    print("Entrenando Cerebro 2/4: RandomForest (Calibrado)...")
    rf_base = RandomForestClassifier(
        random_state=42, n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_depth=8, n_jobs=2
    )
    rf_rs = CalibratedClassifierCV(estimator=rf_base, method='isotonic', cv=3)
    rf_rs.fit(X_data, y_class, sample_weight=sample_weights)
    
    print("Entrenando Cerebro 3/4: GradientBoosting (Calibrado)...")
    gb_base = GradientBoostingClassifier(
        random_state=42, subsample=0.9, n_estimators=100, max_depth=3, learning_rate=0.05
    )
    gb_rs = CalibratedClassifierCV(estimator=gb_base, method='isotonic', cv=3)
    gb_rs.fit(X_data, y_class, sample_weight=sample_weights)
    
    print("Entrenando Cerebro 4/4: LightGBM (Calibrado)...")
    lgb_base = LGBMClassifier(
        random_state=42, verbose=-1, n_estimators=150, max_depth=5, learning_rate=0.05, num_leaves=31, subsample=0.9, n_jobs=2
    )
    lgb_rs = CalibratedClassifierCV(estimator=lgb_base, method='isotonic', cv=3)
    lgb_rs.fit(X_data, y_class, sample_weight=sample_weights)
    
    # Guardar los 4 modelos
    ensemble = {
        'xgb': xgb_rs,
        'rf': rf_rs,
        'gb': gb_rs,
        'lgb': lgb_rs
    }
    joblib.dump(ensemble, "database/ensemble_model.pkl")
    joblib.dump(xgb_rs, "database/xgb_model.pkl")
    
    print("Entrenando Regresores de Goles (Ensemble)...")
    hg_xgb = XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.9, n_jobs=2)
    hg_xgb.fit(X_data, df['home_score'], sample_weight=sample_weights)
    
    ag_xgb = XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.9, n_jobs=2)
    ag_xgb.fit(X_data, df['away_score'], sample_weight=sample_weights)
    
    hg_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=2)
    hg_rf.fit(X_data, df['home_score'], sample_weight=sample_weights)
    ag_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=2)
    ag_rf.fit(X_data, df['away_score'], sample_weight=sample_weights)
    
    hg_lgb = LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=2)
    hg_lgb.fit(X_data, df['home_score'], sample_weight=sample_weights)
    ag_lgb = LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=2)
    ag_lgb.fit(X_data, df['away_score'], sample_weight=sample_weights)
    
    goal_ensemble = {
        'hg_xgb': hg_xgb, 'ag_xgb': ag_xgb,
        'hg_rf': hg_rf, 'ag_rf': ag_rf,
        'hg_lgb': hg_lgb, 'ag_lgb': ag_lgb
    }
    joblib.dump(goal_ensemble, "database/goal_ensemble.pkl")
    joblib.dump(hg_xgb, "database/xgb_home_goals.pkl")
    joblib.dump(ag_xgb, "database/xgb_away_goals.pkl")
    
    print("¡Ensemble de 4 cerebros + StatsBomb entrenado exitosamente!")

def get_current_stats(team):
    df = pd.read_csv("data/matches.csv")
    df = df.dropna(subset=['home_score', 'away_score'])
    df['date'] = pd.to_datetime(df['date'])
    tm = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date').tail(5)
    streak, gd, avg_gf, avg_ga = 0, 0, 0, 0
    count = 0
    for _, row in tm.iterrows():
        count += 1
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']: streak += 3
            elif row['home_score'] == row['away_score']: streak += 1
            gd += (row['home_score'] - row['away_score'])
            avg_gf += row['home_score']
            avg_ga += row['away_score']
        else:
            if row['away_score'] > row['home_score']: streak += 3
            elif row['home_score'] == row['away_score']: streak += 1
            gd += (row['away_score'] - row['home_score'])
            avg_gf += row['away_score']
            avg_ga += row['home_score']
    if count > 0:
        avg_gf /= count
        avg_ga /= count
    else:
        avg_gf, avg_ga = 1.0, 1.0
    return int(streak), int(gd), round(avg_gf, 2), round(avg_ga, 2)

def get_current_h2h(t1, t2):
    df = pd.read_csv("data/matches.csv")
    df['date'] = pd.to_datetime(df['date'])
    mask = ((df['home_team'] == t1) & (df['away_team'] == t2)) | ((df['home_team'] == t2) & (df['away_team'] == t1))
    h2h = df[mask].sort_values('date').tail(5)
    w1, w2 = 0, 0
    for _, row in h2h.iterrows():
        if row['home_team'] == t1:
            if row['home_score'] > row['away_score']: w1 += 1
            elif row['away_score'] > row['home_score']: w2 += 1
        else:
            if row['away_score'] > row['home_score']: w1 += 1
            elif row['home_score'] > row['away_score']: w2 += 1
    return w1, w2

def get_elos():
     if not os.path.exists("database/elo_dict.pkl"):
         train_model()
     return joblib.load("database/elo_dict.pkl")

def get_advanced_stats(team):
     """Calcula los 6 nuevos features: streak_10, def_consistency, off_efficiency"""
     df = pd.read_csv("data/matches.csv")
     df = df.dropna(subset=['home_score', 'away_score'])
     df['date'] = pd.to_datetime(df['date'])
     
     # Últimos 10 matches
     tm = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date').tail(10)
     
     # streak_10_weighted (exponential decay)
     points_list = []
     for _, row in tm.iterrows():
         if row['home_team'] == team:
             if row['home_score'] > row['away_score']: points_list.append(3)
             elif row['home_score'] == row['away_score']: points_list.append(1)
             else: points_list.append(0)
         else:
             if row['away_score'] > row['home_score']: points_list.append(3)
             elif row['home_score'] == row['away_score']: points_list.append(1)
             else: points_list.append(0)
     
     if len(points_list) > 0:
         weights = np.exp(np.linspace(-1, 0, len(points_list)))
         streak_10 = float((np.array(points_list) * weights).sum())
     else:
         streak_10 = 0.0
     
     # defensive_consistency y offensive_efficiency
     ga_total = 0.0
     gf_total = 0.0
     ga_count = 0
     gf_count = 0
     
     for _, row in tm.iterrows():
         if row['home_team'] == team:
             ga_total += row['away_score']
             gf_total += row['home_score']
         else:
             ga_total += row['home_score']
             gf_total += row['away_score']
         ga_count += 1
         gf_count += 1
     
     avg_ga = ga_total / ga_count if ga_count > 0 else 1.0
     avg_gf = gf_total / gf_count if gf_count > 0 else 1.0
     
     def_consistency = 1.0 / (1.0 + avg_ga)
     off_efficiency = avg_gf / (avg_gf + avg_ga + 1)
     
     return float(streak_10), float(def_consistency), float(off_efficiency)

def predict_match(team1, team2):
    # Cargar Ensemble
    if os.path.exists("database/ensemble_model.pkl"):
        ensemble = joblib.load("database/ensemble_model.pkl")
    else:
        ensemble = {'xgb': joblib.load("database/xgb_model.pkl")}
    
    elos = joblib.load("database/elo_dict.pkl") if os.path.exists("database/elo_dict.pkl") else {}
    elos_off = joblib.load("database/elo_off_dict.pkl") if os.path.exists("database/elo_off_dict.pkl") else {}
    elos_def = joblib.load("database/elo_def_dict.pkl") if os.path.exists("database/elo_def_dict.pkl") else {}
    
    elo1, elo2 = elos.get(team1, 1500), elos.get(team2, 1500)
    elo_off1, elo_off2 = elos_off.get(team1, 1500), elos_off.get(team2, 1500)
    elo_def1, elo_def2 = elos_def.get(team1, 1500), elos_def.get(team2, 1500)
    
    streak1, gd1, avg_gf1, avg_ga1 = get_current_stats(team1)
    streak2, gd2, avg_gf2, avg_ga2 = get_current_stats(team2)
    h2h1, h2h2 = get_current_h2h(team1, team2)
    mv1, mv2 = get_market_value(team1, elo1), get_market_value(team2, elo2)
    am1, am2 = 1 if team1 in AMERICAS else 0, 1 if team2 in AMERICAS else 0
    is_comp = 1
    
    # StatsBomb tactical features
    s1, a1, c1, p1 = get_team_profile(team1)
    s2, a2, c2, p2 = get_team_profile(team2)
    
    # FIFA rankings
    fifa1 = FIFA_RANKINGS.get(team1, 1400)
    fifa2 = FIFA_RANKINGS.get(team2, 1400)
    
    # Get advanced stats (new features)
    streak10_1, def_cons1, off_eff1 = get_advanced_stats(team1)
    streak10_2, def_cons2, off_eff2 = get_advanced_stats(team2)
    
    X_in = pd.DataFrame([[elo1, elo2, elo_off1, elo_off2, elo_def1, elo_def2, streak1, streak2, mv1, mv2, gd1, gd2, h2h1, h2h2, am1, am2,
                           avg_gf1, avg_ga1, avg_gf2, avg_ga2, is_comp,
                           s1, a1, c1, p1, s2, a2, c2, p2,
                           fifa1, fifa2, elo1-elo2, mv1/(mv2+1),
                           streak10_1, streak10_2, def_cons1, def_cons2, off_eff1, off_eff2]], columns=FEATURES)
    
    # Promediar probabilidades de todos los cerebros del Ensemble
    all_probs = []
    for name, model in ensemble.items():
        probs = model.predict_proba(X_in)[0]
        p_dict = {c: p for c, p in zip(model.classes_, probs)}
        all_probs.append(p_dict)
    
    # Promediar las probabilidades
    avg_p = {}
    for label in [0, 1, 2]:
        avg_p[label] = np.mean([p.get(label, 0) for p in all_probs])
    
    # Goles: promediar XGB + RF + LightGBM regressors
    if os.path.exists("database/goal_ensemble.pkl"):
        ge = joblib.load("database/goal_ensemble.pkl")
        xg1_raw = max(0.1, float(ge['hg_xgb'].predict(X_in)[0]) * 0.33 + float(ge['hg_rf'].predict(X_in)[0]) * 0.33 + float(ge.get('hg_lgb', ge['hg_xgb']).predict(X_in)[0]) * 0.34)
        xg2_raw = max(0.1, float(ge['ag_xgb'].predict(X_in)[0]) * 0.33 + float(ge['ag_rf'].predict(X_in)[0]) * 0.33 + float(ge.get('ag_lgb', ge['ag_xgb']).predict(X_in)[0]) * 0.34)
    else:
        xg1_raw, xg2_raw = 1.0, 1.0
    
    # Calibración Dinámica de xG con ELO Ofensivo y Defensivo
    # xG Equipo = xG Base * (ELO Ofensivo Equipo / ELO Defensivo Rival)
    xg1_dyn = xg1_raw * (elo_off1 / elo_def2)
    xg2_dyn = xg2_raw * (elo_off2 / elo_def1)
    
    # Safety cap: evita xG irreales
    MAX_XG = 4.5
    xg1 = min(xg1_dyn, MAX_XG)
    xg2 = min(xg2_dyn, MAX_XG)
    
    # ============================================================
    # MONTE CARLO REAL (10,000 Simulaciones Bivariate Poisson)
    # ============================================================
    np.random.seed(42) # para reproducibilidad en la misma sesión
    sims = 10000
    
    # Covariacion (lambda3) para aumentar la probabilidad de empates (0-0, 1-1)
    # Partidos competitivos suelen tener mayor correlacion (equipos se anulan)
    lambda3 = 0.25 if is_comp else 0.15
    
    l1 = max(0.01, xg1 - lambda3)
    l2 = max(0.01, xg2 - lambda3)
    
    Z = np.random.poisson(lambda3, sims)
    X1 = np.random.poisson(l1, sims)
    X2 = np.random.poisson(l2, sims)
    
    g1_sims = X1 + Z
    g2_sims = X2 + Z
    
    prob_w1 = np.mean(g1_sims > g2_sims)
    prob_d = np.mean(g1_sims == g2_sims)
    prob_w2 = np.mean(g2_sims > g1_sims)
    
    over_25 = np.mean((g1_sims + g2_sims) > 2.5)
    btts = np.mean((g1_sims > 0) & (g2_sims > 0))
    
    # Top 3 Scenarios directos del Monte Carlo
    # Crear un array de marcadores "g1-g2"
    scores = [f"{g1}-{g2}" for g1, g2 in zip(g1_sims, g2_sims)]
    scores_series = pd.Series(scores)
    top_scores = scores_series.value_counts(normalize=True).head(3)
    
    top_3_scenarios = []
    for score, prob in top_scores.items():
        top_3_scenarios.append((score, round(prob * 100, 2)))
        
    official_score = top_3_scenarios[0][0]
    official_prob = top_3_scenarios[0][1]
    
    # Posesión esperada
    elo_diff = elo1 - elo2
    poss1 = 50.0 + (elo_diff / 30.0)
    poss1 = max(30.0, min(70.0, poss1))
    poss2 = 100.0 - poss1

    return {
        "team1_win": round(float(prob_w1 * 100), 2),
        "draw": round(float(prob_d * 100), 2),
        "team2_win": round(float(prob_w2 * 100), 2),
        "over_25": round(float(over_25 * 100), 2),
        "btts": round(float(btts * 100), 2),
        "elo1": round(float(elo1), 1), "elo2": round(float(elo2), 1),
        "elo_off1": round(float(elo_off1), 1), "elo_off2": round(float(elo_off2), 1),
        "elo_def1": round(float(elo_def1), 1), "elo_def2": round(float(elo_def2), 1),
        "streak1": int(streak1), "streak2": int(streak2),
        "mv1": round(float(mv1), 1), "mv2": round(float(mv2), 1),
        "xg1": round(float(xg1), 2), "xg2": round(float(xg2), 2),
        "poss1": round(float(poss1), 1), "poss2": round(float(poss2), 1),
        "gd1": int(gd1), "gd2": int(gd2),
        "h2h1": int(h2h1), "h2h2": int(h2h2),
        "am1": int(am1), "am2": int(am2),
        "avg_gf1": avg_gf1, "avg_ga1": avg_ga1,
        "avg_gf2": avg_gf2, "avg_ga2": avg_ga2,
        "official_score": official_score,
        "official_prob": official_prob,
        "top_3_scenarios": top_3_scenarios
    }

if __name__ == "__main__":
    train_model()
