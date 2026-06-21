import joblib
import pandas as pd

def print_importances():
    try:
        xgb = joblib.load("database/snapshot_xgb_model.pkl")
        if hasattr(xgb, "calibrated_classifiers_"):
            base_xgb = xgb.calibrated_classifiers_[0].estimator
        else:
            base_xgb = xgb.estimator
            
        importances = base_xgb.feature_importances_
        
        FEATURES = ['home_elo_dyn', 'away_elo_dyn', 'home_elo_off', 'away_elo_off', 'home_elo_def', 'away_elo_def',
                'home_streak', 'away_streak', 'home_mv', 'away_mv', 
                'home_gd', 'away_gd', 'h2h_home_wins', 'h2h_away_wins', 'home_americas', 'away_americas',
                'home_avg_gf', 'home_avg_ga', 'away_avg_gf', 'away_avg_ga', 'is_competitive',
                'home_shots', 'home_accuracy', 'home_conversion', 'home_pressure',
                'away_shots', 'away_accuracy', 'away_conversion', 'away_pressure',
                'home_fifa_pts', 'away_fifa_pts', 'elo_diff', 'mv_ratio',
                'home_streak_10', 'away_streak_10', 'home_def_consistency', 'away_def_consistency',
                'home_off_efficiency', 'away_off_efficiency']
                
        fi = pd.DataFrame({'feature': FEATURES, 'importance': importances})
        fi = fi.sort_values('importance', ascending=False)
        print("Feature Importances:")
        print(fi)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    print_importances()
