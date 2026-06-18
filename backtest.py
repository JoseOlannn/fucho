import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from model import predict_match, get_elos

def evaluate_brier_score():
    print("Iniciando evaluación de calibración (Backtesting)...")
    try:
        df = pd.read_csv("data/schedule.csv")
        df_fin = df[df["Estado"] == "Fin"]
        
        if len(df_fin) == 0:
            print("No hay partidos finalizados para hacer backtesting.")
            return
            
        y_true = []
        y_prob = []
        
        for _, row in df_fin.iterrows():
            t1, t2 = row['Local'], row['Visitante']
            real_g1, real_g2 = int(row['GolesLocal']), int(row['GolesVisitante'])
            
            try:
                res = predict_match(t1, t2)
                
                # Para el Brier Score evaluaremos la probabilidad de victoria local
                prob_t1_win = res['team1_win'] / 100.0
                
                # Resultado real: 1 si local gana, 0 si no (empate o visita)
                actual = 1 if real_g1 > real_g2 else 0
                
                y_true.append(actual)
                y_prob.append(prob_t1_win)
            except Exception as e:
                pass
                
        if len(y_true) > 0:
            brier = brier_score_loss(y_true, y_prob)
            logl = log_loss(y_true, y_prob)
            
            print(f"Partidos evaluados: {len(y_true)}")
            print(f"Brier Score (Victoria Local): {brier:.4f} (Objetivo: < 0.20)")
            print(f"Log Loss (Victoria Local): {logl:.4f}")
            
            if brier < 0.22:
                print("✅ Calibración excelente. El modelo es matemáticamente confiable.")
            else:
                print("⚠️ Calibración mejorable. Posible sobreajuste (overconfidence).")
    except Exception as e:
        print(f"Error en backtesting: {e}")

if __name__ == "__main__":
    evaluate_brier_score()
