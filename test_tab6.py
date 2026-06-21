import pandas as pd
from model import predict_match

def test_tab6():
    df_cal = pd.read_csv("data/schedule.csv")
    df_fin = df_cal[df_cal["Estado"] == "Fin"]
    
    aciertos_ganador = 0
    aciertos_exacto = 0
    brier_sum = 0.0
    total = len(df_fin)
    
    for _, row in df_fin.iterrows():
        t1, t2 = row['Local'], row['Visitante']
        real_g1, real_g2 = int(row['GolesLocal']), int(row['GolesVisitante'])
        
        try:
            r = predict_match(t1, t2, use_snapshot=True)
            xg1, xg2 = r['xg1'], r['xg2']
            
            pred_g1, pred_g2 = map(int, r['official_score'].split('-'))
            pred_marcador = r['official_score']
            
            if real_g1 > real_g2: real_res = 'L'
            elif real_g1 < real_g2: real_res = 'V'
            else: real_res = 'E'
            
            if pred_g1 > pred_g2: pred_res = 'L'
            elif pred_g1 < pred_g2: pred_res = 'V'
            else: pred_res = 'E'
            es_ganador = (real_res == pred_res)
            es_exacto = (real_g1 == pred_g1 and real_g2 == pred_g2)
            
            if es_ganador: aciertos_ganador += 1
            if es_exacto: aciertos_exacto += 1
            
            prob_l, prob_e, prob_v = r['team1_win']/100.0, r['draw']/100.0, r['team2_win']/100.0
            true_l, true_e, true_v = (1.0 if real_res == 'L' else 0.0), (1.0 if real_res == 'E' else 0.0), (1.0 if real_res == 'V' else 0.0)
            brier_sum += (prob_l - true_l)**2 + (prob_e - true_e)**2 + (prob_v - true_v)**2
            
        except Exception as e:
            pass
            
    pct_ganador = (aciertos_ganador / total) * 100
    pct_exacto = (aciertos_exacto / total) * 100
    brier_score = brier_sum / total
    
    print(f"Total Evaluados: {total}")
    print(f"Brier Score: {brier_score:.3f}")
    print(f"Ganador: {pct_ganador:.1f}% ({aciertos_ganador}/{total})")
    print(f"Exacto: {pct_exacto:.1f}% ({aciertos_exacto}/{total})")

if __name__ == '__main__':
    test_tab6()
