import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from model import predict_match


def parse_schedule_date(fecha_str, default_year=2026):
    """Convierte fechas tipo '11/6' del calendario a formato ISO '2026-06-11'."""
    text = str(fecha_str).strip()
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            return f"{default_year:04d}-{int(parts[1]):02d}-{int(parts[0]):02d}"
    return str(pd.to_datetime(text).date())


def evaluate_brier_score():
    print("Iniciando evaluación de calibración (Backtesting estricto sin data leakage)...")
    try:
        df_fin = pd.read_csv("data/schedule.csv")
        df_fin = df_fin[df_fin["Estado"] == "Fin"]
        if len(df_fin) == 0:
            print("No hay partidos finalizados para hacer backtesting.")
            return

        y_true_bin, y_prob_bin, y_true_mc, y_prob_mc, brier_rows = [], [], [], [], []
        winner_hits, exact_hits = 0, 0

        for _, row in df_fin.iterrows():
            t1, t2 = row["Local"], row["Visitante"]
            rg1, rg2 = int(row["GolesLocal"]), int(row["GolesVisitante"])
            try:
                res = predict_match(t1, t2, as_of_date=parse_schedule_date(row["Fecha"]))
                p1, pd_p, p2 = float(res["team1_win"]) / 100.0, float(res["draw"]) / 100.0, float(res["team2_win"]) / 100.0
                tot = p1 + pd_p + p2
                if tot > 0:
                    p1, pd_p, p2 = p1 / tot, pd_p / tot, p2 / tot

                actual_mc = 2 if rg1 > rg2 else (1 if rg1 == rg2 else 0)
                y1, yd, y2 = (1.0, 0.0, 0.0) if rg1 > rg2 else ((0.0, 1.0, 0.0) if rg1 == rg2 else (0.0, 0.0, 1.0))
                y_true_bin.append(1 if rg1 > rg2 else 0)
                y_prob_bin.append(p1)
                y_true_mc.append(actual_mc)
                y_prob_mc.append([p2, pd_p, p1])
                brier_rows.append(((p1 - y1) ** 2 + (pd_p - yd) ** 2 + (p2 - y2) ** 2) / 3.0)

                pg1, pg2 = map(int, res["official_score"].split("-"))
                winner_hits += int((rg1 > rg2 and pg1 > pg2) or (rg1 < rg2 and pg1 < pg2) or (rg1 == rg2 and pg1 == pg2))
                exact_hits += int(rg1 == pg1 and rg2 == pg2)
            except Exception as e:
                print(f"Error evaluando {t1} vs {t2}: {e}")

        if y_true_mc:
            n = len(y_true_mc)
            brier_mc = float(np.mean(brier_rows))
            print(f"Partidos evaluados: {n}")
            print(f"Acierto de Ganador (1X2): {(winner_hits / n) * 100:.1f}% ({winner_hits}/{n})")
            print(f"Acierto Marcador Exacto:  {(exact_hits / n) * 100:.1f}% ({exact_hits}/{n})")
            print(f"Brier Score Multiclase (1X2): {brier_mc:.4f} (Objetivo: < 0.22)")
            print(f"Brier Score Binario (Local):  {brier_score_loss(y_true_bin, y_prob_bin):.4f} (Objetivo: < 0.20)")
            print(f"Log Loss Multiclase (1X2):    {log_loss(y_true_mc, y_prob_mc, labels=[0, 1, 2]):.4f}")
            print("Calibración sólida y confiable sin data leakage." if brier_mc < 0.22 else "Calibración mejorable.")
    except Exception as e:
        print(f"Error en backtesting: {e}")


if __name__ == "__main__":
    evaluate_brier_score()
