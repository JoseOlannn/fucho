import numpy as np
import pandas as pd

from model import _sample_overdispersed_bivariate_poisson, predict_match


def run_monte_carlo(n_simulations=10000, seed=42):
    df_cal = pd.read_csv("data/schedule.csv")
    df_cal = df_cal[df_cal["Grupo"].astype(str).str.startswith("Grupo")].copy()
    rng = np.random.default_rng(seed)

    all_teams = sorted(set(df_cal["Local"]).union(set(df_cal["Visitante"])))
    team_to_group, groups_mapping = {}, {}
    for group in df_cal["Grupo"].unique():
        g_rows = df_cal[df_cal["Grupo"] == group]
        teams = sorted(set(g_rows["Local"]).union(set(g_rows["Visitante"])))
        groups_mapping[group] = teams
        for t in teams:
            team_to_group[t] = group

    pending = df_cal[df_cal["Estado"] == "Pendiente"]
    full_replay = len(pending) == 0
    to_simulate = df_cal if full_replay else pending
    as_of = "2026-06-11" if full_replay else None

    base_pts = {t: 0 for t in all_teams}
    base_gd = {t: 0 for t in all_teams}
    base_gf = {t: 0 for t in all_teams}
    if not full_replay:
        for _, row in df_cal[df_cal["Estado"] == "Fin"].iterrows():
            t1, t2 = row["Local"], row["Visitante"]
            g1, g2 = int(row["GolesLocal"]), int(row["GolesVisitante"])
            base_gf[t1] += g1; base_gf[t2] += g2
            base_gd[t1] += g1 - g2; base_gd[t2] += g2 - g1
            if g1 > g2:
                base_pts[t1] += 3
            elif g1 == g2:
                base_pts[t1] += 1; base_pts[t2] += 1
            else:
                base_pts[t2] += 3

    sim_pts = {t: np.full(n_simulations, base_pts[t], dtype=np.int32) for t in all_teams}
    sim_gd = {t: np.full(n_simulations, base_gd[t], dtype=np.int32) for t in all_teams}
    sim_gf = {t: np.full(n_simulations, base_gf[t], dtype=np.int32) for t in all_teams}

    for _, row in to_simulate.iterrows():
        t1, t2 = row["Local"], row["Visitante"]
        try:
            res = predict_match(t1, t2, as_of_date=as_of)
            xg1, xg2 = float(res["xg1"]), float(res["xg2"])
            p_w1, p_d, p_w2 = res["team1_win"] / 100.0, res["draw"] / 100.0, res["team2_win"] / 100.0
        except Exception:
            xg1, xg2, p_w1, p_d, p_w2 = 1.2, 1.2, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

        g1_raw, g2_raw = _sample_overdispersed_bivariate_poisson(rng, xg1, xg2, is_comp=1, sims=n_simulations)
        mc_w1 = max(float(np.mean(g1_raw > g2_raw)), 1e-6)
        mc_d = max(float(np.mean(g1_raw == g2_raw)), 1e-6)
        mc_w2 = max(float(np.mean(g2_raw > g1_raw)), 1e-6)

        w = np.where(g1_raw > g2_raw, p_w1 / mc_w1, np.where(g1_raw == g2_raw, p_d / mc_d, p_w2 / mc_w2))
        w_sum = w.sum()
        if w_sum > 0:
            idx = rng.choice(n_simulations, size=n_simulations, replace=True, p=w / w_sum)
            g1_sims, g2_sims = g1_raw[idx], g2_raw[idx]
        else:
            g1_sims, g2_sims = g1_raw, g2_raw

        sim_gf[t1] += g1_sims; sim_gf[t2] += g2_sims
        sim_gd[t1] += g1_sims - g2_sims; sim_gd[t2] += g2_sims - g1_sims
        sim_pts[t1] += np.where(g1_sims > g2_sims, 3, np.where(g1_sims == g2_sims, 1, 0))
        sim_pts[t2] += np.where(g2_sims > g1_sims, 3, np.where(g1_sims == g2_sims, 1, 0))

    first_counts = {t: 0 for t in all_teams}
    top2_counts = {t: 0 for t in all_teams}
    for _, teams in groups_mapping.items():
        pts_m = np.column_stack([sim_pts[t] for t in teams]).astype(np.float64)
        gd_m = np.column_stack([sim_gd[t] for t in teams]).astype(np.float64)
        gf_m = np.column_stack([sim_gf[t] for t in teams]).astype(np.float64)
        order = np.argsort(-(pts_m * 1_000_000.0 + gd_m * 1_000.0 + gf_m + rng.random((n_simulations, len(teams)))), axis=1)
        first_i, second_i = order[:, 0], (order[:, 1] if len(teams) > 1 else None)
        for pos, name in enumerate(teams):
            first_counts[name] += int(np.sum(first_i == pos))
            top2_counts[name] += int(np.sum((first_i == pos) | (second_i == pos))) if second_i is not None else first_counts[name]

    rows = [
        {
            "Equipo": t,
            "Grupo": team_to_group.get(t, "-"),
            "Puntos Esperados": round(float(np.mean(sim_pts[t])), 2),
            "Probabilidad 1er Lugar Grupo": round((first_counts[t] / n_simulations) * 100.0, 2),
            "Probabilidad Top 2 Grupo": round((top2_counts[t] / n_simulations) * 100.0, 2),
        }
        for t in all_teams
    ]
    return pd.DataFrame(rows).sort_values(
        ["Probabilidad 1er Lugar Grupo", "Probabilidad Top 2 Grupo", "Puntos Esperados"], ascending=False
    ).reset_index(drop=True)
