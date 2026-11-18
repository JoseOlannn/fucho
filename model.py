import datetime
import os
import sqlite3
import zlib
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from features import (
    AMERICAS, COMPETITIVE, DB_PATH, FEATURES, FIFA_RANKINGS, HOSTS_2026, MATCHES_PATH, SCHEDULE_PATH,
    add_h2h, add_streaks, build_feature_row, build_team_snapshot, calculate_dynamic_elo,
    get_current_h2h, get_market_value, get_team_profile, resolve_team_name, update_elo_single_match,
)

_ARTIFACT_CACHE = {}
_MATCHES_CACHE = {"mtime": None, "all": None, "completed": None}
_ELO_AS_OF_CACHE = {}
_PREDICT_CACHE = {}


def clear_model_cache():
    _ARTIFACT_CACHE.clear()
    _MATCHES_CACHE.update({"mtime": None, "all": None, "completed": None})
    _ELO_AS_OF_CACHE.clear()
    _PREDICT_CACHE.clear()


def _load_matches_df(completed_only=True):
    mtime = os.path.getmtime(MATCHES_PATH)
    if _MATCHES_CACHE["mtime"] != mtime or _MATCHES_CACHE["all"] is None:
        df = pd.read_csv(MATCHES_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        _MATCHES_CACHE["mtime"] = mtime
        _MATCHES_CACHE["all"] = df
        _MATCHES_CACHE["completed"] = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    return _MATCHES_CACHE["completed"] if completed_only else _MATCHES_CACHE["all"]


def _set_single_thread(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _set_single_thread(v)
        return
    if hasattr(obj, "n_jobs"):
        try:
            obj.n_jobs = 1
        except Exception:
            pass
    if hasattr(obj, "estimator"):
        _set_single_thread(obj.estimator)
    if hasattr(obj, "calibrated_classifiers_"):
        for cc in obj.calibrated_classifiers_:
            if hasattr(cc, "estimator"):
                _set_single_thread(cc.estimator)


def _load_artifact(path):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    cached = _ARTIFACT_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        loaded = joblib.load(path)
        _set_single_thread(loaded)
        _ARTIFACT_CACHE[path] = (mtime, loaded)
    return _ARTIFACT_CACHE[path][1]


def _get_elo_as_of(df_completed, as_of_date=None):
    if as_of_date is None:
        elos = _load_artifact("database/elo_dict.pkl") or {}
        if not elos:
            train_model()
            elos = _load_artifact("database/elo_dict.pkl") or {}
        return elos, _load_artifact("database/elo_off_dict.pkl") or {}, _load_artifact("database/elo_def_dict.pkl") or {}
    cutoff = pd.to_datetime(as_of_date)
    cache_key = (_MATCHES_CACHE.get("mtime"), str(cutoff.date()))
    if cache_key in _ELO_AS_OF_CACHE:
        return _ELO_AS_OF_CACHE[cache_key]
    hist = df_completed[df_completed["date"] < cutoff]
    elo_dict, elo_off, elo_def = {}, {}, {}
    neutrals = hist["neutral"] if "neutral" in hist.columns else [False] * len(hist)
    tours = hist["tournament"] if "tournament" in hist.columns else [""] * len(hist)
    for t1, t2, hg, ag, tour, neut in zip(hist["home_team"], hist["away_team"], hist["home_score"], hist["away_score"], tours, neutrals):
        update_elo_single_match(elo_dict, elo_off, elo_def, t1, t2, float(hg), float(ag), str(tour), str(neut).strip().lower() == "true")
    _ELO_AS_OF_CACHE[cache_key] = (elo_dict, elo_off, elo_def)
    return _ELO_AS_OF_CACHE[cache_key]


def _filter_history(as_of_date=None):
    base_df = _load_matches_df(completed_only=True)
    return base_df[base_df["date"] < pd.to_datetime(as_of_date)] if as_of_date is not None else base_df


def train_model():
    print("Calculando ELO Dinámico...")
    os.makedirs("database", exist_ok=True)
    df = pd.read_csv(MATCHES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df, elo_dict, elo_off_dict, elo_def_dict = calculate_dynamic_elo(df)
    joblib.dump(elo_dict, "database/elo_dict.pkl")
    joblib.dump(elo_off_dict, "database/elo_off_dict.pkl")
    joblib.dump(elo_def_dict, "database/elo_def_dict.pkl")

    rankings_df = pd.DataFrame(list(elo_dict.items()), columns=["Team", "ELO"]).sort_values("ELO", ascending=False)
    with sqlite3.connect(DB_PATH) as conn:
        rankings_df.to_sql("Rankings", conn, if_exists="replace", index=False)

    df = df[(df["date"].dt.year >= 2010) & (df["date"] < "2026-06-11")].dropna(subset=["home_score", "away_score"]).copy()
    df = add_h2h(add_streaks(df))
    df["home_americas"] = df["home_team"].isin(AMERICAS).astype(int)
    df["away_americas"] = df["away_team"].isin(AMERICAS).astype(int)
    df["is_competitive"] = df["tournament"].isin(COMPETITIVE).astype(int)
    df["home_mv"] = [get_market_value(t, e) for t, e in zip(df["home_team"], df["home_elo_dyn"])]
    df["away_mv"] = [get_market_value(t, e) for t, e in zip(df["away_team"], df["away_elo_dyn"])]
    hp = [get_team_profile(t) for t in df["home_team"]]
    ap = [get_team_profile(t) for t in df["away_team"]]
    df["home_shots"], df["home_accuracy"], df["home_conversion"], df["home_pressure"] = zip(*hp)
    df["away_shots"], df["away_accuracy"], df["away_conversion"], df["away_pressure"] = zip(*ap)
    df["home_fifa_pts"] = df["home_team"].map(lambda t: FIFA_RANKINGS.get(resolve_team_name(t), 1400.0))
    df["away_fifa_pts"] = df["away_team"].map(lambda t: FIFA_RANKINGS.get(resolve_team_name(t), 1400.0))
    df["elo_diff"] = df["home_elo_dyn"] - df["away_elo_dyn"]
    df["mv_ratio"] = df["home_mv"] / (df["away_mv"] + 1.0)

    X_data = df[FEATURES].fillna(0)
    y_class = np.where(df["home_score"] > df["away_score"], 2, np.where(df["home_score"] == df["away_score"], 1, 0))
    years_ago = (df["date"].max() - df["date"]).dt.days / 365.25
    weights = np.exp(-years_ago * 0.15) * np.where(df["tournament"].isin(COMPETITIVE), 1.5, 0.7)

    print("Entrenando Ensemble de 4 Clasificadores Calibrados...")
    ensemble = {
        "xgb": CalibratedClassifierCV(XGBClassifier(random_state=42, eval_metric="mlogloss", subsample=1.0, n_estimators=200, max_depth=4, learning_rate=0.01, colsample_bytree=0.8, n_jobs=2), method="isotonic", cv=3).fit(X_data, y_class, sample_weight=weights),
        "rf": CalibratedClassifierCV(RandomForestClassifier(random_state=42, n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_depth=8, n_jobs=2), method="isotonic", cv=3).fit(X_data, y_class, sample_weight=weights),
        "gb": CalibratedClassifierCV(GradientBoostingClassifier(random_state=42, subsample=0.9, n_estimators=100, max_depth=3, learning_rate=0.05), method="isotonic", cv=3).fit(X_data, y_class, sample_weight=weights),
        "lgb": CalibratedClassifierCV(LGBMClassifier(random_state=42, verbose=-1, n_estimators=150, max_depth=5, learning_rate=0.05, num_leaves=31, subsample=0.9, n_jobs=2), method="isotonic", cv=3).fit(X_data, y_class, sample_weight=weights),
    }
    joblib.dump(ensemble, "database/ensemble_model.pkl")

    print("Entrenando Regresores de Goles (Ensemble)...")
    goal_ensemble = {
        "hg_xgb": XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.9, n_jobs=2).fit(X_data, df["home_score"], sample_weight=weights),
        "ag_xgb": XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.9, n_jobs=2).fit(X_data, df["away_score"], sample_weight=weights),
        "hg_rf": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=2).fit(X_data, df["home_score"], sample_weight=weights),
        "ag_rf": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=2).fit(X_data, df["away_score"], sample_weight=weights),
        "hg_lgb": LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=2).fit(X_data, df["home_score"], sample_weight=weights),
        "ag_lgb": LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=2).fit(X_data, df["away_score"], sample_weight=weights),
    }
    joblib.dump(goal_ensemble, "database/goal_ensemble.pkl")
    clear_model_cache()
    print("¡Ensemble entrenado exitosamente!")


def get_elos():
    if not os.path.exists("database/elo_dict.pkl"):
        train_model()
    return _load_artifact("database/elo_dict.pkl") or {}


def _predict_batch(rows_list, ensemble, goal_ensemble):
    X_df = pd.DataFrame(rows_list, columns=FEATURES)
    n = len(rows_list)
    per_clf = [{label: 0.0 for label in (0, 1, 2)} for _ in range(n)]
    for clf in ensemble.values():
        probs_mat = clf.predict_proba(X_df)
        for i in range(n):
            for c, p in zip(clf.classes_, probs_mat[i]):
                per_clf[i][int(c)] += float(p) / len(ensemble)
    if goal_ensemble is None:
        return per_clf, [1.2] * n, [1.0] * n
    hg = goal_ensemble["hg_xgb"].predict(X_df) * 0.33 + goal_ensemble["hg_rf"].predict(X_df) * 0.33 + goal_ensemble["hg_lgb"].predict(X_df) * 0.34
    ag = goal_ensemble["ag_xgb"].predict(X_df) * 0.33 + goal_ensemble["ag_rf"].predict(X_df) * 0.33 + goal_ensemble["ag_lgb"].predict(X_df) * 0.34
    return per_clf, [max(0.1, float(v)) for v in hg], [max(0.1, float(v)) for v in ag]


def _sample_overdispersed_bivariate_poisson(rng, xg1, xg2, is_comp, sims=10000, variance_mult=1.15):
    lambda3 = min(0.25 if is_comp else 0.15, 0.35 * min(xg1, xg2))
    l1, l2 = max(0.02, xg1 - lambda3), max(0.02, xg2 - lambda3)
    if variance_mult > 1.0:
        scale = variance_mult - 1.0
        r1, r2 = rng.gamma(max(1e-4, l1 / scale), scale, sims), rng.gamma(max(1e-4, l2 / scale), scale, sims)
    else:
        r1, r2 = l1, l2
    z = rng.poisson(lambda3, sims)
    return rng.poisson(r1) + z, rng.poisson(r2) + z


def predict_match(team1, team2, neutral=None, mod_xg1=1.0, mod_xg2=1.0, as_of_date=None):
    t1, t2 = resolve_team_name(team1), resolve_team_name(team2)
    if not os.path.exists("database/ensemble_model.pkl"):
        train_model()
    if neutral is None:
        neutral = (t1 not in HOSTS_2026) and (t2 not in HOSTS_2026)

    hist_df = _filter_history(as_of_date=as_of_date)
    cache_key = (
        t1, t2, bool(neutral), round(float(mod_xg1), 4), round(float(mod_xg2), 4),
        str(as_of_date), _MATCHES_CACHE.get("mtime"),
        os.path.getmtime("database/ensemble_model.pkl"),
        os.path.getmtime("database/goal_ensemble.pkl") if os.path.exists("database/goal_ensemble.pkl") else None,
    )
    if cache_key in _PREDICT_CACHE:
        return dict(_PREDICT_CACHE[cache_key])

    ensemble, goal_ensemble = _load_artifact("database/ensemble_model.pkl"), _load_artifact("database/goal_ensemble.pkl")
    elos, elos_off, elos_def = _get_elo_as_of(hist_df, as_of_date=as_of_date)
    s1, s2 = build_team_snapshot(t1, elos, elos_off, elos_def, hist_df), build_team_snapshot(t2, elos, elos_off, elos_def, hist_df)
    h2h1, h2h2 = get_current_h2h(t1, t2, df=hist_df)

    if neutral:
        probs, xgh, xga = _predict_batch([build_feature_row(s1, s2, h2h1, h2h2), build_feature_row(s2, s1, h2h2, h2h1)], ensemble, goal_ensemble)
        p_w1 = 0.5 * probs[0][2] + 0.5 * probs[1][0]
        p_d = 0.5 * probs[0][1] + 0.5 * probs[1][1]
        p_w2 = 0.5 * probs[0][0] + 0.5 * probs[1][2]
        xg1_raw, xg2_raw = 0.5 * xgh[0] + 0.5 * xga[1], 0.5 * xga[0] + 0.5 * xgh[1]
    elif t2 in HOSTS_2026 and t1 not in HOSTS_2026:
        probs, xgh, xga = _predict_batch([build_feature_row(s2, s1, h2h2, h2h1)], ensemble, goal_ensemble)
        p_w1, p_d, p_w2, xg1_raw, xg2_raw = probs[0][0], probs[0][1], probs[0][2], xga[0], xgh[0]
    else:
        probs, xgh, xga = _predict_batch([build_feature_row(s1, s2, h2h1, h2h2)], ensemble, goal_ensemble)
        p_w1, p_d, p_w2, xg1_raw, xg2_raw = probs[0][2], probs[0][1], probs[0][0], xgh[0], xga[0]

    xg1 = min(4.5, max(0.08, xg1_raw * ((max(500.0, s1["elo_off"]) / max(1000.0, s2["elo_def"])) ** 0.35) * float(mod_xg1)))
    xg2 = min(4.5, max(0.08, xg2_raw * ((max(500.0, s2["elo_off"]) / max(1000.0, s1["elo_def"])) ** 0.35) * float(mod_xg2)))

    seed = zlib.crc32(f"{t1}|{t2}|{xg1:.4f}|{xg2:.4f}|{int(bool(neutral))}".encode("utf-8"))
    g1_sims, g2_sims = _sample_overdispersed_bivariate_poisson(np.random.default_rng(seed), xg1, xg2, is_comp=1, sims=10000)
    mc_w1, mc_d, mc_w2 = float(np.mean(g1_sims > g2_sims)), float(np.mean(g1_sims == g2_sims)), float(np.mean(g2_sims > g1_sims))

    w_clf = 0.15 if (abs(float(mod_xg1) - 1.0) > 1e-6 or abs(float(mod_xg2) - 1.0) > 1e-6) else 0.45
    prob_w1, prob_d, prob_w2 = w_clf * p_w1 + (1.0 - w_clf) * mc_w1, w_clf * p_d + (1.0 - w_clf) * mc_d, w_clf * p_w2 + (1.0 - w_clf) * mc_w2
    tot = prob_w1 + prob_d + prob_w2
    prob_w1, prob_d, prob_w2 = (prob_w1 / tot, prob_d / tot, prob_w2 / tot) if tot > 0 else (mc_w1, mc_d, mc_w2)

    sw = np.where(g1_sims > g2_sims, prob_w1 / max(mc_w1, 1e-6), np.where(g1_sims == g2_sims, prob_d / max(mc_d, 1e-6), prob_w2 / max(mc_w2, 1e-6)))
    sw_sum = float(sw.sum())
    over_25 = float(np.sum(sw[(g1_sims + g2_sims) > 2.5]) / sw_sum)
    btts = float(np.sum(sw[(g1_sims > 0) & (g2_sims > 0)]) / sw_sum)

    score_probs = {}
    for g1, g2, w in zip(g1_sims.tolist(), g2_sims.tolist(), sw.tolist()):
        score_probs[(g1, g2)] = score_probs.get((g1, g2), 0.0) + w / sw_sum
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top_3 = [(f"{g1}-{g2}", round(p * 100.0, 2)) for (g1, g2), p in sorted_scores[:3]]

    max_w = max(prob_w1, prob_w2)
    target = "D" if ((prob_d >= 0.285 and max_w < 0.50) or prob_d >= max_w) else ("H" if prob_w1 >= prob_w2 else "A")
    official_score, official_prob, best_u = top_3[0][0], top_3[0][1], -1.0
    for (g1, g2), p in sorted_scores:
        if ("H" if g1 > g2 else ("A" if g2 > g1 else "D")) == target:
            u = p * np.exp(-0.38 * ((g1 - xg1) ** 2 + (g2 - xg2) ** 2))
            if u > best_u:
                best_u, official_score, official_prob = u, f"{g1}-{g2}", round(p * 100.0, 2)

    poss1 = max(30.0, min(70.0, 50.0 + ((s1["elo"] - s2["elo"]) / 30.0)))
    result = {
        "team1_win": round(prob_w1 * 100.0, 2), "draw": round(prob_d * 100.0, 2), "team2_win": round(prob_w2 * 100.0, 2),
        "over_25": round(over_25 * 100.0, 2), "btts": round(btts * 100.0, 2),
        "elo1": round(s1["elo"], 1), "elo2": round(s2["elo"], 1),
        "elo_off1": round(s1["elo_off"], 1), "elo_off2": round(s2["elo_off"], 1),
        "elo_def1": round(s1["elo_def"], 1), "elo_def2": round(s2["elo_def"], 1),
        "streak1": int(s1["streak"]), "streak2": int(s2["streak"]),
        "mv1": round(s1["mv"], 1), "mv2": round(s2["mv"], 1),
        "xg1": round(float(xg1), 2), "xg2": round(float(xg2), 2),
        "poss1": round(float(poss1), 1), "poss2": round(100.0 - poss1, 1),
        "gd1": int(s1["gd"]), "gd2": int(s2["gd"]), "h2h1": int(h2h1), "h2h2": int(h2h2),
        "am1": int(s1["am"]), "am2": int(s2["am"]),
        "avg_gf1": s1["avg_gf"], "avg_ga1": s1["avg_ga"], "avg_gf2": s2["avg_gf"], "avg_ga2": s2["avg_ga"],
        "official_score": official_score, "official_prob": official_prob, "top_3_scenarios": top_3,
    }
    _PREDICT_CACHE[cache_key] = dict(result)
    return result


def register_live_match(t1, t2, g1, g2, match_date=None, tournament="FIFA World Cup", neutral=True):
    """Registra resultado en vivo en matches.csv, schedule.csv, diccionarios ELO y SQLite."""
    c1, c2 = resolve_team_name(t1), resolve_team_name(t2)
    df_m = pd.read_csv(MATCHES_PATH)
    mask = df_m["home_score"].isna() & (((df_m["home_team"] == c1) & (df_m["away_team"] == c2)) | ((df_m["home_team"] == c2) & (df_m["away_team"] == c1)))
    if mask.any():
        idx = df_m[mask].index[0]
        hg, ag = (float(g1), float(g2)) if df_m.loc[idx, "home_team"] == c1 else (float(g2), float(g1))
        df_m.loc[idx, ["home_score", "away_score"]] = [hg, ag]
        t_home, t_away, tour, is_neut = df_m.loc[idx, "home_team"], df_m.loc[idx, "away_team"], str(df_m.loc[idx, "tournament"]), str(df_m.loc[idx, "neutral"]).strip().lower() == "true"
    else:
        t_home, t_away, hg, ag, tour, is_neut = c1, c2, float(g1), float(g2), tournament, bool(neutral)
        df_m = pd.concat([df_m, pd.DataFrame([{"date": str(match_date or datetime.date.today()), "home_team": c1, "away_team": c2, "home_score": hg, "away_score": ag, "tournament": tour, "city": "Neutral", "country": "Neutral", "neutral": is_neut}])], ignore_index=True)
    df_m.to_csv(MATCHES_PATH, index=False, lineterminator="\n")

    if os.path.exists(SCHEDULE_PATH):
        df_cal = pd.read_csv(SCHEDULE_PATH)
        sm = ((df_cal["Local"] == c1) & (df_cal["Visitante"] == c2)) | ((df_cal["Local"] == c2) & (df_cal["Visitante"] == c1))
        if sm.any():
            idx = df_cal[sm].index[0]
            df_cal.loc[idx, "Estado"] = "Fin"
            gl, gv = (float(g1), float(g2)) if df_cal.loc[idx, "Local"] == c1 else (float(g2), float(g1))
            df_cal.loc[idx, ["GolesLocal", "GolesVisitante"]] = [gl, gv]
            df_cal.to_csv(SCHEDULE_PATH, index=False, lineterminator="\n")

    elo_d = joblib.load("database/elo_dict.pkl") if os.path.exists("database/elo_dict.pkl") else {}
    off_d = joblib.load("database/elo_off_dict.pkl") if os.path.exists("database/elo_off_dict.pkl") else {}
    def_d = joblib.load("database/elo_def_dict.pkl") if os.path.exists("database/elo_def_dict.pkl") else {}
    update_elo_single_match(elo_d, off_d, def_d, t_home, t_away, hg, ag, tour, is_neut)
    joblib.dump(elo_d, "database/elo_dict.pkl"); joblib.dump(off_d, "database/elo_off_dict.pkl"); joblib.dump(def_d, "database/elo_def_dict.pkl")
    with sqlite3.connect(DB_PATH) as conn:
        pd.DataFrame(list(elo_d.items()), columns=["Team", "ELO"]).sort_values("ELO", ascending=False).to_sql("Rankings", conn, if_exists="replace", index=False)
    clear_model_cache()


if __name__ == "__main__":
    train_model()
