import os
import numpy as np
import pandas as pd

MATCHES_PATH = "data/matches.csv"
PROFILES_PATH = "data/team_profiles.csv"
FIFA_PATH = "data/fifa_rankings.csv"
SCHEDULE_PATH = "data/schedule.csv"
DB_PATH = "database/worldcup.db"

HOSTS_2026 = {"Mexico", "United States", "USA", "Canada"}

CANONICAL_TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
}

AMERICAS = [
    "Mexico", "USA", "United States", "Canada", "Brazil", "Argentina", "Uruguay", "Colombia",
    "Chile", "Peru", "Ecuador", "Costa Rica", "Panama", "Paraguay", "Venezuela", "Bolivia",
    "Jamaica", "Haiti",
]

COMPETITIVE = [
    "FIFA World Cup", "FIFA World Cup qualification", "Copa América", "UEFA Euro",
    "UEFA Euro qualification", "African Cup of Nations", "AFC Asian Cup", "CONCACAF Gold Cup",
    "UEFA Nations League", "CONCACAF Nations League", "African Cup of Nations qualification",
    "AFC Asian Cup qualification", "Copa América qualification",
]

FEATURES = [
    "home_elo_dyn", "away_elo_dyn", "home_elo_off", "away_elo_off", "home_elo_def", "away_elo_def",
    "home_streak", "away_streak", "home_mv", "away_mv", "home_gd", "away_gd", "h2h_home_wins",
    "h2h_away_wins", "home_americas", "away_americas", "home_avg_gf", "home_avg_ga", "away_avg_gf",
    "away_avg_ga", "is_competitive", "home_shots", "home_accuracy", "home_conversion", "home_pressure",
    "away_shots", "away_accuracy", "away_conversion", "away_pressure", "home_fifa_pts", "away_fifa_pts",
    "elo_diff", "mv_ratio", "home_streak_10", "away_streak_10", "home_def_consistency",
    "away_def_consistency", "home_off_efficiency", "away_off_efficiency",
]

MARKET_VALUES = {
    "England": 1400, "France": 1200, "Brazil": 1100, "Portugal": 1000, "Spain": 950,
    "Argentina": 900, "Germany": 850, "Italy": 800, "Netherlands": 750, "Belgium": 600,
    "Uruguay": 500, "Norway": 450, "Colombia": 400, "Sweden": 360, "Senegal": 350,
    "Croatia": 350, "Turkey": 340, "Austria": 320, "Ivory Coast": 320, "Morocco": 300,
    "Denmark": 300, "USA": 250, "United States": 250, "Mexico": 250, "Switzerland": 250,
    "Scotland": 220, "Canada": 220, "Japan": 200, "Czech Republic": 190, "South Korea": 180,
    "Algeria": 180, "Ghana": 180, "Ecuador": 150, "Egypt": 150, "Paraguay": 130,
    "DR Congo": 110, "Chile": 100, "Australia": 90, "Bosnia and Herzegovina": 85, "Peru": 80,
    "Tunisia": 70, "Saudi Arabia": 50, "South Africa": 45, "Qatar": 40, "Cape Verde": 40,
    "Uzbekistan": 35, "New Zealand": 35, "Panama": 30, "Jordan": 25, "Curaçao": 20,
    "Iraq": 15, "Haiti": 15,
}


def resolve_team_name(team_name):
    """Normaliza alias de selecciones al nombre canónico de matches.csv y schedule.csv."""
    if not isinstance(team_name, str):
        return team_name
    return CANONICAL_TEAM_ALIASES.get(team_name.strip(), team_name.strip())


def load_team_profiles():
    if not os.path.exists(PROFILES_PATH):
        return {}
    df = pd.read_csv(PROFILES_PATH)
    profiles = {}
    for _, row in df.iterrows():
        name = str(row["team"]).strip()
        profiles[name] = row
        profiles[resolve_team_name(name)] = row
    return profiles


TEAM_PROFILES = load_team_profiles()


def get_team_profile(team_name):
    canonical = resolve_team_name(team_name)
    p = TEAM_PROFILES.get(canonical, TEAM_PROFILES.get(team_name))
    if p is not None:
        return float(p["shots_per_match"]), float(p["shot_accuracy"]), float(p["conversion_rate"]), float(p["pressures_per_match"])
    return 11.0, 0.35, 0.10, 155.0


def load_fifa_rankings():
    if not os.path.exists(FIFA_PATH):
        return {}
    df = pd.read_csv(FIFA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["canonical_team"] = df["team"].map(resolve_team_name)
    df = df.sort_values("date", ascending=False).drop_duplicates("canonical_team")
    rankings = {}
    for _, row in df.iterrows():
        pts = float(row["total_points"])
        rankings[row["canonical_team"]] = pts
        rankings[row["team"]] = pts
    for alias, canon in CANONICAL_TEAM_ALIASES.items():
        if canon in rankings and alias not in rankings:
            rankings[alias] = rankings[canon]
    return rankings


FIFA_RANKINGS = load_fifa_rankings()


def get_market_value(team_name, elo):
    canonical = resolve_team_name(team_name)
    if canonical in MARKET_VALUES:
        return MARKET_VALUES[canonical]
    if team_name in MARKET_VALUES:
        return MARKET_VALUES[team_name]
    return max(5.0, (elo - 1400.0) * 0.5)


def update_elo_single_match(elo_dict, elo_off_dict, elo_def_dict, t1, t2, hg, ag, tournament, is_neutral):
    """Actualiza in-place los 3 diccionarios ELO para un partido finalizado."""
    elo1, elo2 = elo_dict.get(t1, 1500.0), elo_dict.get(t2, 1500.0)
    off1, def1 = elo_off_dict.get(t1, 1500.0), elo_def_dict.get(t1, 1500.0)
    off2, def2 = elo_off_dict.get(t2, 1500.0), elo_def_dict.get(t2, 1500.0)
    is_comp = tournament in COMPETITIVE
    w1 = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    k_base = 40.0 if is_comp else 20.0
    we1 = 1.0 / (10.0 ** (-(elo1 + (0.0 if is_neutral else 50.0) - elo2) / 400.0) + 1.0)
    elo_dict[t1] = elo1 + k_base * (w1 - we1)
    elo_dict[t2] = elo2 + k_base * ((1.0 - w1) - (1.0 - we1))
    we_hg = max(0.1, 1.0 + ((off1 - def2) / 400.0))
    we_ag = max(0.1, 1.0 + ((off2 - def1) / 400.0))
    k_goal = 20.0 if is_comp else 10.0
    elo_off_dict[t1] = off1 + k_goal * (hg - we_hg)
    elo_def_dict[t2] = def2 + k_goal * (we_hg - hg)
    elo_off_dict[t2] = off2 + k_goal * (ag - we_ag)
    elo_def_dict[t1] = def1 + k_goal * (we_ag - ag)


def calculate_dynamic_elo(df):
    """Calcula ELO dinámico general y ofensivo/defensivo respetando cancha neutral."""
    elo_dict, elo_off_dict, elo_def_dict = {}, {}, {}
    h_elo, a_elo, h_off, a_off, h_def, a_def = [], [], [], [], [], []
    neutrals = df["neutral"] if "neutral" in df.columns else [False] * len(df)
    tours = df["tournament"] if "tournament" in df.columns else [""] * len(df)
    for t1, t2, hg, ag, tour, neut in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"], tours, neutrals):
        h_elo.append(elo_dict.get(t1, 1500.0)); a_elo.append(elo_dict.get(t2, 1500.0))
        h_off.append(elo_off_dict.get(t1, 1500.0)); h_def.append(elo_def_dict.get(t1, 1500.0))
        a_off.append(elo_off_dict.get(t2, 1500.0)); a_def.append(elo_def_dict.get(t2, 1500.0))
        if pd.isna(hg) or pd.isna(ag):
            continue
        update_elo_single_match(elo_dict, elo_off_dict, elo_def_dict, t1, t2, float(hg), float(ag), str(tour), str(neut).strip().lower() == "true")
    df["home_elo_dyn"], df["away_elo_dyn"] = h_elo, a_elo
    df["home_elo_off"], df["away_elo_off"] = h_off, a_off
    df["home_elo_def"], df["away_elo_def"] = h_def, a_def
    return df, elo_dict, elo_off_dict, elo_def_dict


def add_streaks(df):
    """Calcula rachas y métricas históricas de 5 y 10 partidos alineadas con inferencia."""
    home = df[["date", "home_team", "home_score", "away_score"]].rename(columns={"home_team": "team", "home_score": "gf", "away_score": "ga"})
    away = df[["date", "away_team", "away_score", "home_score"]].rename(columns={"away_team": "team", "away_score": "gf", "home_score": "ga"})
    long_df = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)
    long_df["points"] = np.where(long_df["gf"] > long_df["ga"], 3, np.where(long_df["gf"] == long_df["ga"], 1, 0))
    long_df["gd"] = long_df["gf"] - long_df["ga"]
    grp = long_df.groupby("team")
    long_df["streak"] = grp["points"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0.0)
    long_df["gd_streak"] = grp["gd"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0.0)
    long_df["avg_goals"] = grp["gf"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.0)
    long_df["avg_conceded"] = grp["ga"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.0)

    def _w10(series):
        vals = series.values
        out = np.zeros(len(vals), dtype=float)
        for i in range(1, len(vals)):
            w = vals[max(0, i - 10):i]
            out[i] = (w * np.exp(np.linspace(-1, 0, len(w)))).sum()
        return pd.Series(out, index=series.index)

    long_df["streak_10_weighted"] = grp["points"].transform(_w10)
    gf10 = grp["gf"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(1.0)
    ga10 = grp["ga"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(1.0)
    long_df["defensive_consistency"] = 1.0 / (1.0 + ga10)
    long_df["offensive_efficiency"] = gf10 / (gf10 + ga10 + 1.0)

    cols = ["date", "team", "streak", "gd_streak", "avg_goals", "avg_conceded", "streak_10_weighted", "defensive_consistency", "offensive_efficiency"]
    feats = long_df[cols].drop_duplicates(subset=["date", "team"], keep="last")
    for side, col in [("home", "home_team"), ("away", "away_team")]:
        df = df.merge(feats, left_on=["date", col], right_on=["date", "team"], how="left").rename(columns={
            "streak": f"{side}_streak", "gd_streak": f"{side}_gd", "avg_goals": f"{side}_avg_gf",
            "avg_conceded": f"{side}_avg_ga", "streak_10_weighted": f"{side}_streak_10",
            "defensive_consistency": f"{side}_def_consistency", "offensive_efficiency": f"{side}_off_efficiency",
        }).drop(columns=["team"])
    return df.drop_duplicates(subset=["date", "home_team", "away_team"])


def add_h2h(df):
    """Calcula victorias H2H en los últimos 5 enfrentamientos directos previos."""
    df = df.sort_values("date")
    history, hw_list, aw_list = {}, [], []
    for t1, t2, hg, ag in zip(df["home_team"], df["away_team"], df["home_score"], df["away_score"]):
        key = tuple(sorted([t1, t2]))
        hw, aw = 0, 0
        for pm in history.get(key, [])[-5:]:
            if pm["home"] == t1:
                hw += int(pm["hg"] > pm["ag"]); aw += int(pm["ag"] > pm["hg"])
            else:
                hw += int(pm["ag"] > pm["hg"]); aw += int(pm["hg"] > pm["ag"])
        hw_list.append(hw); aw_list.append(aw)
        history.setdefault(key, []).append({"home": t1, "hg": hg, "ag": ag})
    df["h2h_home_wins"], df["h2h_away_wins"] = hw_list, aw_list
    return df


def _load_completed_fallback(df=None, as_of_date=None):
    if df is None:
        df = pd.read_csv(MATCHES_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    hist = df.dropna(subset=["home_score", "away_score"])
    return hist[hist["date"] < pd.to_datetime(as_of_date)] if as_of_date is not None else hist


def get_current_stats(team, df=None, as_of_date=None):
    hist = _load_completed_fallback(df, as_of_date)
    tc = resolve_team_name(team)
    tm = hist[(hist["home_team"] == tc) | (hist["away_team"] == tc)].tail(5)
    streak, gd, avg_gf, avg_ga = 0, 0, 0.0, 0.0
    for ht, hs, as_ in zip(tm["home_team"], tm["home_score"], tm["away_score"]):
        gf, ga = (hs, as_) if ht == tc else (as_, hs)
        streak += 3 if gf > ga else (1 if gf == ga else 0)
        gd += int(gf - ga); avg_gf += float(gf); avg_ga += float(ga)
    n = len(tm)
    return int(streak), int(gd), round(avg_gf / n if n else 1.0, 2), round(avg_ga / n if n else 1.0, 2)


def get_current_h2h(t1, t2, df=None, as_of_date=None):
    hist = _load_completed_fallback(df, as_of_date)
    c1, c2 = resolve_team_name(t1), resolve_team_name(t2)
    h2h = hist[((hist["home_team"] == c1) & (hist["away_team"] == c2)) | ((hist["home_team"] == c2) & (hist["away_team"] == c1))].tail(5)
    w1, w2 = 0, 0
    for ht, hs, as_ in zip(h2h["home_team"], h2h["home_score"], h2h["away_score"]):
        if ht == c1:
            w1 += int(hs > as_); w2 += int(as_ > hs)
        else:
            w1 += int(as_ > hs); w2 += int(hs > as_)
    return w1, w2


def get_advanced_stats(team, df=None, as_of_date=None):
    hist = _load_completed_fallback(df, as_of_date)
    tc = resolve_team_name(team)
    tm = hist[(hist["home_team"] == tc) | (hist["away_team"] == tc)].tail(10)
    pts, gf_tot, ga_tot = [], 0.0, 0.0
    for ht, hs, as_ in zip(tm["home_team"], tm["home_score"], tm["away_score"]):
        gf, ga = (float(hs), float(as_)) if ht == tc else (float(as_), float(hs))
        pts.append(3 if gf > ga else (1 if gf == ga else 0))
        gf_tot += gf; ga_tot += ga
    n = len(tm)
    streak_10 = float((np.array(pts) * np.exp(np.linspace(-1, 0, n))).sum()) if n else 0.0
    avg_gf, avg_ga = (gf_tot / n, ga_tot / n) if n else (1.0, 1.0)
    return streak_10, float(1.0 / (1.0 + avg_ga)), float(avg_gf / (avg_gf + avg_ga + 1.0))


def build_team_snapshot(team, elos, elos_off, elos_def, hist_df):
    elo = float(elos.get(team, 1500.0))
    streak, gd, avg_gf, avg_ga = get_current_stats(team, df=hist_df)
    s, a, c, p = get_team_profile(team)
    s10, d_cons, o_eff = get_advanced_stats(team, df=hist_df)
    return {
        "elo": elo, "elo_off": float(elos_off.get(team, 1500.0)), "elo_def": float(elos_def.get(team, 1500.0)),
        "streak": streak, "gd": gd, "avg_gf": avg_gf, "avg_ga": avg_ga, "mv": float(get_market_value(team, elo)),
        "am": 1 if team in AMERICAS else 0, "shots": s, "accuracy": a, "conversion": c, "pressure": p,
        "fifa": float(FIFA_RANKINGS.get(team, 1400.0)), "streak10": s10, "def_cons": d_cons, "off_eff": o_eff,
    }


def build_feature_row(h, a, h2h_h, h2h_a, is_comp=1):
    return [
        h["elo"], a["elo"], h["elo_off"], a["elo_off"], h["elo_def"], a["elo_def"], h["streak"], a["streak"],
        h["mv"], a["mv"], h["gd"], a["gd"], h2h_h, h2h_a, h["am"], a["am"], h["avg_gf"], h["avg_ga"],
        a["avg_gf"], a["avg_ga"], is_comp, h["shots"], h["accuracy"], h["conversion"], h["pressure"],
        a["shots"], a["accuracy"], a["conversion"], a["pressure"], h["fifa"], a["fifa"], h["elo"] - a["elo"],
        h["mv"] / (a["mv"] + 1.0), h["streak10"], a["streak10"], h["def_cons"], a["def_cons"], h["off_eff"], a["off_eff"],
    ]
