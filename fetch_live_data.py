import requests
import pandas as pd
import datetime
import os
from traducciones import traducir_pais as traducir

# ==========================================
# CONFIGURACIÓN DE APIs
# (Regístrate gratis para obtener tus llaves)
# ==========================================

# 1. API-Football (Vía api-sports.io)
# Para obtener partidos del día, resultados en vivo y estadísticas.
API_FOOTBALL_KEY = "c687d9103a88777c35b154b7bc85f726"
API_FOOTBALL_HOST = "v3.football.api-sports.io"

# 2. The Odds API (the-odds-api.com)
# Para comparar tu predicción con las cuotas reales de las casas de apuestas (Value Betting).
ODDS_API_KEY = "8426d9a56f0263d509508668833cefbc"

def fetch_todays_matches():
    """
    Obtiene los partidos programados para el día de hoy usando API-Football.
    Filtra por las competiciones más relevantes (Mundial, Euro, Copa América, etc.)
    """
    if API_FOOTBALL_KEY == "TU_API_KEY_AQUI":
        print("⚠️ Advertencia: No has configurado tu API_FOOTBALL_KEY.")
        return pd.DataFrame()

    url = "https://v3.football.api-sports.io/fixtures"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    headers = {
        'x-apisports-key': API_FOOTBALL_KEY,
        'x-apisports-host': API_FOOTBALL_HOST
    }
    
    params = {
        'date': today
    }

    try:
        print(f"📥 Obteniendo partidos para hoy ({today})...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        matches = []
        for fixture in data.get('response', []):
            league = fixture['league']['name']
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            status = fixture['fixture']['status']['short']
            time = fixture['fixture']['date']
            
            matches.append({
                'League': league,
                'Time': time,
                'Home Team': traducir(home_team),
                'Away Team': traducir(away_team),
                'Status': status
            })
            
        df = pd.DataFrame(matches)
        
        # Guardar como un archivo temporal si quieres que app.py lo lea
        if not df.empty:
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/todays_live_matches.csv", index=False)
            print(f"✅ Se guardaron {len(df)} partidos en data/todays_live_matches.csv")
            
        return df

    except Exception as e:
        print(f"❌ Error conectando a API-Football: {e}")
        return pd.DataFrame()

def fetch_live_odds(sport_key='soccer_fifa_world_cup'):
    """
    Obtiene las cuotas (momios) en tiempo real para encontrar "Value Bets".
    sport_key: 'soccer_fifa_world_cup', 'soccer_conmebol_copa_america', 'soccer_uefa_euro_qualification'
    """
    if ODDS_API_KEY == "TU_API_KEY_AQUI":
        print("⚠️ Advertencia: No has configurado tu ODDS_API_KEY.")
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us,eu,uk', # Regiones de casas de apuestas
        'markets': 'h2h',      # h2h = Home, Draw, Away (Ganador del partido)
        'oddsFormat': 'decimal'
    }

    try:
        print(f"📊 Obteniendo momios en vivo para {sport_key}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        odds_data = response.json()
        
        results = []
        for match in odds_data:
            home = match['home_team']
            away = match['away_team']
            
            # Tomamos la casa de apuestas que mejor pague o un promedio
            bookmakers = match.get('bookmakers', [])
            if not bookmakers:
                continue
                
            # Usaremos la primera casa de apuestas disponible para el ejemplo
            best_odds = bookmakers[0]['markets'][0]['outcomes']
            
            odds_dict = {
                'Match': f"{home} vs {away}",
                'Home Team': home,
                'Away Team': away,
                'Bookmaker': bookmakers[0]['title']
            }
            
            for outcome in best_odds:
                if outcome['name'] == home:
                    odds_dict['Odd_Home'] = outcome['price']
                    odds_dict['Implied_Prob_Home'] = round((1 / outcome['price']) * 100, 2)
                elif outcome['name'] == away:
                    odds_dict['Odd_Away'] = outcome['price']
                    odds_dict['Implied_Prob_Away'] = round((1 / outcome['price']) * 100, 2)
                elif outcome['name'] == 'Draw':
                    odds_dict['Odd_Draw'] = outcome['price']
                    odds_dict['Implied_Prob_Draw'] = round((1 / outcome['price']) * 100, 2)
                    
            results.append(odds_dict)
            
        df = pd.DataFrame(results)
        
        if not df.empty:
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/live_odds.csv", index=False)
            print(f"✅ Se guardaron los momios de {len(df)} partidos en data/live_odds.csv")
            
        return df

    except Exception as e:
        print(f"❌ Error conectando a The Odds API: {e}")
        return pd.DataFrame()

def generate_value_bets():
    """
    Cruza los momios del Mundial (live_odds.csv) con tu modelo de IA (model.py)
    para encontrar Apuestas de Valor automáticamente.
    """
    print("\n[Paso 3] Cruzando IA con Momios del Casino (Buscando Value Bets para el Mundial)...")
    try:
        if not os.path.exists("data/live_odds.csv"):
            print("⚠️ No hay momios descargados. Descarga las cuotas primero.")
            return pd.DataFrame()
            
        odds_df = pd.read_csv("data/live_odds.csv")
        if odds_df.empty:
            return pd.DataFrame()
            
        import model  # Importamos tu Inteligencia Artificial
        import warnings
        warnings.filterwarnings('ignore') # Ocultar warnings de scikit-learn
        
        value_bets = []
        
        for _, row in odds_df.iterrows():
            home = row['Home Team']
            away = row['Away Team']
            
            # Asegurar compatibilidad de nombres (ej. USA vs United States)
            if home == 'USA': home = 'United States'
            if away == 'USA': away = 'United States'
            
            try:
                # 1. Le pedimos a tu modelo que analice el partido
                pred = model.predict_match(home, away)
                
                # 2. Comparamos tu probabilidad vs la del casino
                # Local
                edge_home = pred['team1_win'] - row['Implied_Prob_Home']
                if edge_home > 5.0: # Si tu modelo le da 5% más de prob que el casino
                    value_bets.append({
                        'Partido': f"{traducir(home)} vs {traducir(away)}",
                        'Apuesta Sugerida': f"Gana {traducir(home)}",
                        'Tu Probabilidad (IA)': f"{pred['team1_win']}%",
                        'Probabilidad Casino': f"{row['Implied_Prob_Home']}%",
                        'Ventaja Matemática (Edge)': f"+{round(edge_home, 1)}%",
                        'Momio a Comprar': row.get('Odd_Home', 0)
                    })
                    
                # Visitante
                edge_away = pred['team2_win'] - row['Implied_Prob_Away']
                if edge_away > 5.0:
                    value_bets.append({
                        'Partido': f"{traducir(home)} vs {traducir(away)}",
                        'Apuesta Sugerida': f"Gana {traducir(away)}",
                        'Tu Probabilidad (IA)': f"{pred['team2_win']}%",
                        'Probabilidad Casino': f"{row['Implied_Prob_Away']}%",
                        'Ventaja Matemática (Edge)': f"+{round(edge_away, 1)}%",
                        'Momio a Comprar': row.get('Odd_Away', 0)
                    })
                    
                # Empate
                edge_draw = pred['draw'] - row['Implied_Prob_Draw']
                if edge_draw > 5.0:
                    value_bets.append({
                        'Partido': f"{traducir(home)} vs {traducir(away)}",
                        'Apuesta Sugerida': f"Empate",
                        'Tu Probabilidad (IA)': f"{pred['draw']}%",
                        'Probabilidad Casino': f"{row['Implied_Prob_Draw']}%",
                        'Ventaja Matemática (Edge)': f"+{round(edge_draw, 1)}%",
                        'Momio a Comprar': row.get('Odd_Draw', 0)
                    })
                    
            except Exception as e:
                # Puede que un equipo no esté en tu base de datos (ej. muy desconocido)
                continue
                
        vb_df = pd.DataFrame(value_bets)
        if not vb_df.empty:
            # Ordenar de mayor a menor ventaja (edge)
            vb_df['Edge_Num'] = vb_df['Ventaja Matemática (Edge)'].str.replace('+', '').str.replace('%', '').astype(float)
            vb_df = vb_df.sort_values('Edge_Num', ascending=False).drop('Edge_Num', axis=1)
            
            vb_df.to_csv("data/value_bets_today.csv", index=False)
            print(f"💰 ¡ÉXITO! Tu IA encontró {len(vb_df)} apuestas de valor para el Mundial.")
        else:
            print("❌ Hoy no se encontraron apuestas de valor suficientes. (El casino tiene momios perfectos hoy).")
            
        return vb_df

    except Exception as e:
        print(f"❌ Error generando Value Bets: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("===================================================")
    print(" INICIANDO INTEGRACIÓN DE DATOS EN TIEMPO REAL")
    print("===================================================")
    
    print("\n[Paso 1] Descargando Partidos del Día...")
    matches_df = fetch_todays_matches()
        
    print("\n[Paso 2] Descargando Cuotas de Apuestas (MUNDIAL)...")
    odds_df = fetch_live_odds('soccer_fifa_world_cup') 
    
    # NUEVO PASO: Generar predicciones automatizadas
    generate_value_bets()
    
    print("\n¡Proceso Terminado!")
