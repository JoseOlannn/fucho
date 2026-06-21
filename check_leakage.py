import pandas as pd

def check_leakage():
    # Cargar calendario y encontrar finalizados
    schedule = pd.read_csv("data/schedule.csv")
    finished = schedule[schedule["Estado"] == "Fin"]
    
    # Cargar base de datos de entrenamiento
    matches = pd.read_csv("data/matches.csv")
    matches['date'] = pd.to_datetime(matches['date']).dt.date
    
    leakage_count = 0
    leaked_matches = []
    
    for _, row in finished.iterrows():
        t1, t2 = row['Local'], row['Visitante']
        
        # Buscar en matches.csv si existe este partido
        # Chequeamos t1 y t2. Las fechas pueden variar ligeramente, así que filtramos por equipos.
        mask = ((matches['home_team'] == t1) & (matches['away_team'] == t2)) | ((matches['home_team'] == t2) & (matches['away_team'] == t1))
        
        # Ordenamos por fecha descendente y tomamos el más reciente
        match_history = matches[mask].sort_values('date', ascending=False)
        
        if not match_history.empty:
            # Si el partido en entrenamiento tiene exactamente el mismo resultado y fecha reciente, es leakage seguro.
            latest = match_history.iloc[0]
            # Podría ser un partido anterior, así que verificamos el resultado exacto
            # O simplemente asumimos que si el torneo es "FIFA World Cup" y el año es reciente (ej. 2026), es el mismo.
            if latest.get('tournament') == 'FIFA World Cup' and latest['date'].year >= 2026:
                leakage_count += 1
                leaked_matches.append(f"{t1} {latest['home_score']} - {latest['away_score']} {t2}")
            else:
                # Comprobar si el marcador coincide
                r_g1, r_g2 = int(row['GolesLocal']), int(row['GolesVisitante'])
                m_g1, m_g2 = latest['home_score'], latest['away_score']
                if (latest['home_team'] == t1 and r_g1 == m_g1 and r_g2 == m_g2) or (latest['home_team'] == t2 and r_g1 == m_g2 and r_g2 == m_g1):
                     leakage_count += 1
                     leaked_matches.append(f"{t1} vs {t2} (Marcador coincide: {r_g1}-{r_g2})")
                     
    print(f"Total partidos finalizados en schedule.csv: {len(finished)}")
    print(f"Partidos encontrados en matches.csv (LEAKAGE): {leakage_count}")
    for m in leaked_matches[:5]:
         print(f" - {m}")
    if len(leaked_matches) > 5:
         print("   ...")

if __name__ == '__main__':
    check_leakage()
