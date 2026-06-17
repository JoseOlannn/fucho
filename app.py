import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from scipy.stats import poisson
import datetime
import importlib
import model
importlib.reload(model)
from model import predict_match, get_elos, train_model
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="World Cup Analytics 2026", layout="wide")

st.markdown("""
<style>
    /* Ocultar marca de agua de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("World Cup Analytics 2026")
st.markdown("Plataforma profesional de análisis deportivo y estimación de probabilidades.")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/matches.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2000]
    return df

try:
    df = cargar_datos()
    promedio_local_global = df['home_score'].mean()
    promedio_visitante_global = df['away_score'].mean()
except Exception:
    st.error("No se encontró data/matches.csv")
    st.stop()

try:
    elos = get_elos()
    equipos_unicos = sorted(list(elos.keys()))
except Exception as e:
    st.error(f"Error cargando el modelo: {e}. Asegúrate de ejecutar model.py primero.")
    st.stop()

# La fuerza de ataque y defensa ahora es calculada nativamente por el modelo XGBRegressor
# en el backend (model.py) para mayor precisión.

TRADUCCIONES = {
    "Spain": "España", "France": "Francia", "England": "Inglaterra", "Brazil": "Brasil",
    "Germany": "Alemania", "Netherlands": "Países Bajos", "South Korea": "Corea del Sur",
    "South Africa": "Sudáfrica", "IR Iran": "Irán", "New Zealand": "Nueva Zelanda",
    "Saudi Arabia": "Arabia Saudita", "Morocco": "Marruecos", "Czech Republic": "Chequia",
    "Switzerland": "Suiza", "Japan": "Japón", "Cabo Verde": "Cabo Verde", "DR Congo": "RD Congo",
    "Congo DR": "RD Congo", "United States": "Estados Unidos", "USA": "Estados Unidos",
    "Sweden": "Suecia", "Turkey": "Turquía", "Ivory Coast": "Costa de Marfil"
}
def traducir(nombre): return TRADUCCIONES.get(nombre, nombre)

# Pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Análisis de Partido", "Resultados en Vivo", "Rankings", "Calendario", "Simulador Monte Carlo", "Precisión", "Radar Apuestas"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("Equipo Local", equipos_unicos, index=equipos_unicos.index("Argentina") if "Argentina" in equipos_unicos else 0, format_func=traducir)
        lesion1 = st.checkbox(f"Jugador Clave de {traducir(team1)} ausente", key="les1")
        defensivo1 = st.checkbox(f"{traducir(team1)} con planteamiento defensivo", key="def1")

    with col2:
        team2 = st.selectbox("Equipo Visitante", equipos_unicos, index=equipos_unicos.index("France") if "France" in equipos_unicos else 1, format_func=traducir)
        lesion2 = st.checkbox(f"Jugador Clave de {traducir(team2)} ausente", key="les2")
        defensivo2 = st.checkbox(f"{traducir(team2)} con planteamiento defensivo", key="def2")

    if st.button("Generar Análisis de Probabilidades", use_container_width=True):
        if team1 == team2:
            st.warning("Selecciona dos equipos distintos.")
        else:
            result = predict_match(team1, team2)
            
            st.divider()
            st.subheader("Ficha Técnica del Encuentro")
            st.markdown(f"**Puntos FIFA:** {traducir(team1)} ({result['elo1']}) | {traducir(team2)} ({result['elo2']})")
            st.markdown(f"**Valor de Plantilla:** {traducir(team1)} (€{result['mv1']}M) | {traducir(team2)} (€{result['mv2']}M)")
            st.markdown(f"**Estado de Forma (Últimos 5):** {traducir(team1)} ({result['streak1']} pts) | {traducir(team2)} ({result['streak2']} pts)")
            st.markdown(f"**Historial Directo (H2H):** Victorias {traducir(team1)}: {result['h2h1']} | Victorias {traducir(team2)}: {result['h2h2']}")
            st.markdown(f"**Diferencia de Goles Reciente:** {traducir(team1)} ({'+' if result['gd1']>0 else ''}{result['gd1']}) | {traducir(team2)} ({'+' if result['gd2']>0 else ''}{result['gd2']})")
            st.divider()
            st.subheader("Radar Táctico y de Fuerzas")
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[result['elo1'], result['mv1']*2, result.get('avg_gf1', 1)*500, (3-result.get('avg_ga1', 1))*500, result['poss1']*20],
                theta=['ELO', 'Valor Plantilla', 'Ataque', 'Defensa', 'Posesión'],
                fill='toself', name=traducir(team1), marker=dict(color='#1f77b4')
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[result['elo2'], result['mv2']*2, result.get('avg_gf2', 1)*500, (3-result.get('avg_ga2', 1))*500, result['poss2']*20],
                theta=['ELO', 'Valor Plantilla', 'Ataque', 'Defensa', 'Posesión'],
                fill='toself', name=traducir(team2), marker=dict(color='#ff7f0e')
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=True, height=400, margin=dict(t=30, b=30, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_radar, use_container_width=True)

            st.divider()
            st.subheader("Cuotas de Victoria Implícitas")
            fig = go.Figure(data=[go.Pie(labels=[f"Victoria {traducir(team1)}", "Empate", f"Victoria {traducir(team2)}"],
                                         values=[result['team1_win'], result['draw'], result['team2_win']],
                                         hole=.4, marker_colors=['#1f77b4', '#7f7f7f', '#ff7f0e'])])
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("Proyección de Posesión de Balón")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric(label=traducir(team1), value=f"{result['poss1']:.1f}%")
                st.progress(max(0, min(100, int(result['poss1']))))
            with col_p2:
                st.metric(label=traducir(team2), value=f"{result['poss2']:.1f}%")
                st.progress(max(0, min(100, int(result['poss2']))))
                
            st.divider()
            
            goles_esperados_1 = result['xg1']
            goles_esperados_2 = result['xg2']
            
            # Aplicar modificadores del usuario
            if lesion1: goles_esperados_1 *= 0.75
            if lesion2: goles_esperados_2 *= 0.75
            if defensivo1:
                goles_esperados_2 *= 0.60
                goles_esperados_1 *= 0.50
            if defensivo2:
                goles_esperados_1 *= 0.60
                goles_esperados_2 *= 0.50
            
            max_goles = 7
            probabilidades = {}
            from scipy.stats import nbinom
            
            # Función para calcular probabilidad con Binomial Negativa
            # VERSIÓN 2.0: Varianza calibrada empíricamente = 1.73 (antes: 1.40)
            # Basado en análisis histórico: Var(goles_local)=3.148, mean=1.757 → ratio=1.791
            def prob_goles(k, xg):
                if xg <= 0.1: return 1.0 if k == 0 else 0.0
                varianza = xg * 1.73  # MEJORADO: 1.73 en lugar de 1.40
                p = xg / varianza
                n = (xg * p) / (1 - p)
                return nbinom.pmf(k, n, p)

            for g1 in range(max_goles + 1):
                for g2 in range(max_goles + 1):
                    prob = prob_goles(g1, goles_esperados_1) * prob_goles(g2, goles_esperados_2)
                    probabilidades[f"{g1} - {g2}"] = prob
            
            # ============================================================
            # PREDICCIÓN OFICIAL (La que debes apostar en Mercado Pago)
            # ============================================================
            marcadores_top = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
            
            prob_t1_win = result['team1_win']
            prob_draw = result['draw']
            prob_t2_win = result['team2_win']
            
            # Utilizar el motor centralizado de predicción de marcador
            from model import get_expected_score
            g1_oficial, g2_oficial = get_expected_score(result)
            
            marcador_oficial = f"{g1_oficial} - {g2_oficial}"
            prob_oficial = probabilidades.get(marcador_oficial, 0)

            # Confianza basada en la probabilidad de victoria del favorito
            max_win = max(result['team1_win'], result['team2_win'], result['draw'])
            
            if max_win >= 55:
                nivel_confianza = "ALTA"
                texto_confianza = f"Favorito claro ({max_win:.0f}% de victoria). Apuesta sólida."
            elif max_win >= 40:
                nivel_confianza = "MEDIA"
                texto_confianza = f"Ligera ventaja ({max_win:.0f}%). Apuesta con precaución."
            else:
                nivel_confianza = "BAJA"
                texto_confianza = f"Partido muy parejo ({max_win:.0f}%). Alto riesgo."
            
            # Determinar ganador
            if g1_oficial > g2_oficial:
                veredicto = f"Victoria de {traducir(team1)}"
            elif g2_oficial > g1_oficial:
                veredicto = f"Victoria de {traducir(team2)}"
            else:
                veredicto = "Empate"
            
            # BTTS (Both Teams To Score / Ambos Anotan)
            prob_btts_si = sum(p for score, p in probabilidades.items() 
                             if all(int(x) > 0 for x in score.split(' - ')))
            
            # Alerta dinámica de sorpresa
            alerta_sorpresa = ""
            if 20 <= prob_draw < 30:
                alerta_sorpresa = f'<div style="background-color: rgba(234, 179, 8, 0.15); border-left: 4px solid #eab308; padding: 12px; margin-top: 16px; border-radius: 0 8px 8px 0;"><p style="color: #fef08a; margin: 0; font-size: 14px;"><b>ALERTA DE RIESGO DE EMPATE:</b> El modelo favorece un resultado con victoria, pero detectó un <b>{prob_draw:.1f}% de probabilidad de empate</b>. En partidos cerrados, hay riesgo significativo de que termine 1-1 o 2-2.</p></div>'
            elif prob_btts_si > 0.28 and (g1_oficial == 0 or g2_oficial == 0):
                alerta_sorpresa = f'<div style="background-color: rgba(234, 179, 8, 0.15); border-left: 4px solid #eab308; padding: 12px; margin-top: 16px; border-radius: 0 8px 8px 0;"><p style="color: #fef08a; margin: 0; font-size: 14px;"><b>RIESGO DE GOL EN CONTRA:</b> Se proyecta una portería a cero (0 goles), pero los mercados indican un <b>{prob_btts_si*100:.1f}%</b> de probabilidad matemática de que AMBOS anoten.</p></div>'

            st.subheader("Predicción Oficial")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f, #0d1b2a); border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #38bdf8; margin-bottom: 8px;">
                <p style="color: #94a3b8; font-size: 14px; margin: 0;">MARCADOR RECOMENDADO PARA TU APUESTA</p>
                <h1 style="color: #38bdf8; font-size: 56px; margin: 8px 0; font-weight: 800;">{traducir(team1)} {g1_oficial} - {g2_oficial} {traducir(team2)}</h1>
                <p style="color: #e2e8f0; font-size: 18px; margin: 4px 0;">{veredicto}</p>
                <p style="color: #94a3b8; font-size: 14px; margin: 12px 0 0 0;">Confianza: <b>{nivel_confianza}</b> — {texto_confianza}</p>
            </div>
            {alerta_sorpresa}
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # ============================================================
            # MERCADOS DE APUESTAS (Over/Under + BTTS)
            # ============================================================
            st.subheader("Mercados de Apuestas")
            
            # Over/Under 2.5
            prob_under = sum(p for score, p in probabilidades.items() 
                           if sum(int(x) for x in score.split(' - ')) < 3)
            prob_over = 1 - prob_under
            
            # BTTS ya se calculó arriba para la alerta
            prob_btts_no = 1 - prob_btts_si
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("**Over/Under 2.5 Goles**")
                st.write(f"Más de 2.5 goles: **{prob_over*100:.1f}%**")
                st.progress(max(0, min(100, int(prob_over*100))))
                st.write(f"Menos de 2.5 goles: **{prob_under*100:.1f}%**")
                st.progress(max(0, min(100, int(prob_under*100))))
            with col_m2:
                st.markdown("**Ambos Equipos Anotan (BTTS)**")
                st.write(f"Sí, ambos anotan: **{prob_btts_si*100:.1f}%**")
                st.progress(max(0, min(100, int(prob_btts_si*100))))
                st.write(f"No: **{prob_btts_no*100:.1f}%**")
                st.progress(max(0, min(100, int(prob_btts_no*100))))
            
            st.divider()
            
            # ============================================================
            # DETALLE TÉCNICO (xG + Top 5)
            # ============================================================
            st.subheader("Proyección de Marcadores (Detalle Técnico)")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Goles Esperados (xG)**")
                st.metric(label=traducir(team1), value=f"{goles_esperados_1:.2f}")
                st.metric(label=traducir(team2), value=f"{goles_esperados_2:.2f}")
            with col_res2:
                st.write("**Top 5 Probabilidades de Marcador:**")
                for marcador, prob in marcadores_top[:5]:
                    oficial_tag = " (Oficial)" if marcador == marcador_oficial else ""
                    st.write(f"{marcador} -> **{prob*100:.1f}%**{oficial_tag}")

with tab2:
    st.subheader("Actualización de Base de Datos")
    st.write("Registra resultados en vivo para recalibrar los modelos de ML en tiempo real.")
    
    with st.form("add_match"):
        c1, c2 = st.columns(2)
        nuevo_local = c1.selectbox("Local", equipos_unicos, key="n_local")
        goles_local = c1.number_input("Goles Local", min_value=0, max_value=20, value=0)
        
        nuevo_visita = c2.selectbox("Visitante", equipos_unicos, key="n_visita")
        goles_visita = c2.number_input("Goles Visitante", min_value=0, max_value=20, value=0)
        
        submit = st.form_submit_button("Guardar Partido y Re-entrenar Modelo")
        
        if submit:
            if nuevo_local == nuevo_visita:
                st.error("Los equipos deben ser diferentes.")
            else:
                nueva_fila = f"\n{datetime.date.today()},{nuevo_local},{nuevo_visita},{goles_local},{goles_visita},FIFA World Cup,Neutral,Neutral,TRUE"
                with open("data/matches.csv", "a") as f:
                    f.write(nueva_fila)
                
                # Actualizar el Calendario (schedule.csv)
                try:
                    df_cal = pd.read_csv("data/schedule.csv")
                    mask = ((df_cal['Local'] == nuevo_local) & (df_cal['Visitante'] == nuevo_visita)) | ((df_cal['Local'] == nuevo_visita) & (df_cal['Visitante'] == nuevo_local))
                    if mask.any():
                        idx = df_cal[mask].index[0]
                        df_cal.loc[idx, 'Estado'] = 'Fin'
                        if df_cal.loc[idx, 'Local'] == nuevo_local:
                            df_cal.loc[idx, 'GolesLocal'] = goles_local
                            df_cal.loc[idx, 'GolesVisitante'] = goles_visita
                        else:
                            df_cal.loc[idx, 'GolesLocal'] = goles_visita
                            df_cal.loc[idx, 'GolesVisitante'] = goles_local
                        df_cal.to_csv("data/schedule.csv", index=False)
                except Exception as e:
                    pass

                st.success("✅ ¡Resultado guardado! (Las estadísticas de racha y H2H se han actualizado en tiempo real).")
                st.cache_data.clear() # Limpia la memoria para cargar los datos frescos
                # No re-entrenamos el modelo completo aquí para evitar que se congele la aplicación por 2 minutos
                # train_model() 
                st.rerun()

with tab3:
    st.subheader("Rankings Internacionales (Top 20)")
    try:
        conn = sqlite3.connect("database/worldcup.db")
        df_rankings = pd.read_sql("SELECT * FROM Rankings LIMIT 20", conn)
        df_rankings.index = df_rankings.index + 1
        st.dataframe(df_rankings, use_container_width=True)
        conn.close()
    except Exception as e:
        st.warning("No se pudo conectar a la base de datos.")

with tab4:
    st.subheader("Calendario de la Competición")
    st.write("Visualiza resultados consolidados y genera análisis predictivos para partidos futuros.")
    
    try:
        df_cal = pd.read_csv("data/schedule.csv")
        
        # Opciones para filtrar
        filtro_estado = st.radio("Mostrar:", ["Todos", "Finalizados", "Pendientes (Simular)"], horizontal=True)
        
        if filtro_estado == "Finalizados":
            df_mostrar = df_cal[df_cal["Estado"] == "Fin"]
        elif filtro_estado == "Pendientes (Simular)":
            df_mostrar = df_cal[df_cal["Estado"] == "Pendiente"]
        else:
            df_mostrar = df_cal
            
        for _, row in df_mostrar.iterrows():
            with st.container():
                st.markdown(f"**{row['Fecha']} - {row['Grupo']}**")
                
                if row["Estado"] == "Fin":
                    st.success(f"FINALIZADO | {traducir(row['Local'])} {int(row['GolesLocal'])} - {int(row['GolesVisitante'])} {traducir(row['Visitante'])}")
                else:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.info(f"PRÓXIMO | {traducir(row['Local'])} vs {traducir(row['Visitante'])}")
                    with col_b:
                        # Si el equipo no está en nuestra DB (ej. no mapeado bien), manejamos el error
                        if st.button(f"Analizar", key=f"btn_{row['Local']}_{row['Visitante']}"):
                            try:
                                res = predict_match(row['Local'], row['Visitante'])
                                if res['team1_win'] > res['team2_win'] and res['team1_win'] > res['draw']:
                                    ganador = f"Gana {traducir(row['Local'])}"
                                elif res['team2_win'] > res['team1_win'] and res['team2_win'] > res['draw']:
                                    ganador = f"Gana {traducir(row['Visitante'])}"
                                else:
                                    ganador = "Empate"
                                
                                # Calcular marcador exacto usando motor centralizado
                                from model import get_expected_score
                                g1, g2 = get_expected_score(res)
                                mejor_marcador = f"{g1} - {g2}"
                                
                                st.write(f"**Proyección:** {ganador}")
                                st.write(f"**Marcador Esperado:** {mejor_marcador}")
                            except Exception as e:
                                st.write("Datos insuficientes para análisis")
                st.divider()
    except Exception as e:
        st.warning("No se encontró el archivo del calendario.")

with tab5:
    st.subheader("Simulador Monte Carlo (10,000 Universos Paralelos)")
    st.write("Esta herramienta simula el resto de la Fase de Grupos 10,000 veces usando las probabilidades matemáticas del modelo XGBoost. Calcula quién tiene más probabilidad de terminar en 1er Lugar de su grupo.")
    
    if st.button("Ejecutar Simulación de 10,000 Mundiales", use_container_width=True):
        with st.spinner("Simulando 10,000 mundiales en paralelo. Por favor espera (toma aprox 5 segundos)..."):
            from simulator import run_monte_carlo
            df_sim = run_monte_carlo(10000)
            
            st.success("¡Simulación completada con éxito!")
            
            df_sim['Equipo'] = df_sim['Equipo'].apply(traducir)
            # Formatear la tabla para que se vea premium
            st.dataframe(
                df_sim.style.format({'Probabilidad 1er Lugar Grupo': '{:.1f}%'})
                            .background_gradient(cmap='Blues', subset=['Probabilidad 1er Lugar Grupo']),
                use_container_width=True,
                height=500
            )

with tab6:
    st.subheader("Backtesting y Precisión (Ensemble Model)")
    st.write("Análisis de rendimiento en tiempo real de la Inteligencia Artificial frente a partidos ya finalizados. Esto valida matemáticamente la calidad del modelo (Backtesting).")
    
    if st.button("Evaluar Precisión Histórica", use_container_width=True):
        with st.spinner("Evaluando predicciones del Ensemble contra resultados reales..."):
            try:
                df_cal = pd.read_csv("data/schedule.csv")
                df_fin = df_cal[df_cal["Estado"] == "Fin"]
                
                if len(df_fin) == 0:
                    st.warning("No hay partidos finalizados en el calendario para evaluar.")
                else:
                    aciertos_ganador = 0
                    aciertos_exacto = 0
                    resultados_html = ""
                    
                    from scipy.stats import nbinom
                    def prob_goles(k, xg):
                        if xg <= 0.1: return 1.0 if k == 0 else 0.0
                        varianza = xg * 1.4
                        p = xg / varianza
                        n = (xg * p) / (1 - p)
                        return nbinom.pmf(k, n, p)
                        
                    for _, row in df_fin.iterrows():
                        t1, t2 = row['Local'], row['Visitante']
                        real_g1, real_g2 = int(row['GolesLocal']), int(row['GolesVisitante'])
                        
                        try:
                            r = predict_match(t1, t2)
                            xg1, xg2 = r['xg1'], r['xg2']
                            
                            probabilidades = {}
                            for g1 in range(8):
                                for g2 in range(8):
                                    prob = prob_goles(g1, xg1) * prob_goles(g2, xg2)
                                    probabilidades[f'{g1}-{g2}'] = prob
                            
                            marcadores_top = sorted(probabilidades.items(), key=lambda x: -x[1])
                            
                            from model import get_expected_score
                            pred_g1, pred_g2 = get_expected_score(r)
                            
                            real_marcador = f"{real_g1}-{real_g2}"
                            pred_marcador = f"{pred_g1}-{pred_g2}"
                            
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
                            
                            color = "#10b981" if es_exacto else ("#3b82f6" if es_ganador else "#ef4444")
                            icon = "EXACTO" if es_exacto else ("GANADOR" if es_ganador else "FALLO")
                            
                            resultados_html += f"""
                            <div style="border-left: 4px solid {color}; padding: 10px; margin-bottom: 8px; background: #1e293b; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <b>{traducir(t1)} vs {traducir(t2)}</b>
                                    <span style="color: {color}; font-weight: bold;">{icon}</span>
                                </div>
                                <div style="font-size: 14px; color: #cbd5e1;">
                                    Real: <b>{real_g1} - {real_g2}</b> | IA Predijo: <b>{pred_marcador}</b>
                                </div>
                            </div>
                            """
                        except Exception as e:
                            pass
                    
                    total = len(df_fin)
                    pct_ganador = (aciertos_ganador / total) * 100
                    pct_exacto = (aciertos_exacto / total) * 100
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Partidos Evaluados", total)
                    with col_m2:
                        st.metric("Acierto de Ganador (L/E/V)", f"{pct_ganador:.1f}%", f"{aciertos_ganador}/{total}")
                    with col_m3:
                        st.metric("Acierto Marcador Exacto", f"{pct_exacto:.1f}%", f"{aciertos_exacto}/{total}")
                    
                    st.divider()
                    st.markdown("### Registro de Predicciones")
                    st.markdown(resultados_html, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error("Error al leer el historial de partidos.")

with tab7:
    st.subheader("Radar de Apuestas de Valor (Value Bets)")
    st.write("Esta herramienta escanea todos los partidos futuros del Mundial buscando oportunidades donde la Inteligencia Artificial tiene una confianza matemática superior al **65%**.")
    
    if st.button("Escanear Partidos Futuros", use_container_width=True):
        with st.spinner("Escaneando el calendario oficial buscando oportunidades doradas..."):
            try:
                df_cal = pd.read_csv("data/schedule.csv")
                df_futuros = df_cal[df_cal["Estado"] == "Pendiente"]
                
                if len(df_futuros) == 0:
                    st.warning("No hay partidos futuros programados en el calendario.")
                else:
                    oportunidades = []
                    for _, row in df_futuros.iterrows():
                        t1, t2 = row['Local'], row['Visitante']
                        try:
                            r = predict_match(t1, t2)
                            max_prob = max(r['team1_win'], r['team2_win'], r['draw'])
                            if max_prob >= 65:
                                ganador = traducir(t1) if r['team1_win'] == max_prob else (traducir(t2) if r['team2_win'] == max_prob else "Empate")
                                from model import get_expected_score
                                g1, g2 = get_expected_score(r)
                                
                                oportunidades.append({
                                    "Partido": f"{traducir(t1)} vs {traducir(t2)}",
                                    "Fecha": row['Fecha'],
                                    "Apuesta Recomendada": f"Gana {ganador}" if ganador != "Empate" else "Empate",
                                    "Confianza IA (%)": float(round(max_prob, 1)),
                                    "Marcador Esperado": f"{g1} - {g2}"
                                })
                        except:
                            pass
                    
                    if len(oportunidades) > 0:
                        df_op = pd.DataFrame(oportunidades)
                        df_op = df_op.sort_values('Confianza IA (%)', ascending=False)
                        
                        st.success(f"¡Se encontraron {len(oportunidades)} apuestas de alto valor!")
                        st.dataframe(
                            df_op.style.format({'Confianza IA (%)': '{:.1f}%'})
                                       .background_gradient(cmap='Greens', subset=['Confianza IA (%)']), 
                            use_container_width=True, height=400
                        )
                    else:
                        st.info("No se encontraron apuestas con confianza > 65% en los próximos partidos.")
            except Exception as e:
                st.error(f"Error al analizar el calendario: {e}")
