import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from backtest import parse_schedule_date
from model import get_elos, predict_match, register_live_match

st.set_page_config(page_title="World Cup Analytics 2026", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
st.title("World Cup Analytics 2026")
st.markdown("Plataforma profesional de análisis deportivo y estimación de probabilidades.")

try:
    elos = get_elos()
    equipos_unicos = sorted(list(elos.keys()))
except Exception as e:
    st.error(f"Error cargando el modelo: {e}. Asegúrate de ejecutar model.py primero.")
    st.stop()

TRADUCCIONES = {
    "Spain": "España", "France": "Francia", "England": "Inglaterra", "Brazil": "Brasil",
    "Germany": "Alemania", "Netherlands": "Países Bajos", "South Korea": "Corea del Sur",
    "South Africa": "Sudáfrica", "IR Iran": "Irán", "New Zealand": "Nueva Zelanda",
    "Saudi Arabia": "Arabia Saudita", "Morocco": "Marruecos", "Czech Republic": "Chequia",
    "Switzerland": "Suiza", "Japan": "Japón", "Cabo Verde": "Cabo Verde", "DR Congo": "RD Congo",
    "Congo DR": "RD Congo", "United States": "Estados Unidos", "USA": "Estados Unidos",
    "Sweden": "Suecia", "Turkey": "Turquía", "Ivory Coast": "Costa de Marfil",
}


def traducir(nombre):
    return TRADUCCIONES.get(nombre, nombre)


def confidence_badge(max_win):
    if max_win >= 65:
        return "STRONG LEAN 🟢", f"Claro favorito ({max_win:.0f}%). Baja varianza esperada."
    if max_win >= 50:
        return "SLIGHT LEAN 🟡", f"Ligera ventaja ({max_win:.0f}%). Partido competido."
    if max_win >= 40:
        return "HIGH VARIANCE 🟠", f"Partido caótico ({max_win:.0f}%). Cualquier cosa puede pasar."
    return "NO BET / RISKY 🔴", f"Extremadamente incierto ({max_win:.0f}%). Riesgo altísimo."


tabs = st.tabs([
    "Análisis de Partido", "Resultados en Vivo", "Rankings", "Calendario",
    "Simulador Monte Carlo", "Precisión", "Radar Apuestas",
])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("Equipo Local", equipos_unicos, index=equipos_unicos.index("Argentina") if "Argentina" in equipos_unicos else 0, format_func=traducir)
        lesion1 = st.checkbox(f"Jugador Clave de {traducir(team1)} ausente", key="les1")
        defensivo1 = st.checkbox(f"{traducir(team1)} con planteamiento defensivo", key="def1")
    with c2:
        team2 = st.selectbox("Equipo Visitante", equipos_unicos, index=equipos_unicos.index("France") if "France" in equipos_unicos else 1, format_func=traducir)
        lesion2 = st.checkbox(f"Jugador Clave de {traducir(team2)} ausente", key="les2")
        defensivo2 = st.checkbox(f"{traducir(team2)} con planteamiento defensivo", key="def2")

    if st.button("Generar Análisis de Probabilidades", use_container_width=True):
        if team1 == team2:
            st.warning("Selecciona dos equipos distintos.")
        else:
            mod1 = (0.75 if lesion1 else 1.0) * (0.75 if defensivo1 else 1.0) * (0.65 if defensivo2 else 1.0)
            mod2 = (0.75 if lesion2 else 1.0) * (0.75 if defensivo2 else 1.0) * (0.65 if defensivo1 else 1.0)
            res = predict_match(team1, team2, mod_xg1=mod1, mod_xg2=mod2)
            if mod1 != 1.0 or mod2 != 1.0:
                st.info(f"Ajuste táctico aplicado al motor Monte Carlo y Ensemble (xG {traducir(team1)} ×{mod1:.2f} | xG {traducir(team2)} ×{mod2:.2f}).")

            st.divider()
            st.subheader("Ficha Técnica del Encuentro")
            st.markdown(f"**ELO Dinámico:** {traducir(team1)} ({res['elo1']}) | {traducir(team2)} ({res['elo2']})")
            st.markdown(f"**Valor de Plantilla:** {traducir(team1)} (€{res['mv1']}M) | {traducir(team2)} (€{res['mv2']}M)")
            st.markdown(f"**Estado de Forma (Últimos 5):** {traducir(team1)} ({res['streak1']} pts) | {traducir(team2)} ({res['streak2']} pts)")
            st.markdown(f"**Historial Directo (H2H):** Victorias {traducir(team1)}: {res['h2h1']} | Victorias {traducir(team2)}: {res['h2h2']}")
            st.markdown(f"**Diferencia de Goles Reciente:** {traducir(team1)} ({res['gd1']:+d}) | {traducir(team2)} ({res['gd2']:+d})")

            st.divider()
            st.subheader("Radar Táctico y de Fuerzas")
            fig_radar = go.Figure()
            for idx, (tm, color) in enumerate([(team1, "#1f77b4"), (team2, "#ff7f0e")], start=1):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[res[f"elo{idx}"], res[f"mv{idx}"] * 2, res.get(f"avg_gf{idx}", 1) * 500, (3 - res.get(f"avg_ga{idx}", 1)) * 500, res[f"poss{idx}"] * 20],
                    theta=["ELO", "Valor Plantilla", "Ataque", "Defensa", "Posesión"], fill="toself", name=traducir(tm), marker=dict(color=color),
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=400, margin=dict(t=30, b=30, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_radar, use_container_width=True)

            st.divider()
            st.subheader("Cuotas de Victoria Implícitas")
            fig = go.Figure(data=[go.Pie(labels=[f"Victoria {traducir(team1)}", "Empate", f"Victoria {traducir(team2)}"], values=[res["team1_win"], res["draw"], res["team2_win"]], hole=0.4, marker_colors=["#1f77b4", "#7f7f7f", "#ff7f0e"])])
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Proyección de Posesión de Balón")
            cp1, cp2 = st.columns(2)
            for col, tm, key in [(cp1, team1, "poss1"), (cp2, team2, "poss2")]:
                with col:
                    st.metric(label=traducir(tm), value=f"{res[key]:.1f}%")
                    st.progress(max(0, min(100, int(res[key]))))

            st.divider()
            g1_of, g2_of = map(int, res["official_score"].split("-"))
            prob_draw, prob_btts_si = res["draw"], res["btts"] / 100.0
            nivel_conf, texto_conf = confidence_badge(max(res["team1_win"], res["team2_win"], prob_draw))
            veredicto = f"Victoria de {traducir(team1)}" if g1_of > g2_of else (f"Victoria de {traducir(team2)}" if g2_of > g1_of else "Empate")

            alerta = ""
            if 20 <= prob_draw < 30:
                alerta = f'<div style="background-color: rgba(234,179,8,0.15); border-left: 4px solid #eab308; padding: 12px; margin-top: 16px; border-radius: 0 8px 8px 0;"><p style="color: #fef08a; margin: 0; font-size: 14px;"><b>ALERTA DE RIESGO DE EMPATE:</b> Se detectó un <b>{prob_draw:.1f}% de probabilidad de empate</b>.</p></div>'
            elif prob_btts_si > 0.28 and (g1_of == 0 or g2_of == 0):
                alerta = f'<div style="background-color: rgba(234,179,8,0.15); border-left: 4px solid #eab308; padding: 12px; margin-top: 16px; border-radius: 0 8px 8px 0;"><p style="color: #fef08a; margin: 0; font-size: 14px;"><b>RIESGO DE GOL EN CONTRA:</b> Se proyecta portería a cero, pero hay <b>{prob_btts_si*100:.1f}%</b> de probabilidad de que ambos anoten.</p></div>'

            st.subheader("Predicción Oficial")
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #1e3a5f, #0d1b2a); border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #38bdf8; margin-bottom: 8px;">'
                f'<p style="color: #94a3b8; font-size: 14px; margin: 0;">MARCADOR RECOMENDADO PARA TU APUESTA</p>'
                f'<h1 style="color: #38bdf8; font-size: 56px; margin: 8px 0; font-weight: 800;">{traducir(team1)} {g1_of} - {g2_of} {traducir(team2)}</h1>'
                f'<p style="color: #e2e8f0; font-size: 18px; margin: 4px 0;">{veredicto}</p>'
                f'<p style="color: #94a3b8; font-size: 14px; margin: 12px 0 0 0;">Confianza: <b>{nivel_conf}</b> — {texto_conf}</p></div>{alerta}',
                unsafe_allow_html=True,
            )

            st.subheader("Mercados de Apuestas")
            prob_over = res["over_25"] / 100.0
            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown("**Over/Under 2.5 Goles**")
                st.write(f"Más de 2.5 goles: **{prob_over*100:.1f}%**"); st.progress(max(0, min(100, int(prob_over * 100))))
                st.write(f"Menos de 2.5 goles: **{(1-prob_over)*100:.1f}%**"); st.progress(max(0, min(100, int((1 - prob_over) * 100))))
            with cm2:
                st.markdown("**Ambos Equipos Anotan (BTTS)**")
                st.write(f"Sí, ambos anotan: **{prob_btts_si*100:.1f}%**"); st.progress(max(0, min(100, int(prob_btts_si * 100))))
                st.write(f"No: **{(1-prob_btts_si)*100:.1f}%**"); st.progress(max(0, min(100, int((1 - prob_btts_si) * 100))))

            st.divider()
            st.subheader("Proyección de Marcadores (Detalle Técnico)")
            cr1, cr2 = st.columns(2)
            with cr1:
                st.write("**Goles Esperados (xG Ajustado)**")
                st.metric(label=traducir(team1), value=f"{res['xg1']:.2f}")
                st.metric(label=traducir(team2), value=f"{res['xg2']:.2f}")
            with cr2:
                st.write("**Top 3 Escenarios Más Probables (Monte Carlo):**")
                for marcador, prob in res["top_3_scenarios"]:
                    tag = " (Recomendado ⭐)" if marcador == res["official_score"] else ""
                    st.write(f"**{marcador}** -> **{prob:.1f}%**{tag}")

with tabs[1]:
    st.subheader("Actualización de Base de Datos")
    st.write("Registra resultados en vivo para actualizar ELO, rachas y H2H en tiempo real.")
    with st.form("add_match"):
        c1, c2 = st.columns(2)
        nuevo_local = c1.selectbox("Local", equipos_unicos, key="n_local")
        goles_local = c1.number_input("Goles Local", min_value=0, max_value=20, value=0)
        nuevo_visita = c2.selectbox("Visitante", equipos_unicos, key="n_visita")
        goles_visita = c2.number_input("Goles Visitante", min_value=0, max_value=20, value=0)
        if st.form_submit_button("Guardar Partido y Actualizar ELO"):
            if nuevo_local == nuevo_visita:
                st.error("Los equipos deben ser diferentes.")
            else:
                register_live_match(nuevo_local, nuevo_visita, goles_local, goles_visita)
                st.success("✅ ¡Resultado guardado! El ELO, las rachas y el H2H se actualizaron al instante.")
                st.cache_data.clear()
                st.rerun()

with tabs[2]:
    st.subheader("Rankings Internacionales (Top 20)")
    try:
        with sqlite3.connect("database/worldcup.db") as conn:
            df_rankings = pd.read_sql("SELECT * FROM Rankings LIMIT 20", conn)
        df_rankings.index = df_rankings.index + 1
        st.dataframe(df_rankings, use_container_width=True)
    except Exception:
        st.warning("No se pudo conectar a la base de datos.")

with tabs[3]:
    st.subheader("Calendario de la Competición")
    try:
        df_cal = pd.read_csv("data/schedule.csv")
        filtro = st.radio("Mostrar:", ["Todos", "Finalizados", "Pendientes (Simular)"], horizontal=True)
        df_show = df_cal[df_cal["Estado"] == "Fin"] if filtro == "Finalizados" else (df_cal[df_cal["Estado"] == "Pendiente"] if filtro == "Pendientes (Simular)" else df_cal)
        if len(df_show) == 0:
            st.info("No hay partidos en esta categoría.")
        for idx, row in df_show.iterrows():
            st.markdown(f"**{row['Fecha']} - {row['Grupo']}**")
            ca, cb = st.columns([3, 1])
            with ca:
                if row["Estado"] == "Fin":
                    st.success(f"FINALIZADO | {traducir(row['Local'])} {int(row['GolesLocal'])} - {int(row['GolesVisitante'])} {traducir(row['Visitante'])}")
                else:
                    st.info(f"PRÓXIMO | {traducir(row['Local'])} vs {traducir(row['Visitante'])}")
            with cb:
                btn_lbl = "Analizar Pre-Partido" if row["Estado"] == "Fin" else "Analizar"
                if st.button(btn_lbl, key=f"btn_{idx}_{row['Local']}_{row['Visitante']}"):
                    try:
                        as_of = parse_schedule_date(row["Fecha"]) if row["Estado"] == "Fin" else None
                        r = predict_match(row["Local"], row["Visitante"], as_of_date=as_of)
                        gan = f"Gana {traducir(row['Local'])}" if r["team1_win"] > max(r["team2_win"], r["draw"]) else (f"Gana {traducir(row['Visitante'])}" if r["team2_win"] > max(r["team1_win"], r["draw"]) else "Empate")
                        st.write(f"**Proyección:** {gan} ({r['official_score']})")
                    except Exception:
                        st.write("Datos insuficientes")
            st.divider()
    except Exception:
        st.warning("No se encontró el archivo del calendario.")

with tabs[4]:
    st.subheader("Simulador Monte Carlo (10,000 Universos Paralelos)")
    st.write("Simula la Fase de Grupos 10,000 veces con goles Bivariate Poisson y desempate FIFA (Puntos → DG → GF). Si la fase ya concluyó, simula el torneo desde probabilidades pre-torneo.")
    if st.button("Ejecutar Simulación de 10,000 Mundiales", use_container_width=True):
        with st.spinner("Simulando 10,000 mundiales..."):
            from simulator import run_monte_carlo
            df_sim = run_monte_carlo(10000)
            df_sim["Equipo"] = df_sim["Equipo"].apply(traducir)
            st.success("¡Simulación completada con éxito!")
            st.dataframe(
                df_sim.style.format({"Puntos Esperados": "{:.2f}", "Probabilidad 1er Lugar Grupo": "{:.1f}%", "Probabilidad Top 2 Grupo": "{:.1f}%"})
                .background_gradient(cmap="Blues", subset=["Probabilidad 1er Lugar Grupo", "Probabilidad Top 2 Grupo"]),
                use_container_width=True, height=500,
            )

with tabs[5]:
    st.subheader("Backtesting y Precisión (Ensemble Model)")
    st.write("Evaluación estricta sin data leakage frente a partidos ya finalizados.")
    if st.button("Evaluar Precisión Histórica", use_container_width=True):
        with st.spinner("Evaluando predicciones..."):
            try:
                df_fin = pd.read_csv("data/schedule.csv")
                df_fin = df_fin[df_fin["Estado"] == "Fin"]
                if len(df_fin) == 0:
                    st.warning("No hay partidos finalizados en el calendario para evaluar.")
                else:
                    aciertos_gan, aciertos_ex, brier_list, html = 0, 0, [], ""
                    for _, row in df_fin.iterrows():
                        t1, t2, rg1, rg2 = row["Local"], row["Visitante"], int(row["GolesLocal"]), int(row["GolesVisitante"])
                        try:
                            r = predict_match(t1, t2, as_of_date=parse_schedule_date(row["Fecha"]))
                            p1, pd_p, p2 = r["team1_win"] / 100.0, r["draw"] / 100.0, r["team2_win"] / 100.0
                            tot = p1 + pd_p + p2
                            if tot > 0:
                                p1, pd_p, p2 = p1 / tot, pd_p / tot, p2 / tot
                            y1, yd, y2 = (1.0, 0.0, 0.0) if rg1 > rg2 else ((0.0, 0.0, 1.0) if rg1 < rg2 else (0.0, 1.0, 0.0))
                            brier_list.append(((p1 - y1) ** 2 + (pd_p - yd) ** 2 + (p2 - y2) ** 2) / 3.0)
                            pg1, pg2 = map(int, r["official_score"].split("-"))
                            es_gan = (rg1 > rg2 and pg1 > pg2) or (rg1 < rg2 and pg1 < pg2) or (rg1 == rg2 and pg1 == pg2)
                            es_ex = rg1 == pg1 and rg2 == pg2
                            aciertos_gan += int(es_gan); aciertos_ex += int(es_ex)
                            color = "#10b981" if es_ex else ("#3b82f6" if es_gan else "#ef4444")
                            icon = "EXACTO" if es_ex else ("GANADOR" if es_gan else "FALLO")
                            html += f'<div style="border-left: 4px solid {color}; padding: 10px; margin-bottom: 8px; background: #1e293b; border-radius: 4px;"><div style="display:flex;justify-content:space-between;"><b>{traducir(t1)} vs {traducir(t2)}</b><span style="color:{color};font-weight:bold;">{icon}</span></div><div style="font-size:14px;color:#cbd5e1;">Real: <b>{rg1} - {rg2}</b> | IA Predijo: <b>{r["official_score"]}</b> (1X2: {p1*100:.0f}% / {pd_p*100:.0f}% / {p2*100:.0f}%)</div></div>'
                        except Exception:
                            pass
                    n = len(brier_list)
                    if n == 0:
                        st.warning("No se pudieron evaluar los partidos finalizados.")
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Partidos Evaluados", n)
                        c2.metric("Acierto de Ganador", f"{(aciertos_gan/n)*100:.1f}%", f"{aciertos_gan}/{n}")
                        c3.metric("Acierto Exacto", f"{(aciertos_ex/n)*100:.1f}%", f"{aciertos_ex}/{n}")
                        c4.metric("Brier Score (1X2)", f"{float(np.mean(brier_list)):.4f}", "Objetivo: < 0.22")
                        st.divider()
                        st.markdown("### Registro de Predicciones (Sin Data Leakage)")
                        st.markdown(html, unsafe_allow_html=True)
            except Exception:
                st.error("Error al leer el historial de partidos.")

with tabs[6]:
    st.subheader("Radar de Apuestas de Valor (Value Bets)")
    st.write("Escanea partidos con confianza matemática superior al **65%**.")
    if st.button("Escanear Partidos", use_container_width=True):
        with st.spinner("Escaneando el calendario oficial..."):
            try:
                df_cal = pd.read_csv("data/schedule.csv")
                df_fut = df_cal[df_cal["Estado"] == "Pendiente"]
                retrospectivo = len(df_fut) == 0
                df_scan = df_cal if retrospectivo else df_fut
                if retrospectivo:
                    st.info("Torneo finalizado: escaneo retrospectivo pre-partido con confianza >= 65%.")
                ops = []
                for _, row in df_scan.iterrows():
                    t1, t2 = row["Local"], row["Visitante"]
                    try:
                        r = predict_match(t1, t2, as_of_date=parse_schedule_date(row["Fecha"]) if retrospectivo else None)
                        mp = max(r["team1_win"], r["team2_win"], r["draw"])
                        if mp >= 65:
                            gan = traducir(t1) if r["team1_win"] == mp else (traducir(t2) if r["team2_win"] == mp else "Empate")
                            ops.append({"Partido": f"{traducir(t1)} vs {traducir(t2)}", "Fecha": row["Fecha"], "Apuesta Recomendada": f"Gana {gan}" if gan != "Empate" else "Empate", "Confianza IA (%)": round(float(mp), 1), "Marcador Esperado": r["official_score"]})
                    except Exception:
                        pass
                if ops:
                    df_op = pd.DataFrame(ops).sort_values("Confianza IA (%)", ascending=False)
                    st.success(f"¡Se encontraron {len(ops)} oportunidades de alto valor!")
                    st.dataframe(df_op.style.format({"Confianza IA (%)": "{:.1f}%"}).background_gradient(cmap="Greens", subset=["Confianza IA (%)"]), use_container_width=True, height=400)
                else:
                    st.info("No se encontraron partidos con confianza >= 65%.")
            except Exception as e:
                st.error(f"Error al analizar el calendario: {e}")
