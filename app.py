"""
Interfaz Streamlit — Predicción de Pérdida Financiera
Dataset: Ciberseguridad LATAM 2021-2025
Variable objetivo: perdida_financiera_usd
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CiberRisk · Predictor LATAM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos personalizados ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Fondo y tipografía general */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Header principal */
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #E8F4FD;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 4px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8FB4CC;
    margin-bottom: 0;
}

/* Tarjeta de resultado */
.result-card {
    background: linear-gradient(135deg, #0F3460 0%, #16213E 100%);
    border: 1px solid #1A6EBD;
    border-radius: 16px;
    padding: 32px 28px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(26,110,189,0.2);
}
.result-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5AAFD4;
    margin-bottom: 8px;
}
.result-value {
    font-size: 2.8rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 4px;
}
.result-range {
    font-size: 0.82rem;
    color: #7BAEC5;
}

/* Tarjetas de contexto */
.info-card {
    background: #0D1B2A;
    border: 1px solid #1C3A55;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.info-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5AAFD4;
    margin-bottom: 2px;
}
.info-card .value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E8F4FD;
}

/* Separador de sección */
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4A8BAA;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1C3A55;
}

/* Badge de severidad */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-critica  { background: #3D0C0C; color: #FF6B6B; border: 1px solid #7A1A1A; }
.badge-alta     { background: #3D2600; color: #FFA040; border: 1px solid #7A4A00; }
.badge-media    { background: #1A2D00; color: #7ECB3F; border: 1px solid #3A5A00; }
.badge-baja     { background: #003A2D; color: #3ECFB0; border: 1px solid #006B50; }

/* Barra lateral */
section[data-testid="stSidebar"] {
    background: #060F1A;
    border-right: 1px solid #1C3A55;
}

/* Botón principal */
.stButton > button {
    background: linear-gradient(135deg, #1A6EBD 0%, #0D4A8A 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Métricas de factores */
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1C3A55;
    font-size: 0.88rem;
    color: #8FB4CC;
}
.factor-name { flex: 1; }
.factor-bar-bg {
    flex: 2;
    height: 6px;
    background: #1C3A55;
    border-radius: 3px;
    margin: 0 12px;
}
.factor-bar-fill {
    height: 6px;
    border-radius: 3px;
    background: #1A6EBD;
}
.factor-pct {
    width: 36px;
    text-align: right;
    font-size: 0.78rem;
    color: #5AAFD4;
}
</style>
""", unsafe_allow_html=True)

# ── Opciones de variables categóricas ────────────────────────────────────────
OPCIONES = {
    "año": [2021, 2022, 2023, 2024, 2025],
    "trimestre": ["Q1", "Q2", "Q3", "Q4"],
    "mes": list(range(1, 13)),
    "pais": [
        "Argentina", "Bolivia", "Brasil", "Chile", "Colombia",
        "Costa Rica", "Ecuador", "México", "Paraguay",
        "Perú", "Uruguay", "Venezuela",
    ],
    "sector": [
        "Agricultura", "Educación", "Energía", "Financiero",
        "Gobierno", "Manufactura", "Retail / Comercio",
        "Salud", "Telecomunicaciones", "Transporte",
    ],
    "tipo_ataque": [
        "DDoS", "Espionaje (APT)", "Extorsión sin cifrado",
        "Intrusión de red", "Malware", "Phishing",
        "RAT (Acceso Remoto)", "Ransomware",
        "Robo de credenciales", "Vulnerabilidad Web",
    ],
    "vector_ataque": [
        "Botnet", "Brecha de base de datos", "Cadena de suministro",
        "Credenciales robadas", "Descarga web", "Email",
        "Email phishing (IA-asistido)", "Email (spear phishing IA)",
        "Exfiltración de datos", "Firmware implant",
        "Fuerza bruta", "Ingeniería social",
        "Infostealer (LummaC2/Vidar)", "Keylogger",
        "Mercado negro de accesos", "RDP expuesto",
        "RDWeb portal", "Redes sociales",
        "Reflexión NTP", "SMS (Smishing)",
        "Software pirata", "SQL Injection",
        "Supply chain", "USB/Dispositivo externo",
        "VPN sin parches", "Vulnerabilidad 0-day",
        "Watering hole", "WhatsApp", "XSS",
    ],
    "severidad": ["Baja", "Media", "Alta", "Crítica"],
    "actor_amenaza": [
        "Akira", "Anonymous", "APT-C-36", "APT15", "APT28",
        "AsyncRAT", "BlackCat/ALPHV", "Blind Eagle",
        "Blind Eagle (Quasar RAT)", "Botnet regional",
        "Clop", "Conti", "Desconocido", "DragonForce",
        "GhostSec", "Grupo organizado local",
        "Hive", "Lazarus Group", "LockBit",
        "LummaC2", "Manipulated Caiman",
        "Nightspire", "Qilin", "Raccoon Stealer",
        "RansomHub", "Redline Stealer", "Remcos RAT",
        "SafePay", "ShinyHunters", "Silver Oryx Blade",
        "Stormous", "The Gentlemen",
        "Troyano bancario (Grandoreiro)", "Vidar Stealer",
    ],
    "tipo_impacto": [
        "Acceso no autorizado", "Cifrado de datos",
        "Daño reputacional", "Exposición de datos sensibles (GenAI)",
        "Filtración de datos", "Fraude financiero (NFC/móvil)",
        "Interrupción de servicios", "Pérdida financiera",
        "Robo de datos",
    ],
    "tamaño_organizacion": ["Pequeña (<50)", "Mediana (50-250)", "Grande (>250)"],
    "reportado_autoridades": ["Sí", "No"],
}

# Pesos de influencia por variable (para mostrar importancias relativas)
# Ajusta estos valores según feature_importances_ de tu modelo real
IMPORTANCIAS = {
    "severidad":                 0.31,
    "tipo_ataque":               0.18,
    "registros_comprometidos":   0.14,
    "sector":                    0.09,
    "tiempo_resolucion_horas":   0.08,
    "pais":                      0.06,
    "actor_amenaza":             0.05,
    "tamaño_organizacion":       0.04,
    "vector_ataque":             0.03,
    "tipo_impacto":              0.02,
}

# Rangos de pérdida por severidad (para estimar si no hay modelo cargado)
RANGOS_SEVERIDAD = {
    "Baja":    (500,    8_000),
    "Media":   (8_000,  70_000),
    "Alta":    (60_000, 700_000),
    "Crítica": (800_000, 15_000_000),
}

# ── Carga del modelo ─────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """Intenta cargar el modelo desde archivo .pkl."""
    rutas = ["modelo.pkl", "model.pkl", "modelo_rf.pkl", "model_xgb.pkl"]
    for ruta in rutas:
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return pickle.load(f), ruta
    return None, None

modelo, ruta_modelo = cargar_modelo()

# ── Función de predicción ────────────────────────────────────────────────────
def predecir(datos: dict) -> tuple[float, float, float]:
    """
    Devuelve (prediccion, limite_inferior, limite_superior).
    Si no hay modelo, usa una heurística basada en severidad.
    """
    if modelo is not None:
        try:
            df = pd.DataFrame([datos])
            pred = float(modelo.predict(df)[0])
            margen = pred * 0.20
            return pred, max(0, pred - margen), pred + margen
        except Exception as e:
            st.warning(f"Error al ejecutar el modelo: {e}. Usando estimación heurística.")

    # Heurística de demostración
    sev = datos.get("severidad", "Media")
    lo, hi = RANGOS_SEVERIDAD[sev]
    base = (lo + hi) / 2

    # Ajustes por sector
    mult_sector = {
        "Financiero": 1.4, "Gobierno": 1.2, "Salud": 1.25,
        "Manufactura": 1.15, "Energía": 1.3,
    }.get(datos.get("sector", ""), 1.0)

    # Ajuste por tipo de ataque
    mult_ataque = {
        "Ransomware": 1.35, "Espionaje (APT)": 1.5,
        "Extorsión sin cifrado": 1.2, "Intrusión de red": 1.1,
    }.get(datos.get("tipo_ataque", ""), 1.0)

    # Ajuste por tamaño
    mult_tam = {"Pequeña (<50)": 0.7, "Mediana (50-250)": 1.0, "Grande (>250)": 1.5}.get(
        datos.get("tamaño_organizacion", "Mediana (50-250)"), 1.0
    )

    # Ajuste por registros comprometidos
    reg = datos.get("registros_comprometidos", 1000)
    mult_reg = 1.0 + min(reg / 500_000, 0.5)

    pred = base * mult_sector * mult_ataque * mult_tam * mult_reg
    pred = max(lo, min(pred, hi * 1.5))
    margen = pred * 0.22
    return pred, max(0, pred - margen), pred + margen


def color_severidad(sev: str) -> str:
    return {
        "Crítica": "#FF6B6B",
        "Alta":    "#FFA040",
        "Media":   "#7ECB3F",
        "Baja":    "#3ECFB0",
    }.get(sev, "#FFFFFF")


def badge_severidad(sev: str) -> str:
    clase = f"badge-{sev.lower().replace('í','i')}"
    return f'<span class="badge {clase}">{sev}</span>'


# ── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<p class="hero-title">🛡️ CiberRisk · Predictor de Pérdida Financiera</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Estima el impacto económico de un incidente de ciberseguridad en América Latina · 2021–2025</p>', unsafe_allow_html=True)

if modelo:
    st.success(f"✅ Modelo cargado: `{ruta_modelo}`", icon="🤖")
else:
    st.info("ℹ️ No se encontró un archivo de modelo (.pkl). Se usará estimación heurística para demostraciones. "
            "Coloca tu modelo entrenado como `modelo.pkl` en el mismo directorio.", icon="💡")

st.divider()

# ── Layout principal ─────────────────────────────────────────────────────────
sidebar, col_main = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════════════════
# SIDEBAR — Inputs del incidente
# ══════════════════════════════════════════════════════
with sidebar:
    st.markdown('<p class="section-title">🗓 Contexto temporal</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    ano      = c1.selectbox("Año",       OPCIONES["año"],       index=4)
    trimestre = c2.selectbox("Trimestre", OPCIONES["trimestre"], index=0)
    mes       = st.selectbox("Mes",       OPCIONES["mes"],       index=0,
                             format_func=lambda m: [
                                 "Enero","Febrero","Marzo","Abril","Mayo","Junio",
                                 "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
                             ][m-1])

    st.markdown('<p class="section-title">🌎 Contexto geográfico y sectorial</p>', unsafe_allow_html=True)
    pais   = st.selectbox("País",   OPCIONES["pais"],   index=2)
    sector = st.selectbox("Sector", OPCIONES["sector"], index=4)
    tamaño = st.selectbox("Tamaño de la organización", OPCIONES["tamaño_organizacion"], index=1)

    st.markdown('<p class="section-title">⚔️ Características del ataque</p>', unsafe_allow_html=True)
    tipo_ataque   = st.selectbox("Tipo de ataque",   OPCIONES["tipo_ataque"],   index=7)
    vector_ataque = st.selectbox("Vector de ataque", OPCIONES["vector_ataque"], index=0)
    severidad     = st.selectbox("Severidad",        OPCIONES["severidad"],     index=2)
    actor_amenaza = st.selectbox("Actor de amenaza", OPCIONES["actor_amenaza"], index=6)
    tipo_impacto  = st.selectbox("Tipo de impacto",  OPCIONES["tipo_impacto"],  index=4)

    st.markdown('<p class="section-title">📊 Métricas del incidente</p>', unsafe_allow_html=True)
    registros    = st.number_input(
        "Registros comprometidos",
        min_value=0, max_value=1_000_000,
        value=5_000, step=500,
        help="Número de registros de datos expuestos o afectados"
    )
    tiempo_res   = st.number_input(
        "Tiempo de resolución (horas)",
        min_value=0.0, max_value=2_000.0,
        value=48.0, step=0.5,
        help="Horas transcurridas hasta resolver el incidente"
    )
    reportado    = st.selectbox("¿Reportado a autoridades?", OPCIONES["reportado_autoridades"])

    st.markdown("")
    predecir_btn = st.button("🔍 Estimar pérdida financiera", use_container_width=True)

# ══════════════════════════════════════════════════════
# PANEL PRINCIPAL — Resultados
# ══════════════════════════════════════════════════════
with col_main:

    # Resumen del incidente configurado
    st.markdown('<p class="section-title">📋 Incidente configurado</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">País</div>
            <div class="value">🌎 {pais}</div>
        </div>
        <div class="info-card">
            <div class="label">Sector</div>
            <div class="value">🏢 {sector}</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Tipo de ataque</div>
            <div class="value">⚔️ {tipo_ataque}</div>
        </div>
        <div class="info-card">
            <div class="label">Severidad</div>
            <div class="value">{badge_severidad(severidad)}</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Actor</div>
            <div class="value">🎭 {actor_amenaza}</div>
        </div>
        <div class="info-card">
            <div class="label">Organización</div>
            <div class="value">🏗️ {tamaño}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Resultado de predicción ───────────────────────────────────────────
    if predecir_btn or True:  # Mostrar resultado siempre para mejor UX
        datos_entrada = {
            "año":                    ano,
            "trimestre":              trimestre,
            "mes":                    mes,
            "pais":                   pais,
            "sector":                 sector,
            "tipo_ataque":            tipo_ataque,
            "vector_ataque":          vector_ataque,
            "severidad":              severidad,
            "actor_amenaza":          actor_amenaza,
            "tipo_impacto":           tipo_impacto,
            "registros_comprometidos": registros,
            "tiempo_resolucion_horas": tiempo_res,
            "tamaño_organizacion":    tamaño,
            "reportado_autoridades":  reportado,
        }

        pred, lo_pred, hi_pred = predecir(datos_entrada)

        st.markdown('<p class="section-title">💰 Estimación de pérdida financiera</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Pérdida financiera estimada (USD)</div>
            <div class="result-value">$ {pred:,.0f}</div>
            <div class="result-range">
                Rango estimado: $ {lo_pred:,.0f} — $ {hi_pred:,.0f}
                &nbsp;·&nbsp; Incertidumbre ±20%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ── Factores de influencia ────────────────────────────────────────
        st.markdown('<p class="section-title">🔬 Importancia de variables en el modelo</p>', unsafe_allow_html=True)

        # Si el modelo tiene feature_importances_, úsalas; si no, usa las predefinidas
        importancias_vis = IMPORTANCIAS
        if modelo is not None and hasattr(modelo, "feature_importances_"):
            try:
                feats = modelo.feature_names_in_
                imps  = modelo.feature_importances_
                importancias_vis = dict(zip(feats, imps))
            except Exception:
                pass

        total = sum(importancias_vis.values())
        items_ordenados = sorted(importancias_vis.items(), key=lambda x: x[1], reverse=True)[:10]

        html_bars = ""
        for nombre, imp in items_ordenados:
            pct  = imp / total
            ancho = int(pct * 100)
            html_bars += f"""
            <div class="factor-row">
                <span class="factor-name">{nombre}</span>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{ancho}%"></div>
                </div>
                <span class="factor-pct">{pct*100:.1f}%</span>
            </div>
            """
        st.markdown(html_bars, unsafe_allow_html=True)

        # ── Contexto de riesgo ────────────────────────────────────────────
        st.markdown("")
        st.markdown('<p class="section-title">⚠️ Contexto de riesgo</p>', unsafe_allow_html=True)

        color_sev = color_severidad(severidad)
        umbral_critico = pred > 500_000

        c_ctx1, c_ctx2 = st.columns(2)
        with c_ctx1:
            nivel_riesgo = "🔴 Muy alto" if umbral_critico else ("🟠 Alto" if pred > 50_000 else "🟡 Moderado")
            st.metric("Nivel de riesgo financiero", nivel_riesgo)
            st.metric("Registros en riesgo", f"{registros:,}")
        with c_ctx2:
            costo_por_registro = pred / registros if registros > 0 else 0
            st.metric("Costo estimado por registro", f"$ {costo_por_registro:.2f}")
            st.metric("Tiempo de resolución", f"{tiempo_res:.0f} h")

        # ── Recomendaciones automáticas ───────────────────────────────────
        st.markdown("")
        st.markdown('<p class="section-title">💡 Acciones recomendadas</p>', unsafe_allow_html=True)

        recomendaciones = []
        if severidad in ("Crítica", "Alta"):
            recomendaciones.append("🚨 **Activar plan de respuesta a incidentes** de forma inmediata.")
        if tipo_ataque == "Ransomware":
            recomendaciones.append("💾 **Aislar sistemas afectados** y verificar integridad de backups offline.")
        if registros > 10_000:
            recomendaciones.append("📣 **Notificar a usuarios afectados** según regulaciones de protección de datos.")
        if reportado == "No" and severidad in ("Crítica", "Alta"):
            recomendaciones.append("🏛️ **Reportar a autoridades competentes** (CSIRT nacional).")
        if tiempo_res > 168:
            recomendaciones.append("⏱️ **Revisar el proceso de respuesta** — tiempos de resolución mayores a 7 días incrementan el daño.")
        if not recomendaciones:
            recomendaciones.append("✅ Mantener monitoreo continuo y documentar el incidente para análisis posterior.")

        for rec in recomendaciones:
            st.markdown(f"- {rec}")

    # ── Exportar datos del incidente ──────────────────────────────────────
    st.markdown("")
    st.markdown('<p class="section-title">📤 Exportar</p>', unsafe_allow_html=True)

    datos_export = {
        **datos_entrada,
        "prediccion_usd": round(pred, 2),
        "limite_inferior_usd": round(lo_pred, 2),
        "limite_superior_usd": round(hi_pred, 2),
    }
    df_export = pd.DataFrame([datos_export])

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar como CSV",
            data=csv_bytes,
            file_name="prediccion_incidente.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp2:
        json_str = df_export.to_json(orient="records", force_ascii=False, indent=2)
        st.download_button(
            label="⬇️ Descargar como JSON",
            data=json_str.encode("utf-8"),
            file_name="prediccion_incidente.json",
            mime="application/json",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:0.78rem;color:#4A6E88'>"
    "CiberRisk · Dataset: Ciberseguridad LATAM 2021–2025 · "
    "Fuentes: Kaspersky, Check Point, Recorded Future, IBM X-Force, ESET"
    "</p>",
    unsafe_allow_html=True,
)
