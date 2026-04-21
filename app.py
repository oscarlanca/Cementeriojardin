import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ── Paleta y estilo global ──────────────────────────────────────────────────
DARK_BG   = "#0D1117"
CARD_BG   = "#161B22"
ACCENT    = "#58A6FF"
ACCENT2   = "#3FB950"
WARN      = "#D29922"
DANGER    = "#F85149"
TEXT_MAIN = "#E6EDF3"
TEXT_MUTED= "#8B949E"
BORDER    = "#30363D"

plt.rcParams.update({
    "figure.facecolor": CARD_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_MAIN,
    "xtick.color":      TEXT_MUTED,
    "ytick.color":      TEXT_MUTED,
    "text.color":       TEXT_MAIN,
    "grid.color":       BORDER,
    "grid.linewidth":   0.6,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": BORDER,
    "font.family":      "monospace",
})

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Invercruz · Análisis de Deudas",
    page_icon="⚰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {{
      background-color: {DARK_BG};
      color: {TEXT_MAIN};
      font-family: 'IBM Plex Sans', sans-serif;
  }}
  h1, h2, h3 {{ font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; }}
  .metric-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 1rem 1.25rem;
      text-align: center;
  }}
  .metric-card .val {{
      font-size: 2rem;
      font-weight: 600;
      font-family: 'IBM Plex Mono', monospace;
  }}
  .metric-card .lbl {{
      font-size: 0.75rem;
      color: {TEXT_MUTED};
      text-transform: uppercase;
      letter-spacing: .08em;
  }}
  .tag {{
      display:inline-block;
      padding:2px 8px;
      border-radius:4px;
      font-size:.75rem;
      font-weight:600;
  }}
  .stDataFrame {{ border: 1px solid {BORDER}; border-radius:8px; }}
  section[data-testid="stSidebar"] {{ background:{CARD_BG}; border-right:1px solid {BORDER}; }}
  .stSelectbox > div > div {{ background:{CARD_BG}; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cargar_y_procesar(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee el CSV y devuelve (df_limpio, df_excluidos, base_contrato)."""
    df_raw = pd.read_csv(file_bytes)

    # ── Separar registros sin LOTE o tipo FUNERARIA ─────────────────────────
    df2  = df_raw[df_raw['LOTE'].isna()].copy()
    df   = df_raw.dropna(subset=['LOTE']).copy()
    df_fun = df[df['TIPO'].astype(str).str.upper().str.strip() == 'FUNERARIA'].copy()
    df2  = pd.concat([df2, df_fun], ignore_index=True)
    df   = df[df['TIPO'].astype(str).str.upper().str.strip() != 'FUNERARIA'].copy()

    # ── Normalizar nombres ──────────────────────────────────────────────────
    df['APELLIDOS'] = df['APELLIDOS'].fillna('').astype(str).str.strip()
    df['NOMBRES']   = df['NOMBRES'].fillna('').astype(str).str.strip()
    df['NOMBRE_COMPLETO'] = (
        (df['APELLIDOS'] + ' ' + df['NOMBRES'])
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.upper()
    )
    df.drop(columns=['APELLIDOS', 'NOMBRES'], inplace=True)

    # ── ID único de contrato ────────────────────────────────────────────────
    for col in ['NOMBRE_COMPLETO', 'SECTOR', 'LOTE']:
        df[col] = df[col].astype(str).str.upper().str.strip()

    df['ID_CONTRATO'] = df['NOMBRE_COMPLETO'] + '_' + df['SECTOR'] + '_' + df['LOTE']

    # ── Limpieza numérica ───────────────────────────────────────────────────
    df["FECORIG"] = pd.to_datetime(df["FECORIG"], errors="coerce")

    base_contrato = df.groupby("ID_CONTRATO").agg(
        NOMBRE_COMPLETO=("NOMBRE_COMPLETO", "first"),
        FECHA_INICIO   =("FECORIG",         "min"),
        TOTAL_MORA     =("INTMORA",         "sum"),
        DEUDA_MAX      =("SALDO",           "max"),
        SALDO_FINAL    =("SALDO",           "last"),
    ).reset_index()

    for col in ["TOTAL_MORA", "DEUDA_MAX", "SALDO_FINAL"]:
        base_contrato[col] = (
            base_contrato[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        base_contrato[col] = pd.to_numeric(base_contrato[col], errors="coerce")

    # ── Antigüedad ──────────────────────────────────────────────────────────
    hoy = pd.Timestamp(datetime.today())
    base_contrato["ANTIGUEDAD_ANIOS"] = (
        hoy - base_contrato["FECHA_INICIO"]
    ).dt.days / 365

    # ── Estado y evento ─────────────────────────────────────────────────────
    def estado_final(row):
        if row["TOTAL_MORA"] > 0:   return "MOROSO"
        elif row["DEUDA_MAX"] > 0:  return "DEUDA"
        elif row["SALDO_FINAL"] == 0: return "LIQUIDADO"
        else:                        return "ACTIVO"

    base_contrato["ESTADO_CONTRATO"] = base_contrato.apply(estado_final, axis=1)
    base_contrato["EVENTO"] = np.where(base_contrato["ESTADO_CONTRATO"] == "MOROSO", 1, 0)

    return df, df2, base_contrato


def km_plot(durations_list, events_list, labels, title):
    """Genera una figura Kaplan-Meier con el estilo oscuro del proyecto."""
    kmf = KaplanMeierFitter()
    colors = [ACCENT, ACCENT2, WARN, DANGER]
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (dur, evt, lbl) in enumerate(zip(durations_list, events_list, labels)):
        kmf.fit(durations=dur, event_observed=evt, label=lbl)
        kmf.plot_survival_function(ax=ax, ci_show=(len(labels) == 1),
                                   color=colors[i % len(colors)], linewidth=2)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=14, color=ACCENT)
    ax.set_xlabel("Antigüedad (años)", fontsize=10)
    ax.set_ylabel("Probabilidad de Supervivencia", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚰️ Invercruz")
    st.markdown(f"<span style='color:{TEXT_MUTED}; font-size:.8rem'>Análisis de Deudas y Supervivencia</span>", unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader(
        "Cargar CSV de contratos",
        type=["csv"],
        help="Sube el archivo invercruz_deudas_columnas_exactas.csv"
    )

    st.divider()
    pagina = st.radio(
        "Navegación",
        ["📊 Resumen General", "🔬 Análisis KM", "🤖 Modelo ML", "📋 Datos Crudos"],
        label_visibility="collapsed"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PANTALLA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("# ⚰️ Invercruz · Análisis de Deudas")
    st.markdown(f"""
    <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px; padding:2rem; max-width:600px; margin-top:1.5rem;">
      <p style="color:{TEXT_MUTED}; font-size:1rem; line-height:1.7">
        Sube el archivo <code>invercruz_deudas_columnas_exactas.csv</code> desde el panel
        izquierdo para comenzar el análisis de supervivencia y clasificación de morosos.
      </p>
      <hr style="border-color:{BORDER}; margin:1rem 0">
      <p style="color:{TEXT_MUTED}; font-size:.85rem">Módulos disponibles:</p>
      <ul style="color:{TEXT_MAIN}; font-size:.9rem; line-height:2">
        <li>📊 <strong>Resumen General</strong> — KPIs y distribución de estados</li>
        <li>🔬 <strong>Análisis KM</strong> — Curvas de Kaplan-Meier interactivas</li>
        <li>🤖 <strong>Modelo ML</strong> — Regresión Logística + métricas</li>
        <li>📋 <strong>Datos Crudos</strong> — Tablas filtrables</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Procesar datos ──────────────────────────────────────────────────────────
with st.spinner("Procesando datos…"):
    df, df2, base_contrato = cargar_y_procesar(uploaded, uploaded.name)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 · RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📊 Resumen General":
    st.markdown("## 📊 Resumen General")

    conteos = base_contrato["ESTADO_CONTRATO"].value_counts()
    total   = len(base_contrato)

    colores_estado = {"MOROSO": DANGER, "DEUDA": WARN, "LIQUIDADO": ACCENT2, "ACTIVO": ACCENT}

    # KPI cards
    cols = st.columns(len(conteos) + 1)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:{ACCENT}">{total:,}</div>
          <div class="lbl">Total Contratos</div>
        </div>""", unsafe_allow_html=True)

    for i, (estado, cnt) in enumerate(conteos.items()):
        color = colores_estado.get(estado, TEXT_MAIN)
        pct   = cnt / total * 100
        with cols[i + 1]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val" style="color:{color}">{cnt:,}</div>
              <div class="lbl">{estado} &nbsp;<span style="color:{TEXT_MUTED}">({pct:.1f}%)</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfica de distribución
    c1, c2 = st.columns([1.4, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        estados = conteos.index.tolist()
        vals    = conteos.values.tolist()
        bar_colors = [colores_estado.get(e, ACCENT) for e in estados]
        bars = ax.barh(estados, vals, color=bar_colors, height=0.55, edgecolor=BORDER)
        for bar, val in zip(bars, vals):
            ax.text(val + total * 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va='center', fontsize=9, color=TEXT_MAIN)
        ax.set_xlabel("Contratos")
        ax.set_title("Distribución de Estados de Contrato", fontweight='bold', color=ACCENT, pad=10)
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)

    with c2:
        st.markdown("**Estadísticas financieras**")
        st.dataframe(
            base_contrato[["TOTAL_MORA", "DEUDA_MAX", "SALDO_FINAL", "ANTIGUEDAD_ANIOS"]]
            .describe()
            .round(2)
            .rename(columns={
                "TOTAL_MORA": "Mora",
                "DEUDA_MAX":  "Deuda Máx.",
                "SALDO_FINAL":"Saldo Final",
                "ANTIGUEDAD_ANIOS": "Años",
            }),
            use_container_width=True,
        )

    # Top morosos
    st.divider()
    st.markdown("### 🔴 Top Morosos por Mora Acumulada")
    top_m = (
        base_contrato[base_contrato["ESTADO_CONTRATO"] == "MOROSO"]
        .sort_values("TOTAL_MORA", ascending=False)
        .head(15)[["NOMBRE_COMPLETO", "TOTAL_MORA", "DEUDA_MAX", "ANTIGUEDAD_ANIOS", "FECHA_INICIO"]]
        .reset_index(drop=True)
    )
    top_m.index += 1
    top_m["TOTAL_MORA"] = top_m["TOTAL_MORA"].map("${:,.2f}".format)
    top_m["DEUDA_MAX"]  = top_m["DEUDA_MAX"].map("${:,.2f}".format)
    top_m["ANTIGUEDAD_ANIOS"] = top_m["ANTIGUEDAD_ANIOS"].map("{:.1f} años".format)
    st.dataframe(top_m, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 · ANÁLISIS KAPLAN-MEIER
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔬 Análisis KM":
    st.markdown("## 🔬 Análisis de Supervivencia — Kaplan-Meier")

    base_km = base_contrato[
        base_contrato["ANTIGUEDAD_ANIOS"].notna() &
        (base_contrato["ANTIGUEDAD_ANIOS"] >= 0)
    ].copy()

    tab1, tab2, tab3 = st.tabs(["Global", "Filtro por Nombre", "Comparación"])

    with tab1:
        st.markdown("Curva global de **probabilidad de no caer en mora** a lo largo del tiempo.")
        fig = km_plot(
            [base_km["ANTIGUEDAD_ANIOS"]],
            [base_km["EVENTO"]],
            ["Total Contratos"],
            "Curva de Supervivencia — Todos los Contratos",
        )
        st.pyplot(fig)

    with tab2:
        busqueda = st.text_input("Buscar contratos por nombre (ej: LANDIVAR)", value="LANDIVAR").upper().strip()
        subset = base_km[base_km["NOMBRE_COMPLETO"].str.contains(busqueda, na=False)]
        st.markdown(f"Contratos encontrados: **{len(subset):,}**")

        if len(subset) < 2:
            st.warning("Se necesitan al menos 2 registros para graficar una curva KM.")
        else:
            fig = km_plot(
                [subset["ANTIGUEDAD_ANIOS"]],
                [subset["EVENTO"]],
                [busqueda],
                f"Supervivencia — Contratos '{busqueda}'",
            )
            st.pyplot(fig)
            st.dataframe(
                subset[["NOMBRE_COMPLETO","ESTADO_CONTRATO","TOTAL_MORA","ANTIGUEDAD_ANIOS"]]
                .reset_index(drop=True),
                use_container_width=True,
            )

    with tab3:
        st.markdown("Compara la curva **global** con cualquier subgrupo filtrado.")
        comp_nombre = st.text_input("Subgrupo a comparar", value="LANDIVAR").upper().strip()
        comp_subset = base_km[base_km["NOMBRE_COMPLETO"].str.contains(comp_nombre, na=False)]

        if len(comp_subset) < 2:
            st.warning("El subgrupo necesita al menos 2 registros.")
        else:
            fig = km_plot(
                [base_km["ANTIGUEDAD_ANIOS"], comp_subset["ANTIGUEDAD_ANIOS"]],
                [base_km["EVENTO"],           comp_subset["EVENTO"]],
                ["Total Contratos",           f"'{comp_nombre}'"],
                "Comparación de Curvas de Supervivencia (No Mora)",
            )
            st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 · MODELO ML
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Modelo ML":
    st.markdown("## 🤖 Modelo de Clasificación — Regresión Logística")

    features = ['TOTAL_MORA', 'DEUDA_MAX', 'SALDO_FINAL', 'ANTIGUEDAD_ANIOS']
    df_model = base_contrato[features + ['EVENTO']].dropna()
    X = df_model[features]
    y = df_model['EVENTO']

    st.markdown(f"Registros usados para entrenamiento: **{len(df_model):,}**")

    c_param, _ = st.columns([1, 2])
    with c_param:
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 50, 30, 5) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_proba)

    # Métricas
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, lbl, val, color in [
        (m1, "Accuracy",   accuracy,  ACCENT),
        (m2, "Precision",  precision, ACCENT2),
        (m3, "Recall",     recall,    WARN),
        (m4, "F1-Score",   f1,        DANGER),
        (m5, "ROC-AUC",    roc_auc,   "#A371F7"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val" style="color:{color}">{val:.3f}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Matriz de confusión + coeficientes
    col_cm, col_coef = st.columns(2)

    with col_cm:
        st.markdown("**Matriz de Confusión**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Moroso", "Moroso"]); ax.set_yticklabels(["No Moroso", "Moroso"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, color='white' if cm[i, j] > cm.max() / 2 else TEXT_MAIN)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

    with col_coef:
        st.markdown("**Importancia de Variables (Coeficientes)**")
        coef_df = pd.DataFrame({
            "Variable": features,
            "Coeficiente": model.coef_[0]
        }).sort_values("Coeficiente", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(4, 3.5))
        colors_c = [DANGER if c > 0 else ACCENT2 for c in coef_df["Coeficiente"]]
        ax.barh(coef_df["Variable"], coef_df["Coeficiente"], color=colors_c, edgecolor=BORDER)
        ax.axvline(0, color=BORDER, linewidth=1)
        ax.set_title("Coeficientes", fontsize=10, color=ACCENT)
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)

    # Predicción individual
    st.divider()
    st.markdown("### 🎯 Predicción Individual")
    st.markdown("Ingresa los valores para predecir si un contrato caerá en mora.")
    pi1, pi2, pi3, pi4 = st.columns(4)
    mora_val  = pi1.number_input("Total Mora ($)",    min_value=0.0, step=10.0)
    deuda_val = pi2.number_input("Deuda Máxima ($)",  min_value=0.0, step=10.0)
    saldo_val = pi3.number_input("Saldo Final ($)",   min_value=0.0, step=10.0)
    anios_val = pi4.number_input("Antigüedad (años)", min_value=0.0, step=0.5)

    if st.button("Predecir"):
        inp  = np.array([[mora_val, deuda_val, saldo_val, anios_val]])
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1]
        if pred == 1:
            st.error(f"⚠️ **MOROSO** — Probabilidad: {prob:.1%}")
        else:
            st.success(f"✅ **NO MOROSO** — Probabilidad de mora: {prob:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 · DATOS CRUDOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📋 Datos Crudos":
    st.markdown("## 📋 Datos Crudos")

    tab_main, tab_excl, tab_cont = st.tabs(
        ["Contratos Activos", "Excluidos (sin lote / funeraria)", "Base Contratos"]
    )

    with tab_main:
        st.markdown(f"**{len(df):,} registros**")
        filtro = st.text_input("Filtrar por nombre", key="f1")
        df_show = df[df['NOMBRE_COMPLETO'].str.contains(filtro.upper(), na=False)] if filtro else df
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True)

    with tab_excl:
        st.markdown(f"**{len(df2):,} registros excluidos**")
        st.dataframe(df2.reset_index(drop=True), use_container_width=True)

    with tab_cont:
        st.markdown(f"**{len(base_contrato):,} contratos únicos**")
        filtro2 = st.text_input("Filtrar por nombre", key="f2")
        bc_show = (
            base_contrato[base_contrato['NOMBRE_COMPLETO'].str.contains(filtro2.upper(), na=False)]
            if filtro2 else base_contrato
        )
        st.dataframe(bc_show.reset_index(drop=True), use_container_width=True)
