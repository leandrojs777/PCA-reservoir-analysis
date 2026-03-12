import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA · Reservoir Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1, h2, h3 {
    font-weight: 700;
}
.stMetric > div {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stMetric label, .stMetric [data-testid="stMetricLabel"] {
    color: #a0a0cc !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #ffffff !important;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
}
div[data-testid="stSidebar"] .stMarkdown h1,
div[data-testid="stSidebar"] .stMarkdown h2,
div[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}
</style>
""", unsafe_allow_html=True)

# ── Color Palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#6C63FF", "#FF6584", "#43E8D8", "#FFD93D", "#FF8C42",
    "#A8E6CF", "#FF6B6B", "#4ECDC4", "#C44DFF", "#3BCEAC",
]

PLOTLY_TEMPLATE = "plotly_dark"


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    capa = pd.read_excel("data/Capa.xlsx")
    malla = pd.read_excel("data/Malla.xlsx")
    
    # Skill de Normalización de columnas
    def normalize_cols(columns):
        norm_cols = []
        for c in columns:
            c = str(c).lower().strip()
            c = c.replace(" ", "_")
            c = re.sub(r'[^a-z0-9_]', '', c)
            norm_cols.append(c)
        return norm_cols

    capa.columns = normalize_cols(capa.columns)
    malla.columns = normalize_cols(malla.columns)
    
    return capa, malla


capa_df, malla_df = load_data()

def compute_well_clusters(df):
    """
    Skill compute_well_clusters: Agrupa las mallas en 3 categorías de rendimiento.
    """
    # Variables de entrada para el modelo
    features = ['so', 'vp', 'np', 'wi', 'oip']
    
    # Realiza una copia para no alterar el caché
    result_df = df.copy()
    
    # Manejo de nulos (KMeans no permite NaNs)
    X = result_df[features].fillna(0)
    
    # Estandarización de las variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar el algoritmo KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    result_df['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    return result_df

malla_df = compute_well_clusters(malla_df)

@st.cache_resource
def predict_and_explain_fr(df):
    """
    Skill predict_and_explain_fr: Predicts FR using So, Vp, Wi_PV, OIP and calculates SHAP values.
    """
    features = ['so', 'vp', 'wi_pv', 'oip']
    target = 'fr'
    
    df_clean = df.dropna(subset=features + [target]).copy()
    
    X = df_clean[features]
    y = df_clean[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return model, r2, explainer, shap_values, features, df_clean

DATASETS = {
    "Capa (Formaciones)": {
        "df": capa_df,
        "label_col": "capa",
        "hover_name_col": "capa",
        "numeric_cols": ["pv", "a", "wi", "wid", "oip", "mov_oip", "so", "np"],
        "description": "Propiedades agregadas por capa/formación del reservorio (8 capas).",
    },
    "Malla (Pares Inyector-Productor)": {
        "df": malla_df,
        "label_col": "capa",
        "hover_name_col": "id_malla",
        "numeric_cols": ["so", "vp", "np", "np_mov", "wi", "wi_pv", "fr", "oip"],
        "description": "Datos por par inyector-productor a nivel de malla (353 registros).",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🔬 PCA Config")
    st.markdown("---")

    dataset_name = st.selectbox(
        "📂 Dataset", 
        list(DATASETS.keys()),
        help="Elige el nivel de agregación de los datos. Malla contiene información detallada por par inyector-productor, y Capa agrupa propiedades a nivel formacional."
    )
    ds = DATASETS[dataset_name]

    st.markdown(f"*{ds['description']}*")

    st.markdown("---")
    st.markdown("### Variables")
    selected_cols = st.multiselect(
        "Seleccionar variables para PCA",
        options=ds["numeric_cols"],
        default=ds["numeric_cols"],
        help="Las variables numéricas a reducir dimensionalmente. Por defecto, todas las métricas de volumen, producción e inyección están seleccionadas."
    )

    max_components = min(len(selected_cols), len(ds["df"])) if selected_cols else 2
    n_components = st.slider(
        "Número de componentes",
        min_value=2,
        max_value=max(max_components, 2),
        value=min(max_components, 3),
        help="Los 'ejes sintéticos' que resumirán la varianza de los datos reales. Generalmente 2 o 3 componentes agrupan el >70% de la información (Varianza)."
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;opacity:0.5;font-size:0.8rem'>"
        "Built with Streamlit + sklearn</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PCA Computation
# ══════════════════════════════════════════════════════════════════════════════

if len(selected_cols) < 2:
    st.error("⚠️ Seleccioná al menos 2 variables para realizar PCA.")
    st.stop()

df = ds["df"].copy()
label_col = ds["label_col"]
hover_name_col = ds.get("hover_name_col", label_col)
X = df[selected_cols].dropna()
labels = df.loc[X.index, label_col]
hover_names = df.loc[X.index, hover_name_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=n_components)
scores = pca.fit_transform(X_scaled)

pc_cols = [f"PC{i+1}" for i in range(n_components)]
scores_df = pd.DataFrame(scores, columns=pc_cols, index=X.index)
scores_df[label_col] = labels.values
scores_df[hover_name_col] = hover_names.values

# Agregar variables originales a scores_df para mostrarlas en el hover
for col in selected_cols:
    if col not in scores_df.columns:
        scores_df[col] = df.loc[X.index, col]

loadings = pd.DataFrame(
    pca.components_.T,
    columns=pc_cols,
    index=selected_cols,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Header + KPIs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🔬 Análisis de Componentes Principales (PCA)")
st.markdown(f"**Dataset:** {dataset_name}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Registros", len(X))
kpi2.metric("Variables", len(selected_cols))
kpi3.metric("Componentes", n_components)
kpi4.metric(
    "Varianza Total Explicada",
    f"{pca.explained_variance_ratio_.sum():.1%}",
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  Tab Layout
# ══════════════════════════════════════════════════════════════════════════════

tab_guide, tab_data, tab_scree, tab_scatter, tab_3d, tab_biplot, tab_heatmap, tab_table, tab_diag, tab_opt = st.tabs([
    "📖 Guía y Metodología",
    "📋 Datos",
    "📊 Scree Plot",
    "🔵 Scatter 2D",
    "🌐 Scatter 3D",
    "🎯 Biplot",
    "🔥 Heatmap",
    "📑 Componentes",
    "🤖 Diagnóstico IA",
    "🎯 Optimización de Mallas",
])

# ── Tab: Guía y Metodología ──────────────────────────────────────────────────
with tab_guide:
    st.markdown("### 📖 Guía y Metodología del Proyecto")
    st.markdown("Esta aplicación forma parte del ecosistema de **Informes de Estado del Arte**. Integra metodologías de Inteligencia Artificial Explicable (XAI) para dotar de transparencia, educación y accionabilidad al análisis de factores de recuperación de petróleo.")
    st.markdown("---")
    
    with st.expander("🔬 PCA (Análisis de Componentes Principales)", expanded=True):
        st.markdown("""
        Reducción de 8 variables a componentes principales para ver la variabilidad del reservorio.
        """)

    with st.expander("🤖 Clustering"):
        st.markdown("""
        Agrupamiento de las 353 mallas por similitud operativa.
        """)

    with st.expander("🎯 SHAP"):
        st.markdown("""
        Explicación de qué variables físicas aumentan (rojo) o frenan (azul) el Factor de Recuperación (FR).
        """)

    st.markdown("### 📚 Glosario de Variables Clave")
    st.markdown("""
    A continuación, el diccionario de variables analizadas por el ensamble de Machine Learning:
    
    | Variable | Significado Técnico | Descripción Funcional |
    |---|---|---|
    | **`So`** | Saturación de Petróleo | Fracción del volumen poroso ocupado por petróleo remanente. |
    | **`Vp`** | Volumen Poral | Capacidad total de almacenamiento de fluidos de la roca en la malla analizada. |
    | **`Wi_PV`** | Agua Inyectada / Vol. Poral | Relación entre el volumen inyectado (Wi) normalizado por el tamaño del poro del reservorio. |
    | **`OIP`** | Petróleo Original In Situ | Estimación inicial de los barriles originales de petróleo atrapados en sitio (Original Oil In Place). |
    | **`FR`** | Factor de Recuperación | (Variable Objetivo). El porcentaje real del hidrocarburo original que se ha logrado bombear hacia superficie. |
    """)


# ── Tab: Raw Data ────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Datos crudos")
    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("### Estadísticas descriptivas")
    st.dataframe(
        df[selected_cols].describe().round(2),
        use_container_width=True,
    )

    # Correlation matrix
    st.markdown("### Matriz de Correlación")
    corr = df[selected_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        zmin=-1,
        zmax=1,
    )
    fig_corr.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ── Tab: Scree Plot ──────────────────────────────────────────────────────────
with tab_scree:
    st.markdown("### Scree Plot — Varianza Explicada")
    st.markdown(
        "Muestra cuánta varianza del dataset original captura cada componente principal."
    )

    var_exp = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)

    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])

    fig_scree.add_trace(
        go.Bar(
            x=pc_cols,
            y=var_exp,
            name="Individual",
            marker_color=PALETTE[0],
            marker_line_width=0,
            opacity=0.85,
            text=[f"{v:.1%}" for v in var_exp],
            textposition="outside",
            textfont=dict(size=13, color="#e0e0ff"),
        ),
        secondary_y=False,
    )

    fig_scree.add_trace(
        go.Scatter(
            x=pc_cols,
            y=cum_var,
            name="Acumulada",
            mode="lines+markers+text",
            line=dict(color=PALETTE[1], width=3),
            marker=dict(size=10, symbol="diamond"),
            text=[f"{v:.1%}" for v in cum_var],
            textposition="top center",
            textfont=dict(size=12, color=PALETTE[1]),
        ),
        secondary_y=True,
    )

    fig_scree.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        yaxis_title="Varianza Explicada",
        yaxis2_title="Varianza Acumulada",
        yaxis=dict(tickformat=".0%", range=[0, max(var_exp) * 1.3]),
        yaxis2=dict(tickformat=".0%", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_scree, use_container_width=True)


# ── Tab: 2D Scatter ──────────────────────────────────────────────────────────
with tab_scatter:
    st.markdown("### Scatter 2D — PC1 vs PC2")
    st.markdown(
        "Cada punto es un registro del dataset, proyectado en las dos primeras componentes principales."
    )

    fig_2d = px.scatter(
        scores_df,
        x="PC1",
        y="PC2",
        color=label_col,
        hover_name=hover_name_col,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=PALETTE,
        hover_data=selected_cols,
    )
    fig_2d.update_traces(marker=dict(size=12, line=dict(width=1, color="#1a1a2e")))
    fig_2d.update_layout(
        height=600,
        xaxis_title=f"PC1 ({var_exp[0]:.1%} varianza)",
        yaxis_title=f"PC2 ({var_exp[1]:.1%} varianza)",
        legend_title=label_col,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Inter"),
    )

    # Add labels for Capa dataset (few points)
    if len(scores_df) <= 20:
        for _, row in scores_df.iterrows():
            fig_2d.add_annotation(
                x=row["PC1"],
                y=row["PC2"],
                text=str(row[label_col]),
                showarrow=True,
                arrowhead=0,
                arrowcolor="rgba(255,255,255,0.3)",
                font=dict(size=10, color="#e0e0ff"),
                bgcolor="rgba(26,26,46,0.7)",
                borderpad=3,
            )

    st.plotly_chart(fig_2d, use_container_width=True)


# ── Tab: 3D Scatter ──────────────────────────────────────────────────────────
with tab_3d:
    if n_components >= 3:
        st.markdown("### Scatter 3D — PC1 × PC2 × PC3")
        st.markdown("Rotá el gráfico arrastrando con el mouse para explorar los datos en 3D.")

        fig_3d = px.scatter_3d(
            scores_df,
            x="PC1",
            y="PC2",
            z="PC3",
            color=label_col,
            hover_name=hover_name_col,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=PALETTE,
            hover_data=selected_cols,
        )
        fig_3d.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#1a1a2e")))
        fig_3d.update_layout(
            height=700,
            scene=dict(
                xaxis_title=f"PC1 ({var_exp[0]:.1%})",
                yaxis_title=f"PC2 ({var_exp[1]:.1%})",
                zaxis_title=f"PC3 ({var_exp[2]:.1%})",
            ),
            legend_title=label_col,
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("🔢 Seleccioná al menos 3 componentes para ver el gráfico 3D.")


# ── Tab: Biplot ──────────────────────────────────────────────────────────────
with tab_biplot:
    st.markdown("### Biplot — Scores + Loadings")
    st.markdown(
        "Los **puntos** son las observaciones en el espacio PCA. "
        "Las **flechas** muestran la dirección y magnitud de cada variable original."
    )

    fig_bi = px.scatter(
        scores_df,
        x="PC1",
        y="PC2",
        color=label_col,
        hover_name=hover_name_col,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=PALETTE,
        hover_data=selected_cols,
    )
    fig_bi.update_traces(marker=dict(size=10, line=dict(width=1, color="#1a1a2e")))

    # Loading vectors — scale to fit nicely
    max_score = max(
        scores_df["PC1"].abs().max(),
        scores_df["PC2"].abs().max(),
    )
    max_loading = max(loadings["PC1"].abs().max(), loadings["PC2"].abs().max())
    scale = max_score / max_loading * 0.8 if max_loading > 0 else 1

    for var in selected_cols:
        lx = loadings.loc[var, "PC1"] * scale
        ly = loadings.loc[var, "PC2"] * scale

        fig_bi.add_trace(
            go.Scatter(
                x=[0, lx],
                y=[0, ly],
                mode="lines",
                line=dict(color="#FF3131", width=2.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Arrowhead
        fig_bi.add_annotation(
            x=lx,
            y=ly,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#FF3131",
        )
        fig_bi.add_annotation(
            x=lx * 1.12,
            y=ly * 1.12,
            text=f"<b>{var}</b>",
            showarrow=False,
            font=dict(size=11, color="#FF3131"),
        )

    fig_bi.update_layout(
        template=PLOTLY_TEMPLATE,
        height=650,
        xaxis_title=f"PC1 ({var_exp[0]:.1%} varianza)",
        yaxis_title=f"PC2 ({var_exp[1]:.1%} varianza)",
        legend_title=label_col,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_bi, use_container_width=True)


# ── Tab: Loadings Heatmap ────────────────────────────────────────────────────
with tab_heatmap:
    st.markdown("### Heatmap de Loadings")
    st.markdown(
        "Muestra la contribución (peso) de cada variable original en cada componente principal. "
        "Valores altos (positivos o negativos) indican mayor influencia."
    )

    fig_heat = px.imshow(
        loadings.values,
        x=pc_cols,
        y=selected_cols,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        zmin=-1,
        zmax=1,
    )
    fig_heat.update_layout(
        height=max(400, len(selected_cols) * 55),
        xaxis_title="Componente Principal",
        yaxis_title="Variable Original",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter"),
        coloraxis_colorbar=dict(title="Peso"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Bar chart of loadings per component
    st.markdown("### Loadings por Componente")
    selected_pc = st.selectbox("Seleccionar componente", pc_cols, key="loading_pc")

    fig_bar = px.bar(
        loadings.reset_index().rename(columns={"index": "Variable"}),
        x=selected_pc,
        y="Variable",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color=selected_pc,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
    )
    fig_bar.update_layout(
        height=max(350, len(selected_cols) * 45),
        xaxis_title=f"Loading en {selected_pc}",
        yaxis_title="",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab: Components Table ────────────────────────────────────────────────────
with tab_table:
    st.markdown("### Resumen de Componentes")

    summary_df = pd.DataFrame({
        "Componente": pc_cols,
        "Autovalor": pca.explained_variance_,
        "Varianza Explicada": pca.explained_variance_ratio_,
        "Varianza Acumulada": np.cumsum(pca.explained_variance_ratio_),
    })
    summary_df["Varianza Explicada"] = summary_df["Varianza Explicada"].map("{:.2%}".format)
    summary_df["Varianza Acumulada"] = summary_df["Varianza Acumulada"].map("{:.2%}".format)
    summary_df["Autovalor"] = summary_df["Autovalor"].round(4)

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("### Tabla de Loadings")
    st.dataframe(loadings.round(4), use_container_width=True)

    st.markdown("### Scores (Datos Transformados)")
    st.dataframe(scores_df.round(4), use_container_width=True, height=400)


# ── Tab: Diagnóstico IA ──────────────────────────────────────────────────────
with tab_diag:
    st.markdown("### 🤖 Diagnóstico IA — Clustering de Rendimiento")
    
    if "cluster_id" in df.columns:
        st.markdown(
            "Agrupación de mallas en **3 categorías** usando KMeans sobre las variables "
            "`so`, `vp`, `np`, `wi` y `oip`. El gráfico muestra los datos proyectados en el espacio PCA."
        )
        
        # Merge de cluster_id al dataframe de scores para poder graficarlo
        diag_df = scores_df.copy()
        diag_df["cluster_id"] = df.loc[diag_df.index, "cluster_id"].astype(str)
        
        fig_diag = px.scatter(
            diag_df,
            x="PC1",
            y="PC2",
            color="cluster_id",
            hover_name=hover_name_col,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#FFD93D"],
            title="Diagnóstico IA: PC1 vs PC2 por Clúster",
            hover_data=selected_cols
        )
        fig_diag.update_traces(marker=dict(size=10, line=dict(width=1, color="#1a1a2e")))
        fig_diag.update_layout(
            height=500,
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(family="Inter"),
        )
        
        st.plotly_chart(fig_diag, use_container_width=True)
        
        st.markdown("### 📊 Eficiencia Promedio por Clúster")
        st.markdown(
            "Comparación del rendimiento promedio iterando métricas clave como `np` (producción) y `wi` (inyección)."
        )
        
        # Agrupar y calcular promedios
        cluster_summary = df.groupby("cluster_id")[["np", "wi"]].mean().reset_index()
        cluster_summary.rename(columns={
            "cluster_id": "Clúster",
            "np": "Producción Promedio (np)",
            "wi": "Inyección Promedio (wi)"
        }, inplace=True)
        
        st.dataframe(cluster_summary.round(2), use_container_width=True, hide_index=True)
        
    else:
        st.info("ℹ️ El diagnóstico IA solo está disponible para el dataset de 'Malla', el cual contiene la skill de Clustering.")

# ── Tab: Optimización de Mallas ──────────────────────────────────────────────
with tab_opt:
    st.markdown("### 🎯 Optimización de Mallas (XAI)")
    
    if dataset_name == "Malla (Pares Inyector-Productor)":
        model, r2, explainer, shap_values, feat_cols, df_ml = predict_and_explain_fr(ds["df"])
        
        st.write(f"Modelo **RandomForestRegressor** entrenado para explicar el Factor de Recuperación (FR).")
        st.write(f"**Precisión (Score $R^2$):** {r2:.4f}")
        
        if r2 < 0.7:
            st.warning(f"⚠️ **Nota sobre la Precisión:** El Score $R^2$ de este modelo es de {r2:.2f}, lo que está por debajo del umbral óptimo de 0.70. Esto indica que las variables actuales (`so`, `vp`, `wi_pv`, `oip`) no explican exhaustivamente la variabilidad del Factor de Recuperación en este conjunto de datos, probablemente debido a dinámicas complejas del reservorio no capturadas. **Las explicaciones SHAP a continuación deben interpretarse como tendencias direccionales, no verdades absolutas.**")
            
        hover_col = ds.get("hover_name_col", label_col)
        mallas_disp = df_ml[hover_col].unique()
        
        sel_malla = st.selectbox(
            "Seleccionar Malla (Pozo) para explicar su FR:", 
            mallas_disp,
            help="Este modelo utiliza Random Forest para predecir el éxito operativo basado en datos históricos de inyección."
        )
        
        idx_array = np.where(df_ml[hover_col] == sel_malla)[0]
        if len(idx_array) > 0:
            idx = idx_array[0]
            
            base_val = explainer.expected_value
            if isinstance(base_val, np.ndarray):
                base_val = base_val[0]
            
            malla_sv = shap_values[idx]
            malla_real = df_ml['fr'].iloc[idx]
            malla_pred = model.predict(df_ml[feat_cols].iloc[idx].values.reshape(1, -1))[0]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Valor Base (Promedio)", f"{base_val:.3f}")
            col2.metric("FR Predicho", f"{malla_pred:.3f}")
            col3.metric("FR Real", f"{malla_real:.3f}", delta=f"{(malla_real - malla_pred):.3f} error", delta_color="inverse")
            
            # Ordenar por importancia absoluta
            sorted_idx = np.argsort(np.abs(malla_sv))
                
            fig_shap = go.Figure()
            # Variables positivas en rojo (mejoran FR), negativas en azul (reducen FR)
            bar_colors = ["#FF3131" if val > 0 else "#0000FF" for val in malla_sv[sorted_idx]]
            

            fig_shap.add_trace(go.Bar(
                y=[feat_cols[i] for i in sorted_idx],
                x=malla_sv[sorted_idx],
                orientation='h',
                marker_color=bar_colors,
                text=[f"{malla_sv[i]:.4f}" for i in sorted_idx],
                textposition="outside",
                hoverinfo="x+y"
            ))
            
            fig_shap.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"Impacto SHAP de cada variable en {sel_malla}",
                xaxis_title="Impacto en FR (Unidades de FR)",
                yaxis_title="Varaible",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # --- Lógica Prescriptiva ---
            impactos = [(feat_cols[i], malla_sv[i]) for i in range(len(feat_cols))]
            impactos.sort(key=lambda x: x[1])  # de menor a mayor
            
            worst_var, worst_val = impactos[0]
            best_var, best_val = impactos[-1]
            
            if worst_val < 0:
                st.info(f"Sugerencia: Revisar y optimizar **{worst_var.capitalize()}** para mejorar la recuperación.")
            if best_val > 0:
                st.success(f"Factor Crítico de Éxito: **{best_var.capitalize()}**.")
            
            st.info(f"💡 **Ayuda de Análisis Clínico**: Las barras **Rojas** (derecha) indican variables de este pozo que aumentan el FR. Las barras **Azules** (izquierda), frenan la recuperación.")
            
    else:
        st.info("ℹ️ La Optimización de Mallas solo está disponible para el dataset de 'Malla'.")
