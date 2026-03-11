# PCA Reservoir Analysis

Aplicación interactiva de **Análisis de Componentes Principales (PCA)** sobre datos de reservorios petroleros, construida con **Streamlit** y **scikit-learn**.

## Datasets

| Dataset | Descripción | Filas |
|---------|-------------|-------|
| **Capa** | Propiedades por capa/formación | 8 |
| **Malla** | Datos por par inyector-productor | 353 |

## Visualizaciones

- 📊 **Scree Plot** — Varianza explicada por componente
- 🔵 **Scatter 2D** — PC1 vs PC2 coloreado por capa
- 🌐 **Scatter 3D** — PC1 × PC2 × PC3 interactivo
- 🎯 **Biplot** — Scores + vectores de loadings
- 🔥 **Heatmap de Loadings** — Contribución de cada variable

## Correr Localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Desplegado en [Streamlit Cloud](https://streamlit.io/cloud).
