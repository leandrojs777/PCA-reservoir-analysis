import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap

df = pd.read_excel('data/Malla.xlsx')

# Limpiar nombres
def normalize_cols(columns):
    import re
    norm_cols = []
    for c in columns:
        c = str(c).lower().strip()
        c = c.replace(" ", "_")
        c = re.sub(r'[^a-z0-9_]', '', c)
        norm_cols.append(c)
    return norm_cols
df.columns = normalize_cols(df.columns)

features = ['so', 'vp', 'wi_pv', 'oip']
target = 'fr'

df_clean = df.dropna(subset=features + [target])

X = df_clean[features]
y = df_clean[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R2 Score: {r2}")

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)

print(f"SHAP values shape: {shap_vals.shape}")
print(f"Type of SHAP values: {type(shap_vals)}")
