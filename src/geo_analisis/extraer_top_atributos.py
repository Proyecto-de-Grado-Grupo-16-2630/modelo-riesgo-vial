import os
import sys
import joblib
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_GEO_OUT = "../../outputs/geo_evaluacion/"
CARPETA_MODELOS = "../../outputs/modelos_optimizados/"

RUTA_MUESTRA_GEO = os.path.join(CARPETA_GEO_OUT, "muestra_geocodificada_evaluacion.joblib")
RUTA_XGB = os.path.join(CARPETA_MODELOS, "multiclase_XGBoost.joblib")
RUTA_RF  = os.path.join(CARPETA_MODELOS, "multiclase_Random_Forest.joblib")
RUTA_TOP8_OUT = os.path.join(CARPETA_GEO_OUT, "top_8_atributos.json")

print("==================================================================")
print("  PASO 1: EXTRACCIÓN Y RANKING DE IMPORTANCIA DE ATRIBUTOS        ")
print("==================================================================")

# --- 2. CARGAR MATRIZ DE CARACTERÍSTICAS (X) ---
if not os.path.exists(RUTA_MUESTRA_GEO):
    print(f"ERROR: No se encuentra {RUTA_MUESTRA_GEO}")
    sys.exit(1)

datos_geo = joblib.load(RUTA_MUESTRA_GEO)
X_geo = datos_geo['X_geo']
nombres_columnas = X_geo.columns.tolist()

print(f" -> Dataset cargado: {len(nombres_columnas)} columnas/variables detectadas.")

# --- 3. CARGAR MODELO Y EXTRAER IMPORTANCIAS ---
if not os.path.exists(RUTA_XGB):
    print(f"ERROR: No se encuentra el modelo XGBoost en {RUTA_XGB}")
    sys.exit(1)

clf_xgb = joblib.load(RUTA_XGB)

# Si el objeto guardado es un Pipeline de scikit-learn, extraer el estimador final
if hasattr(clf_xgb, 'named_steps'):
    modelo_final = clf_xgb.named_steps[list(clf_xgb.named_steps.keys())[-1]]
else:
    modelo_final = clf_xgb

importancias = modelo_final.feature_importances_

# --- 4. CONSTRUIR DATAFRAME DE IMPORTANCIAS ---
df_imp = pd.DataFrame({
    'Atributo': nombres_columnas,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False).reset_index(drop=True)

df_imp['Porcentaje_Acumulado'] = (df_imp['Importancia'] / df_imp['Importancia'].sum() * 100).cumsum()

print("\n--- TOP 15 VARIABLES MÁS IMPORTANTES (XGBoost) ---")
print(df_imp.head(15).to_string(index=False))

# --- 5. SELECCIÓN DE LAS TOP 8 Y GUARDADO EN JSON ---
top_8_df = df_imp.head(8)
top_8_list = top_8_df['Atributo'].tolist()

print("\n------------------------------------------------------------------")
print("  TOP 8 SELECCIONADO PARA EL FORMULARIO WEB Y MODELO LITE:         ")
print("------------------------------------------------------------------")
for i, col in enumerate(top_8_list, 1):
    peso = top_8_df.loc[top_8_df['Atributo'] == col, 'Importancia'].values[0] * 100
    print(f" {i}. {col} (Peso: {peso:.2f}%)")

# Exportar a JSON para el Backend (Spring Boot / FastAPI)
import json
with open(RUTA_TOP8_OUT, 'w', encoding='utf-8') as f:
    json.dump({
        'top_8_columnas': top_8_list,
        'peso_acumulado_top8': float(top_8_df['Importancia'].sum())
    }, f, indent=4, ensure_ascii=False)

print(f"\n -> Archivo guardado con éxito en: {RUTA_TOP8_OUT}")
print("==================================================================")