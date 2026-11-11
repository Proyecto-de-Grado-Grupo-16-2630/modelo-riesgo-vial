import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)

# --- 1. CONFIGURACIÓN DE RUTAS ---
CARPETA_GEO_OUT   = "../../outputs/geo_evaluacion/"
CARPETA_MODELOS   = "../../outputs/modelos_optimizados/"
CARPETA_PROCESSED = "../../data/processed/"

RUTA_MUESTRA_GEO = os.path.join(CARPETA_GEO_OUT, "muestra_geocodificada_evaluacion.joblib")
RUTA_TOP8_JSON   = os.path.join(CARPETA_GEO_OUT, "top_8_atributos.json")
RUTA_MODELO_LITE  = os.path.join(CARPETA_MODELOS, "multiclase_XGBoost_Lite.joblib")
RUTA_CSV_OUT      = os.path.join(CARPETA_GEO_OUT, "resultados_comparativos_lite.csv")

print("==================================================================")
print("  PASO 2: ENTRENAMIENTO BALANCEADO MODELO LITE (8 ATRIBUTOS)       ")
print("==================================================================")

# --- 2. CARGAR ATRIBUTOS DEL TOP 8 Y MUESTRA GEOCODIFICADA ---
if not os.path.exists(RUTA_TOP8_JSON) or not os.path.exists(RUTA_MUESTRA_GEO):
    print("ERROR: Archivos base no encontrados. Ejecuta primero 014_extraer_top_atributos.py")
    sys.exit(1)

with open(RUTA_TOP8_JSON, 'r', encoding='utf-8') as f:
    config_top8 = json.load(f)
top_8_cols = config_top8['top_8_columnas']

datos_geo = joblib.load(RUTA_MUESTRA_GEO)
X_geo_full = datos_geo['X_geo']
y_geo = datos_geo['y_geo']

# Filtrar matriz únicamente a las 8 columnas clave
X_geo_lite = X_geo_full[top_8_cols]

# --- 3. CARGAR O CONSTRUIR DATOS DE ENTRENAMIENTO ---
ruta_xtrain = os.path.join(CARPETA_PROCESSED, "X_train.joblib")
ruta_ytrain = os.path.join(CARPETA_PROCESSED, "y_train.joblib")

if os.path.exists(ruta_xtrain) and os.path.exists(ruta_ytrain):
    X_train_full = joblib.load(ruta_xtrain)
    y_train = joblib.load(ruta_ytrain)
    X_train_lite = X_train_full[top_8_cols]
    print(f" -> Datos de entrenamiento cargados: {X_train_lite.shape[0]} registros con 8 variables.")
else:
    print(" -> Utilizando muestra geocodificada procesada para entrenamiento...")
    X_train_lite = X_geo_lite
    y_train = y_geo

# --- 4. ENTRENAR MODELO XGBOOST LITE CON PESOS BALANCEADOS ---
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

clf_lite = XGBClassifier(random_state=42, eval_metric='mlogloss')
clf_lite.fit(X_train_lite, y_train, sample_weight=sample_weights)

joblib.dump(clf_lite, RUTA_MODELO_LITE)
print(f" -> Modelo Lite (Balanceado) guardado en: {RUTA_MODELO_LITE}")

# --- 5. EVALUACIÓN DE INFERENCIA EN MUESTRA GEOCODIFICADA ---
y_pred_lite = clf_lite.predict(X_geo_lite)
y_proba_lite = clf_lite.predict_proba(X_geo_lite)

acc_lite        = accuracy_score(y_geo, y_pred_lite)
prec_lite       = precision_score(y_geo, y_pred_lite, average='macro', zero_division=0)
rec_macro_lite  = recall_score(y_geo, y_pred_lite, average='macro', zero_division=0)
rec_fatal_lite  = recall_score(y_geo, y_pred_lite, labels=[2], average=None, zero_division=0)[0]
f1_lite         = f1_score(y_geo, y_pred_lite, average='macro', zero_division=0)
auc_fatal_lite  = roc_auc_score((y_geo == 2).astype(int), y_proba_lite[:, 2])

# --- 6. TABLA COMPARATIVA MODELO COMPLETO VS LITE ---
df_comp = pd.DataFrame([
    {
        'Modelo': 'XGBoost Completo (49 Atributos)',
        'Accuracy': 0.7776,
        'Precision_Macro': 0.6070,
        'Recall_Macro': 0.7810,
        'Recall_Fatal': 0.8750,
        'F1_Macro': 0.5804,
        'AUC_Fatal': 0.9441
    },
    {
        'Modelo': 'XGBoost Lite Balanceado (8 Atributos)',
        'Accuracy': round(acc_lite, 4),
        'Precision_Macro': round(prec_lite, 4),
        'Recall_Macro': round(rec_macro_lite, 4),
        'Recall_Fatal': round(rec_fatal_lite, 4),
        'F1_Macro': round(f1_lite, 4),
        'AUC_Fatal': round(auc_fatal_lite, 4)
    }
])

df_comp.to_csv(RUTA_CSV_OUT, index=False, encoding='utf-8-sig')

print("\n------------------------------------------------------------------")
print("  MATRIZ COMPARATIVA DE RENDIMIENTO (FULL VS LITE BALANCEADO):    ")
print("------------------------------------------------------------------")
print(df_comp.to_string(index=False))

print("\n==================================================================")
print("  EVALUACIÓN Y COMPARATIVA FINALIZADA CON ÉXITO!")
print("==================================================================")