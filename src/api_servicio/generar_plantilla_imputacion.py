import os
import sys
import json
import joblib
import pandas as pd

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Ajustamos las rutas relativas considerando que el script vive en 'src/api_servicio/'
CARPETA_GEO_OUT  = "../../outputs/geo_evaluacion/"
RUTA_MUESTRA_GEO = os.path.join(CARPETA_GEO_OUT, "muestra_geocodificada_evaluacion.joblib")
RUTA_TOP8_JSON   = os.path.join(CARPETA_GEO_OUT, "top_8_atributos.json")
RUTA_PLANTILLA   = os.path.join(CARPETA_GEO_OUT, "plantilla_imputacion_49.json")

print("==================================================================")
print("  ESTRATEGIA B: GENERACIÓN DE PLANTILLA DE IMPUTACIÓN (49 VARS)   ")
print("==================================================================")

# --- 2. VALIDAR Y CARGAR ARCHIVOS BASE ---
if not os.path.exists(RUTA_MUESTRA_GEO):
    print(f"ERROR: No se encuentra la muestra geocodificada en {RUTA_MUESTRA_GEO}")
    sys.exit(1)

if not os.path.exists(RUTA_TOP8_JSON):
    print(f"ERROR: No se encuentra el JSON de top 8 atributos en {RUTA_TOP8_JSON}")
    sys.exit(1)

# Cargar las 8 columnas del formulario web
with open(RUTA_TOP8_JSON, 'r', encoding='utf-8') as f:
    top_8_cols = json.load(f)['top_8_columnas']

# Cargar la matriz de datos completa para extraer las 49 columnas y sus valores históricos
datos_geo = joblib.load(RUTA_MUESTRA_GEO)
X_geo = datos_geo['X_geo']  # DataFrame con las 49 columnas

print(f" -> Matriz histórica cargada: {X_geo.shape[1]} columnas detectadas.")
print(f" -> Variables del formulario web (Top 8): {len(top_8_cols)} columnas.")

# --- 3. CONSTRUIR DICCIONARIO DE VALORES POR DEFECTO ---
plantilla_defecto = {}
columnas_imputadas = 0

for col in X_geo.columns:
    if col in top_8_cols:
        # Las 8 variables principales se dejan como Nulas (se llenan dinámicamente con la petición web)
        plantilla_defecto[col] = None
    else:
        columnas_imputadas += 1
        # Para variables dummificadas o enteras (0 o 1), usar la Moda
        if X_geo[col].dtype in ['int64', 'int32', 'uint8', 'bool']:
            valor_moda = int(X_geo[col].mode()[0])
            plantilla_defecto[col] = valor_moda
        # Para variables numéricas continuas, usar la Mediana
        else:
            valor_mediana = float(X_geo[col].median())
            plantilla_defecto[col] = valor_mediana

# --- 4. EXPORTAR ESTRUCTURA A JSON ---
estructura_salida = {
    'orden_columnas_49': X_geo.columns.tolist(),
    'columnas_web_top8': top_8_cols,
    'total_columnas_imputadas': columnas_imputadas,
    'valores_defecto': plantilla_defecto
}

with open(RUTA_PLANTILLA, 'w', encoding='utf-8') as f:
    json.dump(estructura_salida, f, indent=4, ensure_ascii=False)

print("\n------------------------------------------------------------------")
print(f" -> Plantilla generada exitosamente con {columnas_imputadas} atributos de fondo.")
print(f" -> Archivo guardado en: {RUTA_PLANTILLA}")
print("==================================================================")