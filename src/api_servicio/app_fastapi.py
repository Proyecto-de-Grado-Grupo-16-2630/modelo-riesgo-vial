import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. CONFIGURACIÓN DE RUTAS Y ARTEFACTOS ---
CARPETA_GEO     = "../../outputs/geo_evaluacion/"
CARPETA_MODELOS = "../../outputs/modelos_optimizados/"

RUTA_MODELO_LITE = os.path.join(CARPETA_MODELOS, "multiclase_XGBoost_Lite.joblib")
RUTA_MODELO_FULL = os.path.join(CARPETA_MODELOS, "multiclase_XGBoost.joblib")
RUTA_PLANTILLA   = os.path.join(CARPETA_GEO, "plantilla_imputacion_49.json")

# --- 2. CARGA DE MODELOS Y PLANTILLA AL INICIAR LA API ---
if not os.path.exists(RUTA_MODELO_LITE):
    raise FileNotFoundError(f"No se encontró el modelo Lite en: {RUTA_MODELO_LITE}")

modelo_lite = joblib.load(RUTA_MODELO_LITE)
modelo_full = joblib.load(RUTA_MODELO_FULL) if os.path.exists(RUTA_MODELO_FULL) else None

# Carga opcional de la plantilla de imputación para soporte a la Estrategia B
config_imputacion = None
ORDEN_49 = []
VALORES_DEFECTO = {}

if os.path.exists(RUTA_PLANTILLA):
    with open(RUTA_PLANTILLA, 'r', encoding='utf-8') as f:
        config_imputacion = json.load(f)
    ORDEN_49 = config_imputacion.get('orden_columnas_49', [])
    VALORES_DEFECTO = config_imputacion.get('valores_defecto', {})

# --- 3. INICIALIZACIÓN DE FASTAPI ---
app = FastAPI(
    title="API de Predicción de Riesgo Vial - Bogotá",
    description="Microservicio Python para inferencia del modelo Lite (Estrategia A predeterminada)",
    version="1.1.0"
)

# Habilitar CORS para permitir peticiones desde Angular / Spring Boot u otras máquinas/puertos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. ESQUEMA DE DATOS ENVIADOS DESDE LA WEB ---
class SolicitudPrediccion(BaseModel):
    total_actores: int = Field(..., example=3, description="Número de personas involucradas")
    total_vehiculos: int = Field(..., example=2, description="Número de vehículos involucrados")
    cant_ciclistas: int = Field(0, example=1, description="Número de ciclistas involucrados")
    choque_vehiculo: int = Field(..., example=1, description="1 si colisionó contra vehículo, 0 si no")
    choque_sin_registro: int = Field(0, example=0, description="1 si no hay choque registrado (caída), 0 si no")
    objeto_fijo_no_aplica: int = Field(1, example=1, description="1 si NO impactó objeto fijo, 0 si impactó")
    clase_camion_furgon: int = Field(0, example=0, description="1 si participó camión o furgón, 0 si no")
    clase_campero: int = Field(0, example=0, description="1 si participó campero, 0 si no")
    estrategia: str = Field("A", example="A", description="Estrategia a usar: 'A' (Lite 8 vars - Predeterminada) o 'B' (Imputación 49 vars)")


# --- 5. ENDPOINT DE PREDICCIÓN ---
@app.post("/api/v1/predecir-riesgo")
def predecir_riesgo(datos: SolicitudPrediccion):
    # Map con los 8 atributos recibidos del frontend
    input_top8 = {
        "CHOQUE_DESC_sin choque registrado": datos.choque_sin_registro,
        "CHOQUE_DESC_vehiculo": datos.choque_vehiculo,
        "CLASE_DESC_camion, furgon": datos.clase_camion_furgon,
        "cant_ciclistas": datos.cant_ciclistas,
        "CLASE_DESC_campero": datos.clase_campero,
        "total_vehiculos": datos.total_vehiculos,
        "total_actores": datos.total_actores,
        "OBJETO_FIJO_DESC_no aplica (no es objeto fijo)": datos.objeto_fijo_no_aplica
    }

    estrategia_elegida = datos.estrategia.upper()

    if estrategia_elegida == "A":
        # ESTRATEGIA A: Inferencia directa con el Modelo Lite (8 atributos)
        df_input = pd.DataFrame([input_top8])
        probabilidades = modelo_lite.predict_proba(df_input)[0]

    elif estrategia_elegida == "B":
        # ESTRATEGIA B: Imputación con plantilla de 49 columnas (Soporte secundario)
        if modelo_full is None or not ORDEN_49:
            raise HTTPException(status_code=500, detail="El modelo completo (49 vars) o la plantilla de imputación no están disponibles.")
        
        vector_49 = dict(VALORES_DEFECTO)
        for col, val in input_top8.items():
            vector_49[col] = val

        df_input = pd.DataFrame([vector_49])[ORDEN_49]
        probabilidades = modelo_full.predict_proba(df_input)[0]

    else:
        raise HTTPException(status_code=400, detail="Estrategia no válida. Especifique 'A' o 'B'.")

    # Extraer probabilidades por clase (0: Solo Daños, 1: Con Heridos, 2: Con Muertos)
    prob_danos   = float(probabilidades[0])
    prob_heridos = float(probabilidades[1])
    prob_fatal   = float(probabilidades[2])

    clase_predicha = int(probabilidades.argmax())
    mapa_clases = {0: "Solo Daños", 1: "Con Heridos", 2: "Con Muertos"}

    # Determinar nivel de alerta operativa
    if prob_fatal >= 0.35:
        nivel_riesgo = "CRÍTICO"
    elif prob_fatal >= 0.15 or prob_heridos >= 0.50:
        nivel_riesgo = "ALTO"
    elif prob_heridos >= 0.30:
        nivel_riesgo = "MEDIO"
    else:
        nivel_riesgo = "BAJO"

    return {
        "estrategia_aplicada": estrategia_elegida,
        "prediccion_codigo": clase_predicha,
        "prediccion_descripcion": mapa_clases[clase_predicha],
        "probabilidades": {
            "solo_danos": round(prob_danos, 4),
            "con_heridos": round(prob_heridos, 4),
            "con_muertos": round(prob_fatal, 4)
        },
        "nivel_riesgo": nivel_riesgo
    }