import os
import re
import json
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importación segura del geocodificador gratuito (Nominatim / OpenStreetMap)
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_DISPONIBLE = True
except ImportError:
    GEOPY_DISPONIBLE = False

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

# Inicializar cliente de geocodificación Nominatim
geocoder = Nominatim(user_agent="proyecto_grado_riesgo_vial_bogota_2026", timeout=5) if GEOPY_DISPONIBLE else None

# --- 3. INICIALIZACIÓN DE FASTAPI ---
app = FastAPI(
    title="API de Predicción de Riesgo Vial - Bogotá",
    description="Microservicio Python para inferencia ML (Estrategia A/B) y geocodificación espacial",
    version="1.2.0"
)

# Habilitar CORS para permitir peticiones desde Angular / Spring Boot u otras máquinas/puertos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. FUNCIÓN AUXILIAR DE GEOCODIFICACIÓN (BOGOTÁ) ---
def normalizar_direccion_bogota(direccion: str) -> str:
    """Expande abreviaturas viales de Bogotá y asegura el contexto geográfico."""
    texto = direccion.strip()
    reemplazos = [
        (r'\b(ak|av\.?\s*kr|av\.?\s*cra)\b', 'Avenida Carrera'),
        (r'\b(ac|av\.?\s*cl|av\.?\s*cll)\b', 'Avenida Calle'),
        (r'\b(kr|cra|k)\b', 'Carrera'),
        (r'\b(cl|cll|c)\b', 'Calle'),
        (r'\b(dg|diag)\b', 'Diagonal'),
        (r'\b(tv|transv)\b', 'Transversal'),
        (r'\b(av)\b', 'Avenida'),
    ]
    for patron, reemplazo in reemplazos:
        texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)

    if "bogota" not in texto.lower() and "bogotá" not in texto.lower():
        texto = f"{texto}, Bogotá, Colombia"
    elif "colombia" not in texto.lower():
        texto = f"{texto}, Colombia"

    return texto


def geocodificar_evento(direccion: Optional[str]) -> dict:
    """Convierte una dirección en coordenadas (latitud, longitud) sin afectar el modelo ML."""
    if not direccion or not direccion.strip():
        return {
            "estado": "NO_SOLICITADA",
            "direccion_ingresada": None,
            "direccion_normalizada": None,
            "latitud": None,
            "longitud": None
        }

    if not GEOPY_DISPONIBLE or geocoder is None:
        return {
            "estado": "DEPENDENCIA_FALTANTE",
            "direccion_ingresada": direccion,
            "direccion_normalizada": None,
            "latitud": None,
            "longitud": None,
            "detalle": "Instale geopy en el entorno Python ejecutando: pip install geopy"
        }

    direccion_norm = normalizar_direccion_bogota(direccion)

    try:
        # Acotar búsqueda a Colombia y al área metropolitana de Bogotá
        ubicacion = geocoder.geocode(
            direccion_norm,
            country_codes="co",
            viewbox=[(4.83, -74.23), (4.46, -73.99)],
            bounded=True
        )

        if ubicacion:
            return {
                "estado": "ENCONTRADA",
                "direccion_ingresada": direccion,
                "direccion_normalizada": direccion_norm,
                "direccion_oficial": ubicacion.address,
                "latitud": round(float(ubicacion.latitude), 6),
                "longitud": round(float(ubicacion.longitude), 6)
            }
        else:
            return {
                "estado": "NO_ENCONTRADA",
                "direccion_ingresada": direccion,
                "direccion_normalizada": direccion_norm,
                "latitud": None,
                "longitud": None,
                "detalle": "No se encontraron coordenadas exactas en Bogotá para la dirección ingresada."
            }

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return {
            "estado": "ERROR_SERVICIO",
            "direccion_ingresada": direccion,
            "direccion_normalizada": direccion_norm,
            "latitud": None,
            "longitud": None,
            "detalle": f"El servicio de geocodificación no respondió a tiempo: {str(e)}"
        }


# --- 5. ESQUEMA DE DATOS ENVIADOS DESDE LA WEB ---
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
    direccion: Optional[str] = Field(None, example="Calle 26 # 68-75, Bogotá", description="Campo localizador opcional para ubicar el evento en el mapa (no entra al modelo)")


# --- 6. ENDPOINT DE PREDICCIÓN Y GEORREFERENCIACIÓN ---
@app.post("/api/v1/predecir-riesgo")
def predecir_riesgo(datos: SolicitudPrediccion):
    # Map estricto con los 8 atributos predictivos del modelo
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

    # Ejecutar geocodificación independiente del 9.º campo localizador
    resultado_ubicacion = geocodificar_evento(datos.direccion)

    return {
        "estrategia_aplicada": estrategia_elegida,
        "prediccion_codigo": clase_predicha,
        "prediccion_descripcion": mapa_clases[clase_predicha],
        "probabilidades": {
            "solo_danos": round(prob_danos, 4),
            "con_heridos": round(prob_heridos, 4),
            "con_muertos": round(prob_fatal, 4)
        },
        "nivel_riesgo": nivel_riesgo,
        "ubicacion": resultado_ubicacion
    }