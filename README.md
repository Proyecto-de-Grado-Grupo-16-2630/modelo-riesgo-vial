# Pipeline de Analítica, Modelado Predictivo de Siniestros Viales y Microservicio API

Este proyecto contiene una solución integral de Ciencia de Datos y Machine Learning en Python para la ingesta, análisis exploratorio de datos (EDA), limpieza estructurada, ingeniería de características espacio-temporales, entrenamiento y optimización de modelos predictivos de severidad vial (clasificación multiclase y binaria balanceada), validación geoespacial en Bogotá, reducción de dimensionalidad y puesta en producción de un microservicio de inferencia en tiempo real mediante **FastAPI**.

El pipeline abarca desde el procesamiento del registro histórico bruto en Excel (`registro_accidentes.xlsx`) y datos espaciales (`2500_geocodificados_2498.geojson`), hasta la generación de un archivo GeoJSON enriquecido con predicciones para sistemas de información geográfica (SIG) y una API REST capaz de evaluar el riesgo de fatalidad en tiempo real mediante dos estrategias operativas (Modelo Ligero vs. Imputación Inteligente).

---

## Tabla de Contenido
1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Instalación y Requisitos](#instalación-y-requisitos)
3. [Flujo de Ejecución del Pipeline (7 Fases)](#flujo-de-ejecución-del-pipeline)
   - [Fase 1: Diagnóstico Exploratorio Inicial de Nulos](#fase-1-diagnóstico-exploratorio-inicial-de-nulos)
   - [Fase 2: Limpieza y Reglas de Negocio](#fase-2-limpieza-y-reglas-de-negocio)
   - [Fase 3: Análisis Exploratorio Avanzado (EDA Post-Limpieza)](#fase-3-análisis-exploratorio-avanzado-eda-post-limpieza)
   - [Fase 4: Ensamble Relacional, Enriquecimiento Temporal y Matriz ML](#fase-4-ensamble-relacional-enriquecimiento-temporal-y-matriz-ml)
   - [Fase 5: Modelado Supervisado, Optimización y Balanceo](#fase-5-modelado-supervisado-optimización-y-balanceo)
   - [Fase 6: Validación Geoespacial, Ranking de Atributos y Modelo Lite](#fase-6-validación-geoespacial-ranking-de-atributos-y-modelo-lite)
   - [Fase 7: Despliegue del Microservicio de Predicción (FastAPI)](#fase-7-despliegue-del-microservicio-de-predicción-fastapi)
4. [Documentación de la API REST de Inferencia](#documentación-de-la-api-rest-de-inferencia)
   - [Estrategias de Inferencia (A vs. B)](#estrategias-de-inferencia)
   - [Ejecución del Servidor y Soporte CORS](#ejecución-del-servidor-y-soporte-cors)
   - [Especificación de Endpoints y Esquema JSON](#especificación-de-endpoints-y-esquema-json)
   - [Niveles de Alerta y Reglas de Decisión](#niveles-de-alerta-y-reglas-de-decisión)
   - [Ejemplos de Consumo con cURL y PowerShell](#ejemplos-de-consumo)
5. [Resultados y Métricas Empíricas](#resultados-y-métricas-empíricas)
   - [Comparativa Modelo Completo (49 Atributos) vs. Lite (8 Atributos)](#comparativa-completo-vs-lite)
   - [Rendimiento en Experimento Multiclase Balanceado (3x2.951 casos)](#rendimiento-multiclase-balanceado)
   - [Rendimiento en Clasificación Binaria Balanceada (50/50)](#rendimiento-clasificación-binaria)
6. [Entregables y Artefactos del Proyecto](#entregables-y-artefactos-del-proyecto)

---

## Estructura del Proyecto

La organización completa de carpetas, módulos de código y artefactos generados es la siguiente:

```text
.
├── data/
│   ├── raw/                                  # Datasets originales sin procesar
│   │   └── registro_accidentes.xlsx          # Registro histórico bruto de siniestros viales
│   └── processed/                            # Datos procesados, limpios y particionados
│       ├── 2500_geocodificados_2498.geojson  # Muestra geoespacial de siniestros en Bogotá
│       ├── siniestros_limpio.csv             # Entidad siniestros limpia
│       ├── actores_limpio.csv                # Entidad actores viales limpia
│       ├── vehiculos_limpio.csv              # Entidad vehículos limpia
│       ├── hipotesis_limpio.csv              # Entidad hipótesis y causas limpia
│       ├── vista_minable_accidente.tsv       # Consolidado relacional a nivel de accidente
│       ├── dataset_listo_para_ml.tsv         # Matriz con One-Hot Encoding (n-1) y Min-Max Scaling
│       └── datos_clasificacion.joblib        # Diccionario con particiones 70/30 (X/y) y class_weight
├── src/
│   ├── exploracion/                          # Módulo de Análisis Exploratorio (EDA)
│   │   ├── cargar_y_analizar_nulos.py        # Diagnóstico inicial de nulos en datos brutos
│   │   ├── auditoria_faltantes_post_limpieza.py # Auditoría post-limpieza (falsos nulos vs justificantes)
│   │   ├── histogramas_y_distribuciones.py   # Distribuciones de edad, horas pico y severidad
│   │   └── analisis_bivariado_y_cruces.py   # Cruces de gravedad vs actores, franjas y causas
│   ├── limpieza/                             # Módulo de Limpieza y Reglas de Negocio
│   │   ├── limpiar_y_exportar.py             # Mapeos categóricos y corrección de nulos
│   │   ├── verificar_limpieza.py             # Verificación lógica de integridad referencial
│   │   └── consolidar_excel.py               # Generación del archivo Excel estructurado multi-hoja
│   ├── preparacion/                          # Módulo de Feature Engineering y Ensamble ML
│   │   ├── crear_vista_minable_accidente.py  # Ensamble relacional a nivel de accidente
│   │   ├── auditoria_vista_minable.py        # Inspección de integridad, tipos y nulos
│   │   ├── enriquecer_vista_minable.py       # Enriquecimiento: festivos Colombia, fines de semana y franjas
│   │   └── preparar_dataset_ml.py            # Encoding (n-1), Min-Max Scaling y Target ordinal
│   ├── modelado/                             # Módulo de Modelado Predictivo Supervisado
│   │   ├── configurar_clasificacion.py       # Partición estratificada 70/30 y cálculo de class_weight
│   │   ├── regresion_lineal_preliminar.py    # Modelo Baseline preliminar (Regresión Lineal)
│   │   ├── entrenar_regresion_logistica.py   # Clasificador Regresión Logística
│   │   ├── entrenar_random_forest.py         # Clasificador Random Forest
│   │   ├── entrenar_svm.py                   # Clasificador Support Vector Machine (LinearSVC + Calibración)
│   │   ├── entrenar_red_neuronal.py          # Clasificador Red Neuronal (MLPClassifier)
│   │   ├── entrenar_xgboost.py               # Clasificador XGBoost multiclase
│   │   ├── busqueda_hiperparametros.py       # Optimización por GridSearchCV con validación cruzada
│   │   ├── clasificacion_binaria_balanceada.py # Evaluación binaria (No Fatales vs Fatales, 50/50)
│   │   └── clasificacion_multiclase_balanceada.py # Evaluación multiclase balanceada (3x2.951 casos)
│   ├── geo_analisis/                         # Módulo de Validación Geoespacial y Modelo Lite
│   │   ├── filtrado_muestra_geocodificada.py # Extracción y cruce de muestra geocodificada
│   │   ├── evaluacion_modelos_geocodificados.py # Evaluación de modelos y enriquecimiento del GeoJSON
│   │   ├── generar_matrices_confusion_geo.py # Matrices de confusión de modelos en muestra espacial
│   │   ├── extraer_top_atributos.py          # Ranking de importancia de características (Top 8 XGBoost)
│   │   └── entrenar_evaluar_modelo_lite.py   # Entrenamiento y evaluación de XGBoost Lite (8 variables)
│   └── api_servicio/                         # Módulo de Microservicio REST para Inferencia
│       ├── generar_plantilla_imputacion.py   # Generador de plantilla de modas y medianas históricas
│       └── app_fastapi.py                    # API REST con FastAPI (v1.1.0) y soporte CORS
├── outputs/                                  # Reportes, gráficos y artefactos de salida
│   ├── modelos_optimizados/                  # Modelos entrenados y serializados en Joblib (.joblib)
│   │   ├── multiclase_XGBoost.joblib         # Modelo Completo (49 variables)
│   │   ├── multiclase_XGBoost_Lite.joblib    # Modelo Ligero (8 variables principales)
│   │   ├── multiclase_Random_Forest.joblib   # Modelo Random Forest multiclase
│   │   ├── binario_XGBoost.joblib            # Modelo XGBoost para clasificación binaria
│   │   ├── binario_Random_Forest.joblib      # Modelo Random Forest para clasificación binaria
│   │   └── cm_*.png, roc_curve_*.png         # Matrices de confusión y curvas ROC de modelos
│   ├── geo_evaluacion/                       # Artefactos y evaluaciones espaciales
│   │   ├── 2500_geocodificados_con_predicciones.geojson # GeoJSON enriquecido con predicciones (SIG/QGIS)
│   │   ├── predicciones_geocodificadas_completas.csv   # Predicciones por código de accidente
│   │   ├── resultados_metricas_geocodificados.csv     # Métricas de modelos en muestra geoespacial
│   │   ├── muestra_geocodificada_evaluacion.joblib    # Muestra filtrada serializada
│   │   ├── top_8_atributos.json              # Listado y peso acumulado (78.02%) del Top 8
│   │   ├── plantilla_imputacion_49.json      # Modas y medianas para la Estrategia B de imputación
│   │   ├── resultados_comparativos_lite.csv  # Métricas comparativas Modelo Completo vs Lite
│   │   └── cm_geo_*.png                      # Matrices de confusión sobre muestra geocodificada
│   ├── registro_accidentes_limpio.xlsx       # Libro Excel estructurado multi-hoja limpio final
│   ├── reporte_*.txt                         # Reportes técnicos detallados por modelo y fase
│   ├── resultados_*.csv                      # Tablas comparativas de experimentos
│   └── *.png                                 # Visualizaciones EDA, distribuciones y residuos
├── requirements.txt                          # Especificación de librerías y dependencias
└── README.md                                 # Documentación técnica integral del proyecto
```

---

## Instalación y Requisitos

El proyecto requiere **Python 3.10+** (o compatible con Python 3.8+).

### Dependencias Principales:
* **Manipulación de Datos y Matemáticas:** `pandas`, `numpy`, `openpyxl`, `joblib`.
* **Análisis Exploratorio y Diagnóstico:** `missingno`, `matplotlib`, `seaborn`.
* **Enriquecimiento Espacio-Temporal:** `holidays` (festivos oficiales y traslados de la Ley Emiliani en Colombia).
* **Machine Learning y Clasificación:** `scikit-learn`, `xgboost`.
* **Microservicio e Inferencia Web:** `fastapi`, `uvicorn`, `pydantic`.

### Configuración del Entorno:

1. **Clonar o situarse en el directorio raíz del proyecto:**
   ```bash
   cd "codigo data set"
   ```

2. **Crear y activar el entorno virtual:**
   * En Windows (PowerShell / CMD):
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * En macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Flujo de Ejecución del Pipeline

El pipeline metodológico se organiza en 7 fases secuenciales y reproducibles:

### Fase 1: Diagnóstico Exploratorio Inicial de Nulos
Evalúa el porcentaje de valores faltantes y su tipología en el dataset original sin alterar la fuente:
```bash
python src/exploracion/cargar_y_analizar_nulos.py
```
* **Salidas:** Estadísticas en `outputs/reporte_inicial_nulos.txt` y diagramas de barras/matrices en `outputs/faltantes_*.png`.

### Fase 2: Limpieza y Reglas de Negocio
Aplica normalización categórica, homologación de códigos, resolución de faltantes condicionales e integridad referencial:
```bash
# 1. Aplicar mapeos categóricos, tratar nulos y exportar CSVs limpios
python src/limpieza/limpiar_y_exportar.py

# 2. Verificar la calidad e integridad lógica de la limpieza
python src/limpieza/verificar_limpieza.py

# 3. Consolidar los CSVs limpios en un libro Excel estructurado multi-hoja
python src/limpieza/consolidar_excel.py
```
* **Salidas:** Archivos limpios en `data/processed/` (`siniestros_limpio.csv`, `actores_limpio.csv`, `vehiculos_limpio.csv`, `hipotesis_limpio.csv`), `outputs/reporte_verificacion_limpieza.txt` y `outputs/registro_accidentes_limpio.xlsx`.

### Fase 3: Análisis Exploratorio Avanzado (EDA Post-Limpieza)
Analiza correlaciones, distribuciones poblacionales y cruces estratégicos de severidad:
```bash
# 1. Auditoría de clasificación de nulos post-limpieza (falsos nulos vs justificantes)
python src/exploracion/auditoria_faltantes_post_limpieza.py

# 2. Distribución de edades de involucrados, horas pico y severidad
python src/exploracion/histogramas_y_distribuciones.py

# 3. Análisis bivariado (Gravedad vs Actores, Franjas Horarias y Causas)
python src/exploracion/analisis_bivariado_y_cruces.py
```
* **Salidas:** Reportes de texto en `outputs/reporte_*.txt` y dashboards de alta resolución (`histograma_edades.png`, `histograma_horas_pico.png`, `distribucion_gravedad.png`, `cruce_gravedad_actor.png`, `cruce_gravedad_franja.png`, `top_10_causas.png`).

### Fase 4: Ensamble Relacional, Enriquecimiento Temporal y Matriz ML
Construye la vista minable consolidada por accidente (`codigo_accidente`), enriquece con el calendario oficial colombiano y realiza el preprocesamiento matemático para Machine Learning:
```bash
# 1. Ensamble relacional unificando siniestros, actores, vehículos e hipótesis
python src/preparacion/crear_vista_minable_accidente.py

# 2. Auditar la vista minable preliminar (tipos de datos, columnas y nulidad)
python src/preparacion/auditoria_vista_minable.py

# 3. Enriquecimiento temporal con festivos oficiales de Colombia, fines de semana y franjas horarias
python src/preparacion/enriquecer_vista_minable.py

# 4. Generar dataset para ML (One-Hot Encoding n-1 y Min-Max Scaling)
python src/preparacion/preparar_dataset_ml.py
```
* **Salidas:**
  * `data/processed/vista_minable_accidente.tsv` con variables agregadas (`total_actores`, `edad_promedio`, `cant_peatones`, `cant_motociclistas`, `cant_ciclistas`, `total_vehiculos`, `hubo_fuga`, etc.) y temporales (`es_festivo`, `es_fin_de_semana`, `es_dia_habil`, `franja_horaria`).
  * `data/processed/dataset_listo_para_ml.tsv` (49 variables predictoras escaladas $[0, 1]$ + `target_gravedad` ordinal: $0: \text{Solo Daños}, 1: \text{Con Heridos}, 2: \text{Con Muertos}$).

### Fase 5: Modelado Supervisado, Optimización y Balanceo
Aísla conjuntos de datos mediante partición estratificada 70/30, entrena clasificadores, optimiza hiperparámetros y aplica técnicas de balanceo para mitigar el sesgo hacia la clase mayoritaria:
```bash
# 1. Configurar división 70/30 estratificada y calcular pesos de balanceo (class_weight)
python src/modelado/configurar_clasificacion.py

# 2. Entrenar y evaluar clasificadores base individuales
python src/modelado/entrenar_regresion_logistica.py
python src/modelado/entrenar_random_forest.py
python src/modelado/entrenar_svm.py
python src/modelado/entrenar_red_neuronal.py
python src/modelado/entrenar_xgboost.py

# 3. Búsqueda de hiperparámetros con GridSearchCV (StratifiedKFold k=3)
python src/modelado/busqueda_hiperparametros.py

# 4. Experimento controlado de clasificación binaria balanceada (50/50)
python src/modelado/clasificacion_binaria_balanceada.py

# 5. Experimento controlado de clasificación multiclase balanceada (3x2.951 casos)
python src/modelado/clasificacion_multiclase_balanceada.py
```
* **Salidas:** `data/processed/datos_clasificacion.joblib`, modelos serializados en `outputs/modelos_optimizados/` (`multiclase_XGBoost.joblib`, `multiclase_Random_Forest.joblib`, `binario_*.joblib`, etc.), tablas de resultados en `.csv` y curvas ROC/matrices de confusión.

### Fase 6: Validación Geoespacial, Ranking de Atributos y Modelo Lite
Evalúa el desempeño predictivo en la muestra georreferenciada de Bogotá, extrae el ranking de importancia y entrena el modelo ligero para el microservicio:
```bash
# 1. Filtrar y cruzar la muestra espacial de 2.500 accidentes del GeoJSON con el dataset procesado
python src/geo_analisis/filtrado_muestra_geocodificada.py

# 2. Evaluar el desempeño de los modelos sobre la muestra espacial y generar GeoJSON enriquecido
python src/geo_analisis/evaluacion_modelos_geocodificados.py

# 3. Generar matrices de confusión sobre la muestra espacial
python src/geo_analisis/generar_matrices_confusion_geo.py

# 4. Extraer el ranking de importancia de atributos (Top 8 de XGBoost con 78.02% de peso acumulado)
python src/geo_analisis/extraer_top_atributos.py

# 5. Entrenar y evaluar el modelo ligero balanceado (XGBoost Lite de 8 variables)
python src/geo_analisis/entrenar_evaluar_modelo_lite.py
```
* **Salidas:**
  * `outputs/geo_evaluacion/2500_geocodificados_con_predicciones.geojson`: GeoJSON con coordenadas geográficas y propiedades predictivas integradas (`PRED_RF_COD`, `PRED_RF_DESC`, `PROB_FATAL_RF`, `PRED_XGB_COD`, `PRED_XGB_DESC`, `PROB_FATAL_XGB`) para visualización en QGIS o Mapbox.
  * `outputs/geo_evaluacion/top_8_atributos.json`: Estructura JSON con las 8 variables más determinantes.
  * `outputs/modelos_optimizados/multiclase_XGBoost_Lite.joblib`: Modelo compacto balanceado para inferencia rápida.
  * `outputs/geo_evaluacion/resultados_comparativos_lite.csv`: Comparativa de métricas Full vs Lite.

### Fase 7: Despliegue del Microservicio de Predicción (FastAPI)
Genera la plantilla de modas y medianas para imputación en caliente y levanta el microservicio web:
```bash
# 1. Generar la plantilla de imputación de fondo (modas y medianas de las 41 variables no ingresadas)
python src/api_servicio/generar_plantilla_imputacion.py

# 2. Iniciar el microservicio con Uvicorn (desde src/api_servicio/)
cd src/api_servicio
uvicorn app_fastapi:app --reload --host 127.0.0.1 --port 8000
```
* **Salidas:** `outputs/geo_evaluacion/plantilla_imputacion_49.json` y servidor web activo en `http://127.0.0.1:8000`.

---

## Documentación de la API REST de Inferencia

El microservicio desarrollado en [`src/api_servicio/app_fastapi.py`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/api_servicio/app_fastapi.py) (versión 1.1.0) implementa un servicio REST de alta velocidad para inferencia probabilística sobre la gravedad de siniestros viales.

### Estrategias de Inferencia

El servicio cuenta con dos estrategias de inferencia seleccionables en cada petición:

* **Estrategia A (Modelo Lite - 8 Atributos) [Predeterminada]:**
  * Utiliza el clasificador `multiclase_XGBoost_Lite.joblib`, reentrenado con pesos balanceados exclusivamente sobre las 8 variables del formulario web.
  * Garantiza mínima latencia de cómputo y no requiere cargar la matriz histórica de 49 columnas.
* **Estrategia B (Imputación Inteligente - 49 Atributos) [Alta Capacidad Discriminante]:**
  * Emplea el modelo completo `multiclase_XGBoost.joblib` (49 variables).
  * Inyecta los 8 atributos provistos por el usuario y completa los 41 atributos restantes utilizando la moda (para variables discretas y categóricas dummificadas) y la mediana (para variables continuas) calculadas en [`plantilla_imputacion_49.json`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/outputs/geo_evaluacion/plantilla_imputacion_49.json).

### Ejecución del Servidor y Soporte CORS

El microservicio incorpora `CORSMiddleware` configurado con permisos completos (`allow_origins=["*"]`), permitiendo peticiones directas desde aplicaciones frontend (Angular, React, Vue, Spring Boot o tableros de control externos).

Para iniciar el microservicio:
```bash
# Opción 1: Situándose en la carpeta del servicio
cd src/api_servicio
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload

# Opción 2: Ejecutar desde la raíz del proyecto
uvicorn src.api_servicio.app_fastapi:app --host 127.0.0.1 --port 8000 --reload
```

Una vez levantado, se puede acceder a la interfaz de documentación interactiva:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Especificación de Endpoints y Esquema JSON

#### `POST /api/v1/predecir-riesgo`

##### Esquema de Entrada (Request Body):
```json
{
  "total_actores": 3,
  "total_vehiculos": 2,
  "cant_ciclistas": 1,
  "choque_vehiculo": 1,
  "choque_sin_registro": 0,
  "objeto_fijo_no_aplica": 1,
  "clase_camion_furgon": 0,
  "clase_campero": 0,
  "estrategia": "A"
}
```

##### Descripción de las 8 Variables de Entrada (Top 8 Atributos):
| Variable JSON | Tipo | Variable Interna Modelo | Descripción Operativa |
| :--- | :--- | :--- | :--- |
| `total_actores` | `int` | `total_actores` | Cantidad total de personas involucradas en el evento. |
| `total_vehiculos` | `int` | `total_vehiculos` | Cantidad total de vehículos participantes en el siniestro. |
| `cant_ciclistas` | `int` | `cant_ciclistas` | Número de ciclistas involucrados (actor vulnerable). |
| `choque_vehiculo` | `int` | `CHOQUE_DESC_vehiculo` | `1` si colisionó contra otro vehículo, `0` si no. |
| `choque_sin_registro` | `int` | `CHOQUE_DESC_sin choque registrado` | `1` si no hubo choque (ej. caída de ocupante o volcamiento), `0` si no. |
| `objeto_fijo_no_aplica` | `int` | `OBJETO_FIJO_DESC_no aplica (no es objeto fijo)` | `1` si NO colisionó con poste, muro o separador; `0` si impactó objeto fijo. |
| `clase_camion_furgon` | `int` | `CLASE_DESC_camion, furgon` | `1` si participó un vehículo pesado de carga, `0` si no. |
| `clase_campero` | `int` | `CLASE_DESC_campero` | `1` si participó un vehículo campero / 4x4, `0` si no. |
| `estrategia` | `str` | N/A | `"A"` para Modelo Lite (8 vars) o `"B"` para Imputación Inteligente (49 vars). |

##### Esquema de Salida (Response Body):
```json
{
  "estrategia_aplicada": "A",
  "prediccion_codigo": 1,
  "prediccion_descripcion": "Con Heridos",
  "probabilidades": {
    "solo_danos": 0.2854,
    "con_heridos": 0.5821,
    "con_muertos": 0.1325
  },
  "nivel_riesgo": "ALTO"
}
```

### Niveles de Alerta y Reglas de Decisión

Para asistir a centros de comando de emergencias y despacho de ambulancias, la API categoriza la predicción en 4 niveles de riesgo operativo:

$$\text{Nivel de Riesgo} = \begin{cases} 
\textbf{CRÍTICO} & \text{si } P(\text{Con Muertos}) \ge 0.35 \\ 
\textbf{ALTO} & \text{si } P(\text{Con Muertos}) \ge 0.15 \text{ o } P(\text{Con Heridos}) \ge 0.50 \\ 
\textbf{MEDIO} & \text{si } P(\text{Con Heridos}) \ge 0.30 \\ 
\textbf{BAJO} & \text{en cualquier otro caso} 
\end{cases}$$

### Ejemplos de Consumo

#### Con cURL (Bash / Linux / Git Bash):
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predecir-riesgo" \
  -H "Content-Type: application/json" \
  -d '{
    "total_actores": 3,
    "total_vehiculos": 2,
    "cant_ciclistas": 1,
    "choque_vehiculo": 1,
    "choque_sin_registro": 0,
    "objeto_fijo_no_aplica": 1,
    "clase_camion_furgon": 0,
    "clase_campero": 0,
    "estrategia": "A"
  }'
```

#### Con PowerShell (Windows):
```powershell
$body = @{
    total_actores = 3
    total_vehiculos = 2
    cant_ciclistas = 1
    choque_vehiculo = 1
    choque_sin_registro = 0
    objeto_fijo_no_aplica = 1
    clase_camion_furgon = 0
    clase_campero = 0
    estrategia = "A"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/predecir-riesgo" -Method Post -ContentType "application/json" -Body $body
```

---

## Resultados y Métricas Empíricas

A continuación se resumen los resultados obtenidos en las diferentes etapas de experimentación del proyecto:

### Comparativa Modelo Completo vs. Lite

Evaluación en la muestra geoespacial de siniestros viales (`outputs/geo_evaluacion/resultados_comparativos_lite.csv`):

| Modelo | Variables | Accuracy | Precision Macro | Recall Macro | Recall Fatal | F1-Score Macro | AUC Fatal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Completo** | 49 | 77.76% | 0.6070 | 0.7810 | **87.50%** | 0.5804 | **0.9441** |
| **XGBoost Lite Balanceado** | 8 | **80.32%** | **0.6095** | 0.7797 | **79.17%** | **0.6012** | **0.9323** |

> **Hallazgo Clave:** El modelo Lite (8 variables) retiene el **78.02% del peso acumulado** de importancia del modelo completo, alcanzando un $AUC_{\text{fatal}}$ de $0.9323$ y superando al modelo completo en precisión general ($80.32\%$), lo que valida la viabilidad del formulario web simplificado.

### Rendimiento Multiclase Balanceado

Resultados del submuestreo simétrico 3x2.951 casos (8.853 registros totales, 33.3% por cada clase de gravedad):

| Modelo | F1-Score CV | Accuracy | Precision Macro | Recall Macro | Recall Solo Daños | Recall Con Heridos | Recall Con Muertos | F1-Score Test | AUC Fatal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7319** | **0.7225** | **0.7285** | **0.7225** | 87.91% | **62.71%** | **66.14%** | **0.7252** | **0.8459** |
| **XGBoost** | 0.7064 | 0.7142 | 0.7149 | 0.7143 | **90.17%** | 61.24% | 62.87% | 0.7145 | 0.8385 |
| **Regresión Logística (Baseline)**| 0.7062 | 0.6980 | 0.6995 | 0.6981 | 87.68% | 56.05% | 65.69% | 0.6987 | 0.8394 |

### Rendimiento Clasificación Binaria

Resultados del experimento binario balanceado (50% No Fatales vs. 50% Fatales):

| Modelo | F1-Score CV | Accuracy General | Precision | Recall | F1-Score Test | AUC General |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8196** | **80.91%** | **78.22%** | **85.65%** | **0.8177** | **0.8882** |
| **XGBoost** | 0.7874 | 78.94% | 77.12% | 82.26% | 0.7961 | 0.8785 |

---

## Entregables y Artefactos del Proyecto

1. **Datos Procesados y Limpios:**
   * Archivo Excel consolidado [`registro_accidentes_limpio.xlsx`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/outputs/registro_accidentes_limpio.xlsx).
   * Vista minable enriquecida con calendario oficial colombiano (`vista_minable_accidente.tsv`).
   * Matriz One-Hot Encoding lista para Machine Learning (`dataset_listo_para_ml.tsv`).

2. **Modelos Predictivos Serializados (`outputs/modelos_optimizados/`):**
   * `multiclase_XGBoost_Lite.joblib`: Modelo ligero balanceado para el microservicio web.
   * `multiclase_XGBoost.joblib`: Modelo clasificador completo de 49 variables.
   * `multiclase_Random_Forest.joblib`: Modelo Random Forest multiclase.
   * Modelos binarios (`binario_XGBoost.joblib`, `binario_Random_Forest.joblib`).

3. **Artefactos Geoespaciales y de Inferencia (`outputs/geo_evaluacion/`):**
   * [`2500_geocodificados_con_predicciones.geojson`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/outputs/geo_evaluacion/2500_geocodificados_con_predicciones.geojson): Archivo GeoJSON enriquecido con predicciones y probabilidades fatales para software SIG (QGIS, ArcGIS, Mapbox).
   * [`top_8_atributos.json`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/outputs/geo_evaluacion/top_8_atributos.json): Configuración de las 8 variables más influyentes.
   * [`plantilla_imputacion_49.json`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/outputs/geo_evaluacion/plantilla_imputacion_49.json): Vector de modas y medianas para la Estrategia B de inferencia.

4. **Microservicio REST Operativo:**
   * Código de la API en [`src/api_servicio/app_fastapi.py`](file:///c:/JAVERIANA_LOCAL/26-30/Proyecto%20de%20grado%20Sistemas/Data%20Set/codigo%20data%20set/src/api_servicio/app_fastapi.py) con soporte CORS, esquema Pydantic validado, documentación interactiva Swagger UI y lógica de niveles de alerta operativa.
