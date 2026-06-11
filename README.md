<h1 align="center">
  🧠 NEUROLAP — Visión Artificial
</h1>

<p align="center">
  <strong>Análisis automatizado de calidad en pruebas cognitivas motoras mediante visión por computadora</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green?logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
</p>

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Guía Rápida](#guía-rápida)
- [Documentación](#documentación)
- [Datos y Videos](#datos-y-videos)

---

## Descripción

**NEUROLAP Visión Artificial** es un sistema de visión por computadora diseñado para analizar videos de pruebas cognitivas motoras (Escala de Desarrollo Motor — EDM). El sistema procesa grabaciones de pacientes realizando ejercicios de motricidad fina y clasifica automáticamente la calidad de ejecución en tres niveles:

| Color | Significado | Descripción |
|:---:|---|---|
| 🟢 **Verde** | Ejecución correcta | El paciente completa el ejercicio adecuadamente |
| 🟡 **Amarillo** | Ejecución parcial | Ejecución con dificultades o patrones atípicos |
| 🔴 **Rojo** | Ejecución incorrecta | El paciente no logra completar el ejercicio |

### Flujo del Sistema

1. **Preprocesamiento** — Normalización de video (resolución, FPS, recorte)
2. **Extracción de keypoints** — Detección de manos con MediaPipe
3. **Ingeniería de features** — Métricas temporales, espaciales y de calidad de captura
4. **Clasificación** — Modelos ML entrenados (SVC, Random Forest, LightGBM, etc.)
5. **Evaluación clínica** — Scoring EDM con rúbrica de 6 puntos por video

---

## Arquitectura del Sistema

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Videos     │────▶│ Preproceso   │────▶│  Extracción de  │────▶│  Features    │
│   (raw)      │     │ (15fps, crop)│     │  Keypoints (MP) │     │  Temporales  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                        │
                                                                        ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Reporte    │◀────│  Predicción  │◀────│   Modelo ML     │◀────│  Train /     │
│  (color)    │     │  por video   │     │   (clasificador)│     │  Validación  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
```

---

## Estructura del Proyecto

```
NEUROLAP_Vision_Artificial/
│
├── src/                            # Código fuente
│   ├── preprocessing/              # Preparación de videos
│   │   ├── video_preprocess.py     # Normalización individual
│   │   ├── video_preprocess_batch.py # Procesamiento por lotes
│   │   ├── frame_split.py         # Extracción de frames
│   │   ├── media_pipe.py          # Extracción de keypoints con MediaPipe
│   │   ├── analyze_frames.py      # Análisis de calidad de frames
│   │   └── data_augmentation.py   # Augmentación de datos de video
│   │
│   ├── model/                      # Entrenamiento y evaluación de modelos
│   │   ├── train_supervised_quality.py  # Entrenamiento supervisado
│   │   ├── predict_quality.py      # Predicción de calidad
│   │   ├── augment_videos.py       # Augmentación para entrenamiento
│   │   ├── diagnostico.py          # Diagnóstico de modelos
│   │   ├── embed_windows.py        # Embeddings por ventana temporal
│   │   ├── cluster_embeddings.py   # Clustering de embeddings
│   │   ├── score_reconstruction.py # Scoring por reconstrucción
│   │   └── debug_ids.py           # Utilidades de depuración
│   │
│   ├── pipeline/                   # Pipeline V2/V3 completo
│   │   ├── build_index.py         # Construcción del índice de videos
│   │   ├── extract_features.py    # Extracción de features compactos
│   │   ├── enrich_temporal_features.py # Features temporales enriquecidos
│   │   ├── train_quality_v2.py    # Entrenamiento V2
│   │   ├── accuracy_search_v2.py  # Búsqueda de accuracy óptima
│   │   ├── edm_annotation_app_v3.py  # App de anotación EDM
│   │   ├── edm_features_v3.py    # Features EDM-aware V3
│   │   ├── edm_train_v3.py       # Entrenamiento V3
│   │   └── ... (ver docs/pipeline_guide.md)
│   │
│   └── processing/                 # Utilidades de procesamiento
│       ├── extract_points.py      # Extracción de puntos clave
│       ├── create_training.py     # Creación de sets de entrenamiento
│       ├── clean_manifest.py      # Limpieza de manifiestos
│       └── frame_split.py        # Partición de frames
│
├── docs/                           # Documentación
│   ├── pipeline_guide.md          # Guía completa del pipeline
│   ├── architecture.md            # Arquitectura del sistema
│   └── data_management.md         # Gestión de datos y videos
│
├── data/                           # Datos locales (no versionados)
│   ├── raw/                       # Videos originales
│   ├── processed/                 # Videos procesados
│   └── labels/                    # Videos etiquetados (rojo/amarillo/verde)
│
├── artifacts/                      # Modelos y resultados (no versionados)
│   ├── v2/                        # Artifacts del pipeline V2
│   └── v3/                        # Artifacts del pipeline V3
│
├── requirements.txt               # Dependencias de Python
├── .gitignore                     # Archivos excluidos de Git
└── README.md                      # Este archivo
```

> **Nota:** Las carpetas `data/`, `artifacts/`, `datasets/` y `reports/` contienen datos pesados generados localmente y **no se versionan en Git**. Consulte [docs/data_management.md](docs/data_management.md) para instrucciones sobre cómo obtenerlos.

---

## Requisitos

- **Python** 3.10 o superior
- **pip** para gestión de paquetes
- **Espacio en disco:** ~30 GB para el dataset completo de videos

### Dependencias principales

| Librería | Propósito |
|---|---|
| `opencv-python` | Procesamiento de video e imágenes |
| `mediapipe` | Detección de manos y keypoints |
| `numpy` | Operaciones numéricas |
| `scikit-learn` | Modelos de clasificación |
| `pandas` | Manipulación de datos tabulares |
| `matplotlib` | Visualización |
| `tqdm` | Barras de progreso |
| `lightgbm` / `xgboost` | Modelos avanzados (opcional) |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Nuerolap/NEUROLAP_Vision_Artificial.git
cd NEUROLAP_Vision_Artificial

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

---

## Guía Rápida

### 1. Preprocesar videos

```powershell
# Normalizar un video individual (resolución, FPS)
.\.venv\Scripts\python.exe -m src.preprocessing.video_preprocess

# Procesamiento por lotes
.\.venv\Scripts\python.exe -m src.preprocessing.video_preprocess_batch
```

### 2. Extraer keypoints

```powershell
# Extracción de keypoints de manos con MediaPipe
.\.venv\Scripts\python.exe -m src.preprocessing.media_pipe
```

### 3. Ejecutar pipeline de clasificación

```powershell
# Construir índice de videos
.\.venv\Scripts\python.exe -m src.pipeline.build_index --expect-count 343

# Extraer features compactos
.\.venv\Scripts\python.exe -m src.pipeline.extract_features --model-complexity 1

# Enriquecer con features temporales
.\.venv\Scripts\python.exe -m src.pipeline.enrich_temporal_features

# Entrenar clasificador V2
.\.venv\Scripts\python.exe -m src.pipeline.train_quality_v2
```

### 4. App de anotación EDM (V3)

```powershell
# Abrir la aplicación web de anotación
.\.venv\Scripts\python.exe -m src.pipeline.edm_annotation_app_v3 --open
```

> 📖 Para la guía completa de todos los comandos del pipeline, consulta [docs/pipeline_guide.md](docs/pipeline_guide.md).

---

## Documentación

| Documento | Descripción |
|---|---|
| [Pipeline Guide](docs/pipeline_guide.md) | Guía completa de todos los comandos y artifacts del pipeline V2/V3 |
| [Architecture](docs/architecture.md) | Diagrama y descripción de la arquitectura del sistema |
| [Data Management](docs/data_management.md) | Cómo obtener, organizar y gestionar los videos y datasets |

---

## Datos y Videos

Los videos de pruebas cognitivas y los artifacts generados **no están incluidos en este repositorio** debido a su tamaño (~30 GB). Consulta [docs/data_management.md](docs/data_management.md) para:

- Estructura de directorios esperada
- Dónde obtener los videos originales
- Cómo regenerar los artifacts desde el código fuente

---

<p align="center">
  <sub>Desarrollado por el equipo <strong>NEUROLAP</strong> · Universidad de las Fuerzas Armadas ESPE</sub>
</p>
