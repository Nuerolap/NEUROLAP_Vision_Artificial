# Arquitectura del Sistema

## Visión General

NEUROLAP Visión Artificial sigue una arquitectura de pipeline secuencial con dos versiones del clasificador: V2 (clasificación tricolor tabular) y V3 (scoring EDM con features por repetición).

---

## Diagrama de Arquitectura

```
                           NEUROLAP Vision Artificial
                           ═════════════════════════

  ┌────────────────────────────────────────────────────────────────────────────┐
  │                         1. INGESTA DE DATOS                               │
  │                                                                            │
  │  Videos RAW (.mp4, .MOV)  ──▶  Organizados por color en:                  │
  │                                 data/raw/{rojo,amarillo,verde}/           │
  └────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                      2. PREPROCESAMIENTO                                   │
  │                      (src/preprocessing/)                                  │
  │                                                                            │
  │  video_preprocess.py ──▶ Normalización: resolución, 15 FPS, recorte       │
  │  frame_split.py      ──▶ Extracción de frames individuales                │
  │  media_pipe.py       ──▶ Keypoints de manos (21 landmarks × 2 manos)      │
  │  data_augmentation.py──▶ Variantes: brillo, contraste, shift, zoom        │
  └────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                    3. EXTRACCIÓN DE FEATURES                               │
  │                    (src/pipeline/)                                          │
  │                                                                            │
  │  build_index.py        ──▶ Índice maestro de videos con metadatos          │
  │  extract_features.py   ──▶ Features espaciales por frame y por video       │
  │  enrich_temporal_*     ──▶ Features temporales, calidad de captura         │
  │  edm_features_v3.py   ──▶ Features EDM por repetición (V3)                │
  │                                                                            │
  │  Features generados:                                                       │
  │  ├── Velocidad y aceleración de keypoints                                  │
  │  ├── Dispersión espacial de la mano                                        │
  │  ├── Simetría entre manos                                                  │
  │  ├── Tiempo de ejecución efectivo                                          │
  │  ├── Calidad de detección (% frames con mano visible)                      │
  │  └── Métricas de framing (manos cerca del borde/fuera de frame)            │
  └────────────────────┬───────────────────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
  ┌─────────────────┐   ┌──────────────────┐
  │   PIPELINE V2   │   │   PIPELINE V3    │
  │  (Clasificación │   │  (Scoring EDM)   │
  │   tricolor)     │   │                  │
  └────────┬────────┘   └────────┬─────────┘
           │                     │
           ▼                     ▼
  ┌─────────────────┐   ┌──────────────────┐
  │  Modelos:       │   │  Modelos:        │
  │  • SVC          │   │  • Score 0-6     │
  │  • Random Forest│   │  • Color scorer  │
  │  • HistGradBoost│   │  • Two-stage     │
  │  • LightGBM     │   │                  │
  │  • XGBoost      │   │  Anotación:      │
  │                 │   │  • App web local  │
  │  Estrategias:   │   │  • Rúbrica EDM   │
  │  • Flat         │   │  • Review sheets │
  │  • Two-stage    │   │                  │
  │  • Yellow-cap   │   │                  │
  └────────┬────────┘   └────────┬─────────┘
           │                     │
           └──────────┬──────────┘
                      ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                       5. SALIDA                                            │
  │                                                                            │
  │  • Predicción de color por video (rojo / amarillo / verde)                 │
  │  • Score EDM 0-6 (V3)                                                      │
  │  • Reportes de calidad de captura                                          │
  │  • Métricas de evaluación (accuracy, F1, recall por clase)                 │
  │  • Modelos serializados (.joblib) para inferencia                          │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

## Módulos del Sistema

### `src/preprocessing/`

Responsable de transformar los videos crudos a un formato normalizado y extraer los keypoints de manos usando MediaPipe.

| Script | Entrada | Salida |
|---|---|---|
| `video_preprocess.py` | Video RAW | Video normalizado (15 FPS, resolución estándar) |
| `video_preprocess_batch.py` | Directorio de videos | Directorio de videos normalizados |
| `frame_split.py` | Video normalizado | Frames individuales (PNG) |
| `media_pipe.py` | Frames o video | Keypoints JSON (21 landmarks × 2 manos) |
| `analyze_frames.py` | Frames | Reporte de calidad de frames |
| `data_augmentation.py` | Video normalizado | 11 variantes augmentadas por video |

### `src/model/`

Contiene los scripts del modelo original (V1) para entrenamiento con embeddings de ventana temporal y autoencoders.

### `src/pipeline/`

Pipeline V2/V3 completo con extracción de features, búsqueda de hiperparámetros, entrenamiento de modelos, y la aplicación de anotación EDM.

### `src/processing/`

Utilidades auxiliares para extracción de puntos, creación de sets de entrenamiento y gestión de manifiestos.

---

## Estrategias de Validación

El pipeline usa **validación cruzada agrupada** para evitar data leakage:

- Los videos se agrupan por sesión/paciente
- Nunca aparecen videos del mismo grupo en train y test simultáneamente
- Se reportan métricas agrupadas (grouped accuracy, balanced accuracy, macro-F1)

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Detección de manos | MediaPipe Hands (21 landmarks 3D) |
| Procesamiento de video | OpenCV |
| Modelos tabulares | scikit-learn, LightGBM, XGBoost |
| Serialización | joblib |
| Features | NumPy, Pandas |
| Visualización | Matplotlib |
| App de anotación | Servidor HTTP local con JavaScript |
