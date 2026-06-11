# Gestión de Datos y Videos

Este documento explica cómo obtener, organizar y gestionar los datos necesarios para ejecutar el pipeline de NEUROLAP Visión Artificial.

---

## ⚠️ Datos No Versionados

Los siguientes directorios contienen datos pesados que **no se incluyen en el repositorio Git**:

| Directorio | Contenido | Tamaño aprox. |
|---|---|---|
| `data/` | Videos originales y procesados | ~29 GB |
| `artifacts/` | Modelos entrenados, CSVs de features y predicciones | ~200 MB |
| `datasets/` | Datasets de keypoints extraídos | Variable |
| `reports/` | Reportes visuales y diagnósticos | Variable |

Estos datos deben obtenerse por separado (ver sección de obtención más abajo).

---

## Estructura de Datos Esperada

```
NEUROLAP_Vision_Artificial/
│
├── data/
│   ├── raw/                          # Videos originales sin procesar
│   │   ├── rojo/                     # Videos clasificados como rojo
│   │   │   ├── IMG_8222.MOV
│   │   │   └── ...
│   │   ├── amarillo/                 # Videos clasificados como amarillo
│   │   │   ├── IMG_8308.MOV
│   │   │   └── ...
│   │   └── verde/                    # Videos clasificados como verde
│   │       ├── IMG_8230.MOV
│   │       └── ...
│   │
│   ├── processed/                    # Videos normalizados (15 FPS, resolución estándar)
│   │   ├── rojo/
│   │   ├── amarillo/
│   │   └── verde/
│   │
│   ├── augmented/                    # Videos con augmentación aplicada
│   │   └── ...
│   │
│   └── labels/                       # Videos etiquetados para entrenamiento
│       ├── rojo/
│       ├── amarillo/
│       └── verde/
│
├── artifacts/                        # Modelos y resultados generados
│   ├── v2/                           # Artifacts del pipeline V2
│   │   ├── features_enriched.csv
│   │   ├── video_quality_v2.joblib
│   │   ├── keypoints/               # Keypoints cacheados
│   │   └── ...
│   │
│   └── v3/                           # Artifacts del pipeline V3
│       ├── edm_features.csv
│       ├── edm_annotations_working.csv
│       ├── review_sheets/            # Frames clave por video
│       └── ...
│
├── datasets/                         # Datasets procesados
│   └── kp_v1/                       # Keypoints versión 1
│       └── _manifest.csv
│
└── reports/                          # Reportes de diagnóstico
    ├── hand_tracking/
    └── *.png
```

---

## Cómo Obtener los Datos

### Opción 1: Carpeta compartida del equipo

Solicita acceso a la carpeta compartida del equipo NEUROLAP. Los videos originales se encuentran organizados por color (rojo, amarillo, verde) en la carpeta de Google Drive del proyecto.

1. Descarga la carpeta completa de videos
2. Coloca los videos en `data/raw/{rojo,amarillo,verde}/`
3. Ejecuta el pipeline de preprocesamiento

### Opción 2: Regenerar artifacts desde código

Si tienes los videos originales, puedes regenerar todos los artifacts ejecutando el pipeline completo:

```powershell
# 1. Preprocesar videos
.\.venv\Scripts\python.exe -m src.preprocessing.video_preprocess_batch

# 2. Construir índice
.\.venv\Scripts\python.exe -m src.pipeline.build_index --expect-count 343

# 3. Extraer features
.\.venv\Scripts\python.exe -m src.pipeline.extract_features --model-complexity 1

# 4. Enriquecer features
.\.venv\Scripts\python.exe -m src.pipeline.enrich_temporal_features

# 5. Entrenar modelo
.\.venv\Scripts\python.exe -m src.pipeline.train_quality_v2
```

---

## Convenciones de Nomenclatura

### Videos originales

Los videos siguen el formato de nombre de la cámara:
- **iPhone**: `IMG_XXXX.MOV` (ej: `IMG_8230.MOV`)
- **Android**: `VID_YYYYMMDD_HHMMSS.mp4` (ej: `VID_20250820_104800.mp4`)

### Videos procesados

Los videos procesados añaden el sufijo `_15fps`:
- `IMG_8230_15fps.mp4`
- `VID_20250820_104800_15fps.mp4`

### Videos augmentados

Los videos augmentados añaden el sufijo de la transformación:
- `_aug_alejado` — Zoom out
- `_aug_est_hori` — Estiramiento horizontal
- `_aug_est_vert` — Estiramiento vertical
- `_aug_mas_brillo` — Más brillo
- `_aug_menos_brillo` — Menos brillo
- `_aug_mas_cont` — Más contraste
- `_aug_menos_cont` — Menos contraste
- `_aug_shift_abajo/arriba/der/izq` — Desplazamiento

---

## Dataset Actual

| Color | Videos originales | Con augmentación |
|---|---|---|
| 🟢 Verde | ~150 | ~1,800 |
| 🟡 Amarillo | ~78 | ~936 |
| 🔴 Rojo | ~115 | ~1,380 |
| **Total** | **~343** | **~4,116** |

---

## Notas Importantes

1. **No subir videos a Git**: Los videos pesan ~29 GB. El `.gitignore` ya los excluye. Si necesitas versionarlos, considera [DVC](https://dvc.org/) o [Git LFS](https://git-lfs.github.com/).

2. **Los artifacts son reproducibles**: Todos los modelos (`.joblib`) y features (`.csv`) se pueden regenerar ejecutando el pipeline con los videos originales.

3. **Sensibilidad de datos**: Los videos contienen grabaciones de pacientes reales. Manéjalos con las precauciones de privacidad apropiadas según la normativa vigente.

4. **Espacio en disco**: Necesitarás al menos 35 GB libres para el dataset completo más los artifacts generados.
