# Guía del Pipeline de Clasificación de Calidad

Esta guía documenta todos los comandos, pasos y artifacts del pipeline de clasificación de calidad de video para pruebas cognitivas motoras (EDM).

> **Nota sobre rutas:** Todos los comandos usan la estructura reorganizada (`src.pipeline.*`). Los artifacts siguen generándose en `artifacts/v2/` y `artifacts/v3/`.

---

## Tabla de Contenidos

- [Visión General](#visión-general)
- [Pipeline V2 — Clasificación por Calidad](#pipeline-v2--clasificación-por-calidad)
- [Pipeline V3 — EDM-Aware](#pipeline-v3--edm-aware)
- [Artifacts Clave](#artifacts-clave)

---

## Visión General

El pipeline tiene dos versiones principales:

| Versión | Enfoque | Estado |
|---|---|---|
| **V2** | Clasificación tricolor (rojo/amarillo/verde) basada en features tabulares | Estable |
| **V3** | Scoring EDM con rúbrica de 6 puntos, features temporales por repetición | En desarrollo activo |

Ambas versiones comparten el índice de videos y los keypoints extraídos.

---

## Pipeline V2 — Clasificación por Calidad

### 1. Construir el índice de videos

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.build_index --expect-count 343
```

Escanea las carpetas de datos y genera `artifacts/v2/video_index.csv` con rutas, metadatos y etiquetas de color derivadas del directorio.

### 2. Probar complejidad de MediaPipe

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.probe_complexity
```

Evalúa en un subconjunto pequeño qué nivel de `model_complexity` de MediaPipe conviene usar.

### 3. Extraer features compactos

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.extract_features --model-complexity 1
```

Procesa todos los videos indexados, extrae keypoints de manos con MediaPipe y calcula features espaciales y temporales por video.

### 4. Enriquecer con features temporales

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.enrich_temporal_features
```

Añade features de calidad de captura, grupos de validación cruzada, y genera `hand_framing_report.csv` con métricas de riesgo de manos fuera de frame.

### 5. Auditoría de etiquetas

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.label_audit_v2
```

Genera `label_audit/label_conflict_audit.csv`, una plantilla de revisión `label_review_template.csv`, y contact sheets visuales bajo `label_audit/contact_sheets/`.

### 6. Aplicar revisión manual de etiquetas

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.apply_label_review_v2
```

Escribe `features_reviewed.csv` para entrenamiento y `features_reviewed_all.csv` para trazabilidad.

### 7. Features limpios (sin conflictos)

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.make_clean_features_v2
```

Crea un subconjunto conservador excluyendo grupos con conflictos de etiqueta cruzados.

### 8. Entrenar clasificador V2

```powershell
# Entrenamiento balanceado por defecto
.\.venv\Scripts\python.exe -m src.pipeline.train_quality_v2 --features artifacts/v2/features_enriched.csv --selection-tolerance 0.0
```

### 9. Búsqueda de accuracy

```powershell
# Búsqueda amplia con múltiples candidatos
.\.venv\Scripts\python.exe -m src.pipeline.accuracy_search_v2

# Búsqueda limpia eficiente
.\.venv\Scripts\python.exe -m src.pipeline.accuracy_search_v2 \
  --features artifacts/v2/features_enriched_no_label_conflicts.csv \
  --out-dir artifacts/v2/clean_search_efficient \
  --feature-sets no_source_meta motion_quality compact_quality \
  --strategies full \
  --candidates svc_select80_c5 svc_select80_c8 svc_select60_c8 svc_full_c3 \
    histgb_80_leaf15 histgb_120_leaf7 extra_trees_220_depth8 extra_trees_360_depth10 \
    rf_220_depth8 rf_360_depth10 balanced_rf_220_depth8 lgbm_120_leaf7_bal \
    lgbm_180_leaf15 xgb_120_depth2 xgb_180_depth3 \
  --top 20
```

### 10. Clasificador two-stage

```powershell
# Entrenar: primero amarillo vs no-amarillo, luego rojo vs verde
.\.venv\Scripts\python.exe -m src.pipeline.train_two_stage_v2 --min-yellow-recall 0.55

# Calibrar umbral sin reentrenar
.\.venv\Scripts\python.exe -m src.pipeline.calibrate_two_stage_v2 --min-yellow-recall 0.50 --suffix practical
```

### 11. Calibración de bias flat

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.calibrate_flat_bias_v2 --min-yellow-recall 0.50 --suffix biased
```

### 12. Benchmark de modelos

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.benchmark_models_v2
```

### 13. Features de secuencia

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.build_sequence_features_v2 --steps 32
.\.venv\Scripts\python.exe -m src.pipeline.train_sequence_v2
```

---

## Pipeline V3 — EDM-Aware

### 1. Auditoría de orden de manos

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_hand_order_v3
```

Genera `artifacts/v3/edm_hand_order_audit.csv`. La app de anotación lo carga automáticamente y activa el modo espejo para videos que probablemente inician con la mano derecha.

### 2. Cola de anotación piloto

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_annotation_queue_v3 --target-total 120
```

Crea `artifacts/v3/edm_annotation_pilot_queue.csv` con 40 videos por color, balanceando calidad de captura.

### 3. App de anotación EDM

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_annotation_app_v3 --open
```

Aplicación web local para marcar las seis repeticiones del ejercicio: `correcta`, `parcial`, `incorrecta`, `no_visible`. Auto-calcula `score_0_6`, `color_final` y `annotation_status`.

**Instrucciones de uso:**
- Usa `Marcar inicio` cuando el participante comienza la primera repetición real
- Usa `Marcar fin` solo si el segmento útil termina antes del final del video
- Si hay más de 3 repeticiones por mano, puntúa solo las 3 primeras/mejores evaluables
- Para desacuerdos con el espejo automático, usa `Espejo ON/OFF`

### 4. Cola de refuerzo amarillo

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_yellow_boost_queue_v3 --target-total 60
.\.venv\Scripts\python.exe -m src.pipeline.edm_annotation_app_v3 --queue artifacts/v3/edm_yellow_boost_queue.csv --open
```

### 5. Validar anotaciones

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_validate_annotations_v3 \
  --annotations artifacts/v3/edm_annotations_working.csv
```

### 6. Revisión visual EDM

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_review_v3
```

### 7. Features EDM temporales

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_features_v3
```

### 8. Entrenar modelo V3

```powershell
# Entrenamiento con anotaciones manuales completas
.\.venv\Scripts\python.exe -m src.pipeline.edm_train_v3

# Baseline smoke con proxy de carpeta (solo referencia, no clínico)
.\.venv\Scripts\python.exe -m src.pipeline.edm_train_v3 --allow-folder-proxy
```

### 9. Scorer de color conservador

```powershell
.\.venv\Scripts\python.exe -m src.pipeline.edm_train_color_v3 \
  --features artifacts/v3/edm_features.csv \
  --v2-features artifacts/v2/features_enriched.csv \
  --annotations artifacts/v3/edm_annotations_validated.csv \
  --allow-folder-proxy --exclude-label-conflicts \
  --out-dir artifacts/v3/color_hybrid_clean_fg05 \
  --feature-sets compact all \
  --candidates svc_select60_c6 svc_select80_c8 histgb_120_leaf11 extra_trees_260_depth9 rf_260_depth9 \
  --max-false-green 0.05 --max-review-rate 0.18
```

---

## Artifacts Clave

### V2

| Artifact | Descripción |
|---|---|
| `video_quality_v2.joblib` | Clasificador balanceado por defecto |
| `video_quality_v2_accuracy_best.joblib` | Mayor accuracy en CV agrupada |
| `video_quality_v2_yellow_limited.joblib` | Mejor modelo con cap de amarillo |
| `video_quality_v2_two_stage.joblib` | Modelo jerárquico con umbral de amarillo ajustable |
| `video_quality_v2_two_stage_practical.joblib` | Two-stage calibrado para recall práctico |
| `hand_framing_report.csv` | Reporte de calidad de captura por video |
| `features_reviewed.csv` | Features listos para entrenamiento post-revisión |
| `features_enriched_no_label_conflicts.csv` | Subset limpio sin conflictos de etiqueta |

### V3

| Artifact | Descripción |
|---|---|
| `edm_annotation_template.csv` | Hoja de revisión para scoring EDM |
| `edm_features.csv` | Features EDM-aware temporales por mano y repetición |
| `edm_annotations_validated.csv` | Anotaciones manuales validadas |
| `edm_score_v3_baseline.joblib` | Modelo baseline de score (solo clínico con anotaciones completas) |
| `color_hybrid_clean_fg05/` | Scorer de color conservador con features V3+V2 |
| `review_sheets/` | JPGs con frames clave y timeline de visibilidad de manos |

> Todos los artifacts V2 se generan en `artifacts/v2/` y los V3 en `artifacts/v3/`.
