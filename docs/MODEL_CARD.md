# Model Card — Predicción de Abandono Estudiantil

Ficha técnica del modelo servido por la API: qué hace, con qué datos se entrenó, qué tan bien funciona y cuáles son sus límites.

## Resumen

| | |
|---|---|
| **Tarea** | Clasificación binaria: `Dropout` (abandona) vs `Graduate` (se gradúa) |
| **Modelo** | Red neuronal densa (Keras): 28 → 64 → 32 → 16 → 1, ReLU + Sigmoid, Dropout 0.3 / 0.2 |
| **Formato en producción** | ONNX (`modelos/modelo_dropout.onnx`), ejecutado con onnxruntime |
| **Momento de uso** | Al terminar el **1er semestre** del estudiante |
| **Salida** | Probabilidad de abandono (0–1), clase con umbral 0.5 y nivel de riesgo |
| **Autora** | Niurka Vanesa Yupanqui |

## Datos

- **Fuente:** *Higher Education Predictors of Student Retention*, derivado del dataset de UCI *Predict Students' Dropout and Academic Success* (estudiantes de una institución de educación superior de Portugal).
- **Tamaño:** 4.424 estudiantes, 35 columnas.
- **Filtrado:** se excluyeron los 794 estudiantes `Enrolled` (todavía cursando, sin resultado final). Quedan **3.630**: 2.209 Graduate (61%) y 1.421 Dropout (39%).
- **Split:** 80% train (2.904) / 20% test (726), estratificado. Features escaladas con `StandardScaler` ajustado solo en train.
- **Desbalance:** compensado con `class_weight='balanced'` durante el entrenamiento.

### Prevención de data leakage

Se eliminaron las **6 variables del 2do semestre** (`Curricular units 2nd sem ...`). Eran las más correlacionadas con el abandono, pero se conocen al mismo tiempo o después de que el estudiante abandona: usarlas inflaría las métricas y el modelo no serviría para **anticipar** el abandono.

### Features (28)

| Grupo | Variables |
|---|---|
| Demográficas | Marital status, Nacionality, Gender, Age at enrollment, International, Displaced, Educational special needs |
| Socioeconómicas | Mother's/Father's qualification, Mother's/Father's occupation, Debtor, Tuition fees up to date, Scholarship holder |
| Ingreso | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification |
| Académicas (1er semestre) | Curricular units 1st sem: credited, enrolled, evaluations, approved, grade, without evaluations |
| Macroeconómicas | Unemployment rate, Inflation rate, GDP |

> ⚠️ Las variables categóricas vienen **codificadas como enteros** en el dataset (ej: `Course` va de 1 a 17, `Application mode` de 1 a 18). La API espera **esa misma codificación**. Un valor fuera del rango visto en entrenamiento (por ejemplo `Course: 9500`, el código del dataset original de UCI) produce predicciones sin sentido. Los rangos válidos están en la tabla del [Anexo](#anexo-rangos-de-las-variables).

## Rendimiento (test set, 726 estudiantes)

| Métrica | Valor |
|---|---|
| **AUC-ROC** | **0.946** |
| Accuracy | 0.89 |
| Recall Dropout | 0.88 |
| Precision Dropout | 0.84 |
| F1 Dropout | 0.86 |

**Matriz de confusión** (umbral 0.5):

| | Predicho Graduate | Predicho Dropout |
|---|---|---|
| **Real Graduate** | 393 | 49 |
| **Real Dropout** | 34 | 250 |

De cada 100 estudiantes que realmente abandonan, el modelo detecta ~88. De cada 100 que marca como `Dropout`, ~84 efectivamente abandonan.

**Equivalencia Keras ↔ ONNX:** diferencia máxima de 1.2e-7 en la probabilidad, verificada por `src/convertir_a_onnx.py` y por el test `test_modelo_onnx_equivale_a_keras` en cada CI.

## Uso previsto

- ✅ **Alerta temprana** para que tutores o bienestar estudiantil ofrezcan apoyo (tutorías, orientación, ayuda económica) a estudiantes en riesgo.
- ✅ Análisis agregado de factores de riesgo en una cohorte.
- ❌ **No** usar para decisiones automáticas que perjudiquen al estudiante: admisión, asignación o retiro de becas, sanciones.
- ❌ **No** usar como única fuente: siempre con revisión humana.

## Limitaciones y riesgos

- **Una sola institución y un solo país.** Aplicarlo a otra universidad o a otro sistema educativo requiere reentrenar o, al menos, revalidar con datos locales.
- **No considera a quienes siguen cursando.** Se entrenó solo con estudiantes con resultado final (Graduate/Dropout).
- **Variables sensibles.** Incluye género, edad, nacionalidad, estado civil y situación económica (Debtor, Scholarship holder). No se evaluó si el error del modelo es distinto entre grupos (*fairness*). Antes de un uso real, conviene medir las métricas por subgrupo.
- **Categorías como números.** Las variables categóricas se usan como enteros ordinales, lo que asume un orden que no existe (ej: `Course` 3 no está "entre" 2 y 4). Funciona bien en test, pero con one-hot encoding podría mejorar.
- **Umbrales fijos.** El umbral de decisión (0.5) y los cortes de riesgo (0.4 / 0.7) no se ajustaron a un costo concreto de falsos positivos o negativos.
- **Sin validación de rangos en la API.** La API valida que estén las 28 features, pero no que cada valor esté dentro del rango de entrenamiento.
- **Deriva de datos.** Las variables macroeconómicas y el perfil de los estudiantes cambian con los años. El modelo debería revalidarse periódicamente.

## Reentrenamiento

1. Reentrenar en `notebooks/analisis.ipynb` (guarda `modelo_dropout.keras`, `scaler.pkl` y `columnas_features.pkl` en `modelos/`).
2. Convertir a ONNX: `python src/convertir_a_onnx.py` (verifica la equivalencia automáticamente).
3. Correr los tests (`pytest -v`) y hacer push a `main`. El CI valida todo y Render despliega.
4. Actualizar las métricas de esta model card.

## Anexo: rangos de las variables

Rangos observados en los datos de entrenamiento (sin `Enrolled`):

| Variable | Mín | Máx |
|---|---|---|
| Marital status | 1 | 6 |
| Application mode | 1 | 18 |
| Application order | 0 | 6 |
| Course | 1 | 17 |
| Daytime/evening attendance | 0 | 1 |
| Previous qualification | 1 | 17 |
| Nacionality | 1 | 21 |
| Mother's qualification | 1 | 29 |
| Father's qualification | 1 | 34 |
| Mother's occupation | 1 | 32 |
| Father's occupation | 1 | 46 |
| Displaced, Educational special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, International | 0 | 1 |
| Age at enrollment | 17 | 70 |
| Curricular units 1st sem (credited) | 0 | 20 |
| Curricular units 1st sem (enrolled) | 0 | 26 |
| Curricular units 1st sem (evaluations) | 0 | 45 |
| Curricular units 1st sem (approved) | 0 | 26 |
| Curricular units 1st sem (grade) | 0 | 18.875 |
| Curricular units 1st sem (without evaluations) | 0 | 12 |
| Unemployment rate | 7.6 | 16.2 |
| Inflation rate | -0.8 | 3.7 |
| GDP | -4.06 | 3.51 |
