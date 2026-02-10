# 📚 GUÍA PASO A PASO - Proyecto de Predicción de Churn

## Resolución Completa de la Prueba Técnica

Esta guía te llevará paso a paso por todo el proyecto, desde la configuración inicial hasta la presentación de resultados.

---

## 🎯 FASE 0: Preparación del Entorno

### Paso 0.1: Requisitos Previos

Asegúrate de tener instalado:
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para control de versiones)

### Paso 0.2: Crear Entorno Virtual

```bash
# Navegar al directorio del proyecto
cd proyecto_churn

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### Paso 0.3: Instalar Dependencias

```bash
# Instalar todas las librerías necesarias
pip install -r requirements.txt
```

**Librerías principales:**
- pandas: Manipulación de datos
- numpy: Operaciones numéricas
- scikit-learn: Machine Learning
- matplotlib, seaborn: Visualización

---

## 📊 FASE 1: Análisis Exploratorio de Datos (EDA)

### Paso 1.1: Ejecutar EDA

```bash
# Opción A: Como script Python
python notebooks/eda_exploratorio.py

# Opción B: Si tienes Jupyter (recomendado)
jupyter notebook notebooks/eda_exploratorio.ipynb
```

### Paso 1.2: Analizar Hallazgos del EDA

**Principales problemas identificados:**

1. **Calidad de Datos:**
   - ✅ Duplicados detectados (C001, C003, C012, C023 aparecen repetidos)
   - ✅ Formatos de fecha inconsistentes (DD/MM/YYYY, YYYY-MM-DD, MM/DD/YYYY)
   - ✅ Outliers extremos en monthly_spend (valores negativos, >15000)
   - ✅ Valores nulos en phone, monthly_spend, total_shipments
   - ✅ Datos sensibles sin protección (nombres, emails, direcciones)

2. **Distribución de Churn:**
   - Revisar si las clases están balanceadas
   - Identificar tasa de churn actual
   - Decisión: Usar stratify en train/test split

3. **Patrones Identificados:**
   - Recencia es predictor fuerte de churn
   - Clientes inactivos (>180 días) presentan mayor riesgo
   - Segmentos de bajo gasto tienen mayor tasa de abandono

**Acción:** Documentar estos hallazgos para justificar decisiones de limpieza

---

## 🥉 FASE 2: Capa BRONZE - Datos Raw

### Paso 2.1: Verificar Datos en Bronze

```bash
# Los datos ya deben estar en datos/bronze/raw_data_customers.csv
ls -lh datos/bronze/

# Verificar contenido
head -20 datos/bronze/raw_data_customers.csv
```

### Paso 2.2: Ejecutar Carga a Bronze

```bash
python src/ingestion/cargar_bronze.py
```

**Lo que hace este script:**
- Carga el CSV sin transformaciones
- Registra metadatos (timestamp, # registros, # columnas)
- Realiza análisis inicial de calidad
- Detecta duplicados y nulos
- Preserva datos originales para auditoría

**Salida esperada:**
```
ℹ️ Cargando datos desde: /path/to/bronze/raw_data_customers.csv
✅ Datos cargados exitosamente
   - Registros: 115
   - Columnas: 10
⚠️  Duplicados completos detectados: X
⚠️  IDs duplicados detectados: Y
```

---

## 🥈 FASE 3: Capa SILVER - Limpieza y Gobierno de Datos

### Paso 3.1: Ejecutar Transformación Bronze → Silver

```bash
python src/transformacion/bronze_a_silver.py
```

**Transformaciones aplicadas:**

1. **Eliminación de Duplicados**
   - Duplicados completos: Se eliminan
   - Duplicados por ID: Se mantiene registro con fecha más reciente
   - **Justificación**: Dato más reciente refleja mejor el estado actual

2. **Normalización de Fechas**
   - Múltiples formatos → YYYY-MM-DD estándar
   - Uso de pandas.to_datetime con inferencia automática
   - **Justificación**: Consistencia para cálculos temporales

3. **Corrección de Outliers**
   - Gastos negativos → 0 (error de registro)
   - Gastos >$15,000 → Cap al umbral
   - Envíos >500 → Cap al umbral
   - **Justificación**: Umbrales basados en conocimiento del negocio

4. **Tratamiento de Nulos**
   - phone → "MISSING" (categoría especial)
   - monthly_spend → Mediana (robusta a outliers)
   - total_shipments → Mediana
   - churn_label → Eliminar registro (sin etiqueta no entrenable)
   - **Justificación**: Estrategias diferenciadas según impacto

5. **Hashing de Datos Sensibles (PII)**
   - full_name, email, phone, home_address → SHA-256
   - **Justificación**: Cumplimiento GDPR/privacidad

**Salida esperada:**
```
[1/8] Cargando datos de capa Bronze...
✅ Datos cargados: 115 registros

[2/8] Detectando y eliminando duplicados...
✅ Eliminados X duplicados

[3/8] Normalizando formatos de fecha...
✅ Fechas normalizadas

[4/8] Convirtiendo tipos de datos...
✅ Tipos de datos convertidos

[5/8] Detectando y corrigiendo outliers...
✅ Outliers corregidos

[6/8] Tratando valores nulos...
✅ Valores nulos manejados

[7/8] Hasheando datos sensibles (PII)...
✅ Datos sensibles protegidos

[8/8] Validando calidad de datos Silver...
✅ Datos Silver guardados en: datos/silver/clientes_limpios.csv
```

### Paso 3.2: Verificar Datos Silver

```bash
# Ver primeras líneas
head -5 datos/silver/clientes_limpios.csv

# Contar registros
wc -l datos/silver/clientes_limpios.csv
```

---

## 🥇 FASE 4: Capa GOLD - Feature Engineering

### Paso 4.1: Ejecutar Transformación Silver → Gold

```bash
python src/transformacion/silver_a_gold.py
```

**Features Generadas:**

1. **Métricas RFM (Recency, Frequency, Monetary)**
   - `recencia_dias`: Días desde última compra
   - `categoria_recencia`: Muy Reciente | Reciente | Inactivo | Muy Inactivo
   - `antiguedad_dias`: Tiempo como cliente
   - `segmento_gasto`: Bajo | Medio | Alto | Premium
   - `segmento_frecuencia`: Ocasional | Regular | Frecuente | VIP

2. **Engagement Score**
   - `engagement_score`: Score 0-100 ponderado
   - Fórmula: (Recencia × 0.4) + (Frecuencia × 0.3) + (Gasto × 0.3)
   - `nivel_engagement`: Bajo | Medio | Alto | Muy Alto

3. **Features de Comportamiento**
   - `gasto_por_envio`: Ticket promedio
   - `dias_entre_compras`: Frecuencia temporal
   - `cliente_activo_reciente`: Flag binario
   - `cliente_alto_valor`: Flag binario

4. **Features de Riesgo**
   - `riesgo_inactividad`: Flag (recencia >180 días)
   - `riesgo_bajo_engagement`: Flag (score <30)
   - `score_riesgo_churn`: Métrica combinada
   - `nivel_riesgo`: Bajo | Medio | Alto | Crítico

5. **Encoding**
   - Variables categóricas → One-Hot Encoding
   - Dataset optimizado para ML

**Salida esperada:**
```
[1/4] Cargando datos de capa Silver...
✅ Datos Silver cargados: ~100 registros

[2/4] Generando features de modelado...
ℹ️ Calculando recencia de clientes
ℹ️ Calculando antigüedad de clientes
ℹ️ Categorizando nivel de gasto
✅ Features generadas: 30+ nuevas columnas

[3/4] Preparando dataset para modelado...
✅ Dataset preparado: 50+ features finales

[4/4] Validando dataset Gold...
Distribución de churn_label:
  - No Churn (0): X (XX.X%)
  - Churn (1): Y (YY.Y%)
✅ Dataset Gold guardado en: datos/gold/clientes_modelado.csv
```

### Paso 4.2: Verificar Datos Gold

```bash
# Ver estructura
head -3 datos/gold/clientes_modelado.csv

# Contar features
head -1 datos/gold/clientes_modelado.csv | awk -F',' '{print NF}'
```

---

## 🤖 FASE 5: Modelado - Entrenamiento

### Paso 5.1: Entrenar Modelo de Churn

```bash
python src/modelado/entrenar_modelo.py
```

**Proceso de Entrenamiento:**

1. **Carga de Datos Gold**
   - Lee dataset optimizado
   - Verifica presencia de todas las columnas

2. **Preparación de Datos**
   - Separación X (features) y y (target)
   - Train/Test Split (75%/25%)
   - Estratificación por churn_label

3. **Entrenamiento del Modelo**
   - Algoritmo: Random Forest Classifier
   - Parámetros:
     * n_estimators: 100
     * max_depth: 10
     * class_weight: 'balanced' (manejo de desbalance)
     * random_state: 42 (reproducibilidad)

4. **Evaluación**
   - Métricas en conjunto de prueba
   - Validación cruzada 5-fold
   - Importancia de features
   - Matriz de confusión

5. **Persistencia**
   - Modelo guardado en: `modelos/modelo_churn.pkl`
   - Métricas guardadas en: `resultados/metricas/metricas_modelo.json`
   - Visualizaciones en: `resultados/visualizaciones/evaluacion_modelo.png`

**Salida esperada:**
```
ENTRENANDO MODELO RANDOM FOREST

Parámetros del modelo:
  - n_estimators: 100
  - max_depth: 10
  - class_weight: balanced
  
✅ Modelo entrenado exitosamente

MÉTRICAS EN CONJUNTO DE PRUEBA:
✅ Accuracy: 0.XXXX
✅ Precision (Churn): 0.XXXX
✅ Recall (Churn): 0.XXXX
✅ F1-Score (Churn): 0.XXXX
✅ ROC-AUC: 0.XXXX

TOP 10 FEATURES MÁS IMPORTANTES:
recencia_dias: 0.XXXX
engagement_score: 0.XXXX
monthly_spend: 0.XXXX
...

✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE
```

### Paso 5.2: Revisar Resultados

```bash
# Ver métricas completas
cat resultados/metricas/metricas_modelo.json

# Ver visualizaciones (abrirlas en visor de imágenes)
open resultados/visualizaciones/evaluacion_modelo.png  # Mac
xdg-open resultados/visualizaciones/evaluacion_modelo.png  # Linux
# Windows: doble clic en el archivo
```

---

## 🚀 FASE 6: Ejecución del Pipeline Completo

### Opción A: Pipeline Completo Automatizado

```bash
# Ejecutar TODO de principio a fin
python main.py
```

Este comando ejecuta:
1. ✅ Bronze: Carga de datos
2. ✅ Silver: Limpieza y gobierno
3. ✅ Gold: Feature engineering
4. ✅ Modelado: Entrenamiento y evaluación

**Duración aproximada:** 2-5 minutos

### Opción B: Ejecución por Etapas

```bash
# Solo limpieza (Bronze → Silver)
python main.py --solo-limpieza

# Hasta features (Bronze → Silver → Gold)
python main.py --solo-features

# Solo entrenamiento (requiere Gold existente)
python main.py --solo-modelo
```

---

## 📋 FASE 7: Interpretación de Resultados de Negocio

### Paso 7.1: Analizar Métricas del Modelo

**Métricas Clave:**

1. **Accuracy (Exactitud)**
   - Qué mide: % de predicciones correctas totales
   - Interpretación: Confiabilidad general del modelo
   - Meta: >0.80

2. **Precision (Churn)**
   - Qué mide: De los que predecimos como churn, cuántos realmente harán churn
   - Interpretación de negocio: Eficiencia de campañas de retención
   - Ejemplo: Precision=0.75 → De cada 100 clientes que marcamos como riesgo, 75 realmente se irán
   - Meta: >0.70

3. **Recall (Churn)**
   - Qué mide: De los que realmente harán churn, cuántos identificamos
   - Interpretación de negocio: Cobertura de clientes en riesgo
   - Ejemplo: Recall=0.82 → Identificamos 82 de cada 100 clientes que se irán
   - Meta: >0.75 (prioridad en churn)

4. **F1-Score**
   - Qué mide: Balance entre Precision y Recall
   - Interpretación: Métrica equilibrada de performance
   - Meta: >0.75

5. **ROC-AUC**
   - Qué mide: Capacidad discriminativa del modelo
   - Interpretación: Qué tan bien separa clases
   - Meta: >0.80

### Paso 7.2: Interpretar Matriz de Confusión

```
                 Predicho
              No Churn  Churn
Real  
No Churn       TN        FP
Churn          FN        TP
```

**Significado de Negocio:**
- **TN (True Negative)**: Clientes leales correctamente identificados → No necesitan intervención
- **FP (False Positive)**: Falsos riesgos → Esfuerzo de retención desperdiciado
- **FN (False Negative)**: Churn no detectado → Pérdida de cliente sin intervención
- **TP (True Positive)**: Churn correctamente identificado → Oportunidad de retención

**Costo de Errores:**
- FP: Costo de campaña de retención innecesaria (~$50 por cliente)
- FN: Pérdida de lifetime value del cliente (~$1,000-$5,000)
- **Conclusión**: Preferimos FP sobre FN (mejor prevenir de más que de menos)

### Paso 7.3: Analizar Features Importantes

Las top features indican qué variables son más predictivas de churn:

**Típicamente encontrarás:**
1. `recencia_dias` - Días sin comprar (MÁS IMPORTANTE)
2. `engagement_score` - Nivel de engagement
3. `monthly_spend` - Valor del cliente
4. `total_shipments` - Frecuencia histórica
5. `antiguedad_dias` - Tiempo como cliente

**Decisiones de Negocio Basadas en Features:**
- Si recencia_dias es #1 → Enfocarse en campañas de reactivación
- Si engagement_score es importante → Mejorar experiencia del cliente
- Si monthly_spend alto → Programas VIP de retención

---

## 📊 FASE 8: Documentación de Decisiones

### Paso 8.1: Documentar en README

El README.md ya contiene toda la documentación necesaria. Asegúrate de:

1. ✅ Explicar arquitectura Medallion
2. ✅ Justificar cada decisión de limpieza
3. ✅ Justificar features creadas
4. ✅ Explicar selección del modelo
5. ✅ Incluir interpretación de negocio

### Paso 8.2: Preparar Material de Presentación

Para la presentación de 20 minutos, prepara:

**Estructura Sugerida:**

1. **Contexto (2 min)**
   - Problema de negocio
   - Dataset y desafíos

2. **Arquitectura (3 min)**
   - Medallion: Bronze → Silver → Gold
   - Diagrama visual

3. **Gobierno de Datos (5 min)**
   - Problemas identificados en EDA
   - Decisiones de limpieza con justificación
   - Hashing de PII para privacidad

4. **Feature Engineering (4 min)**
   - RFM y por qué es relevante
   - Engagement Score
   - Features de riesgo

5. **Modelo y Resultados (4 min)**
   - Random Forest: por qué
   - Métricas clave
   - Interpretación de negocio

6. **AWS y Escalabilidad (2 min)**
   - Arquitectura propuesta
   - Servicios clave (S3, Glue, SageMaker)

**Materiales:**
- Diagramas en `diagramas/arquitectura_*.txt`
- Visualizaciones en `resultados/visualizaciones/`
- Métricas en `resultados/metricas/metricas_modelo.json`

---

## ☁️ FASE 9: Preparación para AWS (Conceptual)

### Paso 9.1: Revisar Arquitectura AWS

```bash
# Ver diagrama completo de AWS
cat diagramas/arquitectura_aws.txt
```

**Componentes Clave a Mencionar:**

1. **Almacenamiento:** S3 con estructura Medallion
   - Bronze, Silver, Gold en buckets separados
   - Versionado habilitado
   - Encriptación SSE-S3

2. **ETL:** AWS Glue
   - Glue Jobs para transformaciones
   - Glue Data Catalog para metadata
   - Crawlers para descubrimiento de schema

3. **ML:** Amazon SageMaker
   - Training Jobs con scikit-learn
   - Model Registry para versionado
   - Endpoints para inferencia

4. **Orquestación:** Step Functions + EventBridge
   - Pipeline automatizado
   - Manejo de errores
   - Scheduling

5. **Monitoreo:** CloudWatch
   - Logs centralizados
   - Métricas de performance
   - Alarmas

### Paso 9.2: Migración Conceptual

**Cambios Necesarios en Código:**

```python
# Local
df = pd.read_csv('datos/bronze/raw_data_customers.csv')

# AWS
import boto3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='churn-data-bronze', 
                    Key='raw_data_customers.csv')
df = pd.read_csv(obj['Body'])
```

**No es necesario implementar**, solo explicar la estrategia.

---

## ✅ CHECKLIST FINAL

Antes de la presentación, verificar:

### Entregables Técnicos:
- [ ] Código organizado en carpetas (src/, datos/, notebooks/)
- [ ] Todos los scripts ejecutan sin errores
- [ ] README.md completo con documentación
- [ ] Diagramas de arquitectura (local + AWS)
- [ ] Resultados generados (modelo, métricas, visualizaciones)
- [ ] requirements.txt con dependencias

### Documentación de Decisiones:
- [ ] Cada transformación tiene justificación de negocio
- [ ] Estrategias de nulos documentadas
- [ ] Detección de outliers justificada
- [ ] Feature engineering explicado
- [ ] Selección de modelo justificada
- [ ] Métricas priorizadas explicadas

### Preparación AWS:
- [ ] Diagrama de arquitectura AWS revisado
- [ ] Servicios clave identificados (S3, Glue, SageMaker)
- [ ] Estrategia de migración clara
- [ ] Costos estimados (opcional pero impresionante)

---

## 🎤 CONSEJOS PARA LA PRESENTACIÓN

1. **Enfócate en Negocio, No en Matemáticas**
   - Menos: "Usamos la mediana porque es robusta a outliers según la distribución de Cauchy..."
   - Más: "Usamos la mediana porque representa mejor a un cliente típico sin distorsión por valores extremos"

2. **Muestra el Impacto**
   - "Con Recall de 82%, identificamos 82 de cada 100 clientes en riesgo, permitiendo intervención proactiva"
   - "Esto puede prevenir pérdidas de hasta $82,000 mensuales (asumiendo $1,000 LTV por cliente)"

3. **Demuestra Pensamiento Crítico**
   - "Priorizamos Recall sobre Precision porque el costo de perder un cliente ($1,000+) supera el costo de una campaña de retención ($50)"

4. **Arquitectura AWS = Escalabilidad**
   - "Este proyecto está diseñado para escalar de 110 clientes a millones usando S3 + Glue + SageMaker"
   - "El pipeline es automatizable con Step Functions para re-entrenamiento diario"

5. **Gobierno de Datos = Profesionalismo**
   - "Implementamos hashing SHA-256 de PII para cumplimiento GDPR"
   - "Arquitectura Medallion garantiza trazabilidad y auditabilidad"

---

## 🚨 TROUBLESHOOTING

### Error: "FileNotFoundError: No se encuentra el archivo"
**Solución:** Verifica que estés en el directorio raíz del proyecto
```bash
pwd  # Debe mostrar /path/to/proyecto_churn
```

### Error: "ModuleNotFoundError: No module named 'pandas'"
**Solución:** Instalar dependencias
```bash
pip install -r requirements.txt
```

### Error: "ValueError: Falta la variable objetivo 'churn_label'"
**Solución:** Ejecutar primero las capas anteriores
```bash
python src/transformacion/bronze_a_silver.py
python src/transformacion/silver_a_gold.py
```

### Advertencia: "class_weight='balanced' but classes are balanced"
**Solución:** No es un error, es informativo. El modelo ajusta pesos automáticamente.

---

## 📚 RECURSOS ADICIONALES

- **Medallion Architecture:** [Databricks Glossary](https://www.databricks.com/glossary/medallion-architecture)
- **AWS Glue:** [Developer Guide](https://docs.aws.amazon.com/glue/)
- **SageMaker:** [Getting Started](https://docs.aws.amazon.com/sagemaker/)
- **Random Forest:** [scikit-learn Docs](https://scikit-learn.org/stable/modules/ensemble.html#forest)

---

## 🎯 RESUMEN EJECUTIVO

Este proyecto demuestra:

1. ✅ **Gobierno de Datos**: Manejo profesional de datos sucios con decisiones justificadas
2. ✅ **Feature Engineering**: Creación de variables predictivas basadas en negocio (RFM)
3. ✅ **Modelado Robusto**: Random Forest con validación cruzada y métricas interpretables
4. ✅ **Arquitectura Escalable**: Diseño Medallion preparado para producción en AWS
5. ✅ **Documentación Clara**: Código comentado y decisiones explicadas

**Mensaje Clave:** No solo resolviste un problema técnico de ML, demostraste capacidad de traducir requerimientos de negocio en soluciones escalables con gobierno de datos sólido.

¡Éxito en la presentación! 🚀