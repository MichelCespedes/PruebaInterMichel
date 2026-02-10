# 🎯 Proyecto de Predicción de Churn - Sector Logística

**Prueba Técnica: Científico de Datos**

Solución end-to-end para predicción de churn de clientes utilizando arquitectura Medallion (Bronze → Silver → Gold) con enfoque en gobierno de datos, feature engineering y decisiones de negocio.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Configuración](#instalación-y-configuración)
- [Ejecución](#ejecución)
- [Decisiones de Negocio](#decisiones-de-negocio)
- [Resultados](#resultados)
- [Preparación para AWS](#preparación-para-aws)

---

## 📖 Descripción del Proyecto

### Objetivo

Desarrollar un modelo predictivo de churn robusto y reproducible que permita identificar clientes en riesgo de abandono, implementando buenas prácticas de gobierno de datos y preparándose para productivización en AWS.

### Contexto de Negocio

El sector logística requiere identificar proactivamente clientes en riesgo para:
- **Reducir tasa de abandono** mediante campañas de retención dirigidas
- **Optimizar recursos** enfocándose en clientes de alto valor en riesgo
- **Mejorar experiencia del cliente** anticipando necesidades

### Dataset

- **Fuente**: `raw_data_customers.csv`
- **Registros**: 110 clientes (con duplicados en datos raw)
- **Features originales**: 10 columnas
- **Variable objetivo**: `churn_label` (0 = No Churn, 1 = Churn)

---

## 🏗️ Arquitectura

### Arquitectura Medallion

El proyecto implementa la arquitectura de datos Medallion con tres capas:

```
┌─────────────────────────────────────────────────────────────┐
│                        CAPA BRONZE                          │
│              (Datos Raw - Sin Transformar)                  │
│                                                             │
│  • Ingesta tal cual del CSV                                 │
│  • Preserva estado original                                 │
│  • Trazabilidad y auditoría                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                        CAPA SILVER                          │
│            (Datos Limpios y Governados)                     │
│                                                             │
│  • Duplicados eliminados                                    │
│  • Formatos normalizados                                    │
│  • Outliers corregidos                                      │
│  • Nulos manejados                                          │
│  • Datos sensibles hasheados (GDPR)                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                         CAPA GOLD                           │
│            (Datos Analíticos - Listos ML)                   │
│                                                             │
│  • Features RFM (Recency, Frequency, Monetary)              │
│  • Engagement Score                                         │
│  • Métricas de riesgo                                       │
│  • Variables codificadas                                    │
│  • Dataset optimizado para modelo                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODELADO MULTI-MODELO                    │
│          (RF, Logistic Regression, XGBoost)                 │
│                                                             │
│  • Entrenamiento con validación cruzada                     │
│  • Comparación automática de métricas (F1-Score)            │
│  • Selección automática del mejor modelo                    │
│  • Persistencia del modelo ganador (PKL)                    │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

1. **Bronze**: Carga datos raw sin transformaciones
2. **Silver**: Aplica gobierno de datos y limpieza
3. **Gold**: Genera features avanzadas para modelado
4. **Modelado**: Entrena y evalúa modelo predictivo

---

## 📁 Estructura del Proyecto

```
proyecto_churn/
│
├── datos/
│   ├── bronze/                    # 🥉 Datos raw
│   │   └── raw_data_customers.csv
│   ├── silver/                    # 🥈 Datos limpios
│   │   └── clientes_limpios.csv
│   └── gold/                      # 🥇 Datos analíticos
│       └── clientes_modelado.csv
│
├── notebooks/
│   └── eda_exploratorio.ipynb     # 📊 Análisis exploratorio (Jupyter)
│
├── src/
│   ├── configuracion.py           # ⚙️  Parámetros centralizados
│   │
│   ├── ingestion/
│   │   └── cargar_bronze.py       # 📥 Carga a Bronze
│   │
│   ├── transformacion/
│   │   ├── bronze_a_silver.py     # 🧹 Limpieza
│   │   └── silver_a_gold.py       # ⚡ Feature Engineering
│   │
│   ├── modelado/
│   │   └── entrenar_modelo.py     # 🤖 Entrenamiento ML
│   │
│   └── utilidades/
│       ├── hashing.py             # 🔐 Protección PII
│       ├── limpieza.py            # 🧼 Funciones limpieza
│       └── features.py            # 🎯 Generación features
│
├── modelos/
│   └── modelo_churn.pkl           # 💾 Mejor modelo entrenado (Ganador)
│
├── resultados/
│   ├── metricas/
│   │   └── metricas_todos_modelos.json   # 📈 Métricas de todos los modelos
│   └── visualizaciones/
│       └── evaluacion_[modelo].png      # 📊 Gráficos por modelo
│
├── diagramas/
│   ├── arquitectura_local.png     # 🏗️  Diagrama local
│   └── arquitectura_aws.png       # ☁️  Diagrama AWS
│
├── main.py                        # 🚀 Script principal
├── requirements.txt               # 📦 Dependencias
└── README.md                      # 📖 Este archivo
```

---

## 🛠️ Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- pip

### Instalación

1. **Clonar o descargar el proyecto**

```bash
cd proyecto_churn
```

2. **Crear entorno virtual (recomendado)**

```bash
python -m venv venv

# Activar entorno
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

### Opción 1: Pipeline Completo (Recomendado)

Ejecuta todo el pipeline de principio a fin:

```bash
python main.py
```

Esto ejecutará:
1. ✅ Carga de datos a Bronze
2. ✅ Transformación Bronze → Silver (limpieza)
3. ✅ Transformación Silver → Gold (features)
4. ✅ Entrenamiento del modelo

### Opción 2: Ejecución por Etapas

Si deseas ejecutar solo ciertas etapas:

```bash
# Solo limpieza de datos
python main.py --solo-limpieza

# Hasta feature engineering
python main.py --solo-features

# Solo entrenamiento (requiere Gold existente)
python main.py --solo-modelo
```

### Opción 3: Ejecución Manual por Scripts

Ejecutar cada etapa individualmente:

```bash
# 1. Bronze
python src/ingestion/cargar_bronze.py

# 2. Silver
python src/transformacion/bronze_a_silver.py

# 3. Gold
python src/transformacion/silver_a_gold.py

# 4. Modelo
python src/modelado/entrenar_modelo.py
```

### Análisis Exploratorio (EDA)

Para ejecutar el análisis exploratorio:

```bash
# Si tienes Jupyter instalado
jupyter notebook notebooks/eda_exploratorio.ipynb
```

---

## 💼 Decisiones de Negocio

### 1. Gobierno de Datos

#### Tratamiento de Duplicados

**Problema Detectado:**
- Registros con mismo `customer_id` pero datos diferentes
- Duplicados completos (todas las columnas iguales)

**Decisión:**
- Duplicados completos: Eliminar (información redundante)
- Duplicados por ID: Mantener registro con `last_purchase_date` más reciente
- **Justificación**: El dato más reciente refleja mejor el estado actual del cliente

#### Hashing de Datos Sensibles (PII)

**Columnas Protegidas:**
- `full_name`
- `email`
- `phone`
- `home_address`

**Técnica:** SHA-256 con salt

**Justificación de Negocio:**
- Cumplimiento GDPR/CCPA
- Permite análisis sin exponer información personal
- Mantiene unicidad para joins
- Reproducible con mismo salt

#### Tratamiento de Valores Nulos

| Columna | Estrategia | Justificación de Negocio |
|---------|------------|--------------------------|
| `phone` | Categoría "MISSING" | Ausencia de teléfono es información relevante |
| `monthly_spend` | Mediana | Robusta a outliers, representa cliente típico |
| `total_shipments` | Mediana | Similar, comportamiento promedio |
| `last_purchase_date` | Forward fill | Conservador, usa última fecha conocida |
| `churn_label` | Eliminar registro | Sin etiqueta no se puede entrenar |

#### Detección y Corrección de Outliers

**Umbrales Definidos por Negocio:**

```python
# Gasto mensual
UMBRAL_GASTO_MINIMO = 0      # Negativos son errores
UMBRAL_GASTO_MAXIMO = 15000  # Valores extremos a revisar

# Envíos
UMBRAL_ENVIOS_MAXIMO = 500   # Número razonable de envíos
```

**Estrategia:**
- Gastos negativos → Convertir a 0 (error de registro)
- Gastos > 15,000 → Cap al umbral (preservar cliente de alto valor)
- Envíos extremos → Cap (pueden ser clientes VIP legítimos)

---

### 2. Feature Engineering

#### Métricas RFM

Implementación del modelo RFM (Recency, Frequency, Monetary):

**R - Recency (Recencia)**
- **Métrica**: Días desde última compra
- **Negocio**: Indicador más fuerte de churn inminente
- **Categorías**: Muy Reciente (<30d), Reciente (30-90d), Inactivo (90-180d), Muy Inactivo (>180d)

**F - Frequency (Frecuencia)**
- **Métrica**: Total de envíos históricos
- **Negocio**: Indica lealtad y hábito de compra
- **Segmentos**: Ocasional (<10), Regular (10-30), Frecuente (30-100), VIP (>100)

**M - Monetary (Monetario)**
- **Métrica**: Gasto mensual promedio
- **Negocio**: Valor económico del cliente
- **Segmentos**: Bajo (<$500), Medio ($500-1,500), Alto ($1,500-5,000), Premium (>$5,000)

#### Engagement Score

**Fórmula Ponderada:**

```python
Engagement = (Recencia_Norm × 0.4) + 
             (Frecuencia_Norm × 0.3) + 
             (Gasto_Norm × 0.3)
```

**Pesos Justificados:**
- **Recencia (40%)**: Indicador más fuerte de churn inmediato
- **Frecuencia (30%)**: Indica lealtad y engagement
- **Gasto (30%)**: Representa valor económico

**Uso de Negocio:**
- Score bajo (<25): Alta prioridad para retención
- Score alto (>75): Clientes leales, enfoque en upsell

#### Features de Riesgo

Variables creadas específicamente para identificar clientes en peligro:

1. **`riesgo_inactividad`**: Recencia > 180 días
2. **`riesgo_bajo_engagement`**: Score < 30
3. **`riesgo_nuevo_inactivo`**: Cliente reciente sin actividad
4. **`score_riesgo_churn`**: Métrica combinada de señales de riesgo

---

### 3. Modelo de Machine Learning

#### Selección del Modelo: Enfoque Multi-Modelo

El sistema evalúa automáticamente tres algoritmos para encontrar el mejor balance entre complejidad y capacidad de generalización:

| Modelo | Rol | Ventajas |
|--------|-----|----------|
| **XGBoost** | SOTA | Alta precisión, manejo nativo de nulos y regularización avanzada. |
| **Random Forest** | Baseline Robusto | Excelente interpretabilidad y manejo automático de features. |
| **Logistic Regression** | Baseline Simple | Alta eficiencia y base probabilística sólida. |

**Criterio de Selección:** El sistema selecciona el modelo con el mejor **CV F1-Score** (validación cruzada). 

**Innovación Dinámica:** A diferencia de procesos estáticos, este pipeline genera una **Razón Técnica de Selección** automática que compara el desempeño del ganador contra los competidores en puntos reales de F1-Score, garantizando transparencia en la decisión.

#### Configuración de Modelos
Los parámetros están optimizados para evitar overfitting mediante regularización y poda. El sistema permite al usuario experimentar con la estabilidad del modelo cambiando el `RANDOM_STATE` en `configuracion.py` para verificar la consistencia de los resultados.

#### Métricas Priorizadas

Para churn, las métricas tienen diferentes implicaciones de negocio:

| Métrica | Importancia | Interpretación de Negocio |
|---------|-------------|---------------------------|
| **Recall** | Alta | Identificar máximo de clientes en riesgo real. FN = Cliente perdido sin intervención |
| **Precision** | Media-Alta | Evitar falsos positivos. FP = Esfuerzo de retención desperdiciado |
| **F1-Score** | Alta | Balance entre Recall y Precision |
| **ROC-AUC** | Media | Capacidad discriminativa general del modelo |

**Decisión de Negocio:**
- Priorizar **Recall** sobre Precision.
- **Razón**: Es más costoso perder un cliente real (Churn) que realizar una campaña de marketing preventiva sobre un falso positivo.

---

## 📊 Resultados

### Métricas del Modelo

Los resultados se generan dinámicamente en cada ejecución y se consolidan en:
- `resultados/metricas/metricas_todos_modelos.json` (Informe técnico exhaustivo con comparativas).
- `resultados/visualizaciones/evaluacion_[modelo].png` (Curvas ROC y Matrices de Confusión).

**Ejemplo de Reporte Dinámico Sugerido:**
> *"El modelo RandomForest fue seleccionado como ganador debido a que obtuvo el mayor CV F1-Score promedio (0.7989) entre todos los competidores... Comparado con: LogisticRegression (Dif: +0.1533)"*

### Top Features Importantes

El análisis de importancia identifica los factores críticos que disparan el churn:

### Interpretación de Negocio

La evaluación final permite segmentar la base de clientes en:
- **True Positives (TP)**: Clientes identificados con éxito en riesgo de abandono. Objetivo de campañas de retención.
- **False Positives (FP)**: Clientes leales que el modelo marcó como riesgo. Representan un "seguro de retención" pero con costo operativo.
- **False Negatives (FN)**: El escenario más costoso. Clientes que abandonan sin que el modelo envíe una alerta.

> 📊 *Visualización de Matriz de Confusión y Curva ROC disponible en el directorio `resultados/visualizaciones/`.*

---

## ☁️ Preparación para AWS

### Arquitectura Conceptual en AWS

```
┌────────────────────────────────────────────────────────────┐
│                      INGESTA DE DATOS                      │
│                                                            │
│  S3 Bucket (Bronze)                                        │
│  └── raw_data_customers.csv                                │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                   TRANSFORMACIÓN (ETL)                     │
│                                                            │
│  AWS Glue Jobs                                             │
│  ├── Job 1: Bronze → Silver (Limpieza)                     │
│  └── Job 2: Silver → Gold (Features)                       │
│                                                            │
│  AWS Glue Data Catalog                                     │
│  └── churn_db (Base de datos)                              │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                  ALMACENAMIENTO PROCESADO                  │
│                                                            │
│  S3 Bucket (Silver)                                        │
│  └── clientes_limpios/                                     │
│                                                            │
│  S3 Bucket (Gold)                                          │
│  └── clientes_modelado/                                    │
└──────────────┬─────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────┐
│                  ENTRENAMIENTO ML                          │
│                                                            │
│  Amazon SageMaker                                          │
│  ├── Training Job (scikit-learn)                           │
│  ├── Model Registry (Gobierno de Versiones y Métricas)     │
│  └── Endpoint (Inferencia de Churn en Tiempo Real)         │
│                                                            │
│  S3 Bucket (Modelos)                                       │
│  └── modelo_churn_vX.tar.gz                                │
└──────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────────┐
│                   MONITOREO Y ORQUESTACIÓN                 │
│                                                            │
│  CloudWatch (Logs y Métricas Operativas)                   │
│  Step Functions (Orquestación del Pipeline Medallion)      │
│  EventBridge (Scheduling de Ingesta Diaria)                │
└────────────────────────────────────────────────────────────┘
```

### Componentes AWS

#### 1. Amazon S3 (Almacenamiento)

**Buckets por Capa:**
- `s3://churn-data-bronze/` - Datos raw
- `s3://churn-data-silver/` - Datos limpios
- `s3://churn-data-gold/` - Datos analíticos
- `s3://churn-models/` - Modelos entrenados

**Versionado:** Habilitado en todos los buckets para trazabilidad

#### 2. AWS Glue (ETL)

**Jobs de Transformación:**
- `job_bronze_to_silver.py` - Limpieza y gobierno
- `job_silver_to_gold.py` - Feature engineering

**Data Catalog:**
- Base de datos: `churn_db`
- Tablas: `clientes_bronze`, `clientes_silver`, `clientes_gold`

#### 3. Amazon SageMaker (ML)

**Training:**
- Tipo de instancia: `ml.m5.large`
- Framework: scikit-learn
- Script de entrenamiento: `entrenar_modelo.py`

**Endpoint:**
- Nombre: `churn-predictor`
- Auto-scaling: Basado en invocaciones
- Monitoreo: CloudWatch Model Monitor

#### 4. Orquestación

**AWS Step Functions:**
```json
{
  "Estado Inicial": "Cargar Bronze",
  "Estados": {
    "Cargar Bronze": {
      "Tipo": "Glue Job",
      "Siguiente": "Transformar a Silver"
    },
    "Transformar a Silver": {
      "Tipo": "Glue Job",
      "Siguiente": "Transformar a Gold"
    },
    "Transformar a Gold": {
      "Tipo": "Glue Job",
      "Siguiente": "Entrenar Modelo"
    },
    "Entrenar Modelo": {
      "Tipo": "SageMaker Training",
      "Siguiente": "Evaluar Modelo"
    }
  }
}
```

**Scheduling:**
- EventBridge: Ejecución diaria a las 02:00 AM
- Trigger: Nuevos archivos en S3 Bronze

### Migración del Código Local a AWS

**Cambios Necesarios:**

1. **Configuración de S3:**
```python
import boto3

s3 = boto3.client('s3')

# En lugar de:
df = pd.read_csv('datos/bronze/raw_data_customers.csv')

# Usar:
obj = s3.get_object(Bucket='churn-data-bronze', Key='raw_data_customers.csv')
df = pd.read_csv(obj['Body'])
```

2. **Persistencia de Modelos:**
```python
import joblib
import boto3

# Guardar modelo en S3
with open('/tmp/modelo_churn.pkl', 'wb') as f:
    joblib.dump(modelo, f)

s3.upload_file('/tmp/modelo_churn.pkl', 
               'churn-models', 
               'modelo_churn_v1.pkl')
```

3. **Glue Jobs:**
```python
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame

# Leer desde Catalog
df = glueContext.create_dynamic_frame.from_catalog(
    database='churn_db',
    table_name='clientes_bronze'
).toDF()
```

### Costos Estimados (Aproximados)

**Componentes Principales:**

| Servicio | Uso Mensual | Costo Aprox |
|----------|-------------|-------------|
| S3 (3 buckets) | 10 GB | $0.23 |
| Glue (2 jobs/día) | 60 DPU-hours | $13.20 |
| SageMaker Training | 1 job/día, 0.5h | $37.80 |
| SageMaker Endpoint | 1 instancia 24/7 | $69.12 |
| CloudWatch | Logs estándar | $5.00 |
| **Total Mensual** | | **~$125** |

**Nota**: Costos estimados para región us-east-1, pueden variar

### Seguridad y Cumplimiento

**IAM Roles:**
- `GlueServiceRole` - Para jobs de ETL
- `SageMakerExecutionRole` - Para training/inference
- `LambdaExecutionRole` - Para funciones auxiliares

**Encriptación:**
- S3: SSE-S3 o SSE-KMS
- SageMaker: Encriptación en tránsito y reposo
- Glue: Encriptación de datos en ETL

**VPC:**
- Endpoints privados para SageMaker
- Security Groups restrictivos
- No exposición pública de endpoints

---

## 📚 Documentación Adicional

### Archivos de Configuración

Todos los parámetros del proyecto están centralizados en:
- `src/configuracion.py`

Para modificar comportamientos, editar este archivo.

### Logs y Trazabilidad

El proyecto genera logs detallados en cada etapa:
- Mensajes informativos con emoji para facilitar lectura
- Registro de decisiones tomadas
- Métricas de calidad en cada transformación

### Reproducibilidad

**Semillas Aleatorias:**
```python
RANDOM_STATE = 42  # Usado en todos los procesos estocásticos
```

**Versionado de Datos:**
- Bronze: Datos con timestamp de ingesta
- Silver/Gold: Metadata de transformación guardada

---

## 👨‍💻 Autor

**Prueba Técnica - Científico de Datos**

Desarrollado con enfoque en:
- ✅ Gobierno de datos y privacidad
- ✅ Decisiones basadas en negocio
- ✅ Código limpio y documentado
- ✅ Arquitectura escalable
- ✅ Preparación para producción en AWS

---

## 📄 Licencia

Este proyecto es parte de una prueba técnica para evaluación de capacidades en ciencia de datos y MLOps.

---

## 🔗 Referencias

- [AWS Glue Documentation](https://docs.aws.amazon.com/glue/)
- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)