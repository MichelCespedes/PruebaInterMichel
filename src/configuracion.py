"""
Configuración centralizada del proyecto de predicción de Churn.
Contiene todas las rutas, parámetros y constantes del proyecto.
"""

import os
from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================

# Directorio raíz del proyecto
RUTA_RAIZ = Path(__file__).parent.parent

# Rutas de datos - Arquitectura Medallion
RUTA_BRONZE = RUTA_RAIZ / "datos" / "bronze"
RUTA_SILVER = RUTA_RAIZ / "datos" / "silver"
RUTA_GOLD = RUTA_RAIZ / "datos" / "gold"

# Rutas de resultados
RUTA_MODELOS = RUTA_RAIZ / "modelos"
RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
RUTA_METRICAS = RUTA_RESULTADOS / "metricas"
RUTA_VISUALIZACIONES = RUTA_RESULTADOS / "visualizaciones"

# Crear carpetas si no existen
for ruta in [RUTA_BRONZE, RUTA_SILVER, RUTA_GOLD, RUTA_MODELOS, 
             RUTA_METRICAS, RUTA_VISUALIZACIONES]:
    ruta.mkdir(parents=True, exist_ok=True)

# ============================================================================
# NOMBRES DE ARCHIVOS
# ============================================================================

# Bronze
ARCHIVO_BRONZE = "raw_data_customers.csv"

# Silver
ARCHIVO_SILVER = "clientes_limpios.csv"

# Gold
ARCHIVO_GOLD = "clientes_modelado.csv"

# Modelos
ARCHIVO_MODELO = "modelo_churn.pkl"
ARCHIVO_PREPROCESADOR = "preprocesador.pkl"

# ============================================================================
# CONFIGURACIÓN DE DATOS SENSIBLES
# ============================================================================

# Columnas que contienen información personal identificable (PII)
COLUMNAS_SENSIBLES = [
    'full_name',
    'email', 
    'phone',
    'home_address'
]

# Salt para hashing (en producción, esto debe estar en variables de entorno)
# Importante: Este valor debe mantenerse constante para garantizar reproducibilidad
SALT_HASHING = "proyecto_churn_2025_salt_secreto"

# ============================================================================
# CONFIGURACIÓN DE LIMPIEZA DE DATOS
# ============================================================================

# Estrategia para valores nulos
ESTRATEGIA_NULOS = {
    'phone': 'MISSING',  # Imputar con valor especial
    'monthly_spend': 'mediana',  # Imputar con mediana del grupo
    'total_shipments': 'mediana',
    'last_purchase_date': 'ffill',  # Forward fill basado en lógica de negocio
    'churn_label': 'eliminar'  # Eliminar registros sin etiqueta
}

# Umbrales para detección de outliers
UMBRAL_GASTO_MINIMO = 0  # Gastos negativos son errores
UMBRAL_GASTO_MAXIMO = 15000  # Gastos superiores requieren revisión
UMBRAL_ENVIOS_MAXIMO = 500  # Número de envíos razonable

# ============================================================================
# CONFIGURACIÓN DE FEATURE ENGINEERING
# ============================================================================

# Ventanas temporales para análisis de comportamiento
VENTANA_RECENCIA_DIAS = 180  # 6 meses

# Bins para categorización de variables continuas
BINS_GASTO_MENSUAL = [0, 500, 1500, 5000, float('inf')]
ETIQUETAS_GASTO = ['Bajo', 'Medio', 'Alto', 'Premium']

BINS_FRECUENCIA_ENVIOS = [0, 10, 30, 100, float('inf')]
ETIQUETAS_FRECUENCIA = ['Ocasional', 'Regular', 'Frecuente', 'VIP']

# ============================================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================================

# Parámetros de división de datos
TEST_SIZE = 0.25
RANDOM_STATE = 42
STRATIFY = True  # Mantener proporción de churn en train/test

# Parámetros del modelo (Random Forest como baseline)
# Parámetros de modelos (Hyperparameter Tuning)
# Se definen parámetros conservadores para evitar overfitting

MODELOS_CONFIG = {
    'RandomForest': {
        'n_estimators': 50,         # Reducido de 100 a 50 (menos árboles)
        'max_depth': 3,             # Reducido de 8 a 3 (árboles más simples)
        'min_samples_split': 30,    # Aumentado de 20 a 30 (más datos para dividir)
        'min_samples_leaf': 15,     # Aumentado de 10 a 15 (hojas más grandes)
        'max_features': 'sqrt',     # Solo usar sqrt(n_features) por split
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
        'n_jobs': -1
    },
    'LogisticRegression': {
        'penalty': 'l2',            # Regularización Ridge
        'C': 0.01,                  # Aumentada regularización de 0.1 a 0.01 (10x más fuerte)
        'solver': 'liblinear',
        'class_weight': 'balanced',
        'max_iter': 1000,           # Más iteraciones para convergencia
        'random_state': RANDOM_STATE
    },
    'XGBoost': {
        'n_estimators': 25,         # Punto medio conservador
        'learning_rate': 0.015,     # Aprendizaje moderadamente lento
        'max_depth': 2,             # Árboles simples
        'min_child_weight': 12,     # Restrictivo pero no extremo
        'subsample': 0.55,          # Muestreo moderado
        'colsample_bytree': 0.55,   # Muestreo moderado de features
        'gamma': 0.7,               # Penalización moderada-alta
        'reg_alpha': 0.7,           # Regularización L1 moderada-alta
        'reg_lambda': 2.5,          # Regularización L2 alta
        'scale_pos_weight': 3,      # Balance de clases
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
}

# Umbrales de decisión
UMBRAL_PROBABILIDAD_CHURN = 0.5  # Probabilidad para clasificar como churn

# ============================================================================
# CONFIGURACIÓN DE MÉTRICAS
# ============================================================================

# Métricas de evaluación prioritarias para el negocio
# Justificación: En churn, es crítico identificar clientes en riesgo (Recall)
# pero también evitar falsos positivos costosos (Precision)
METRICAS_PRINCIPALES = [
    'accuracy',
    'precision', 
    'recall',
    'f1',
    'roc_auc'
]

# ============================================================================
# CONFIGURACIÓN AWS (CONCEPTUAL)
# ============================================================================

# Configuración para futura implementación en AWS
AWS_CONFIG = {
    'region': 'us-east-1',
    's3': {
        'bucket_bronze': 'churn-data-bronze',
        'bucket_silver': 'churn-data-silver', 
        'bucket_gold': 'churn-data-gold',
        'bucket_modelos': 'churn-models'
    },
    'glue': {
        'database': 'churn_db',
        'crawler_bronze': 'crawler_bronze_customers'
    },
    'sagemaker': {
        'instance_type': 'ml.m5.large',
        'endpoint_name': 'churn-predictor'
    }
}

# ============================================================================
# MENSAJES Y LOGGING
# ============================================================================

VERBOSE = True  # Mostrar mensajes detallados durante ejecución

def log_mensaje(mensaje, tipo="INFO"):
    """
    Función simple para logging consistente.
    
    Args:
        mensaje (str): Mensaje a mostrar
        tipo (str): Tipo de mensaje (INFO, WARNING, ERROR, SUCCESS)
    """
    if VERBOSE:
        prefijos = {
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        print(f"{prefijos.get(tipo, 'ℹ️')} {mensaje}")