"""
Script de Transformación - SILVER → GOLD

Este script genera features para modelado.
Objetivo: Dataset analítico optimizado para predicción de churn.

Procesos aplicados:
-------------------
1. Feature Engineering (RFM, engagement, comportamiento)
2. Encoding de variables categóricas
3. Selección de features relevantes
4. Dataset final para modelado

Decisiones de Negocio:
---------------------
Las features creadas tienen justificación clara basada en:
- Métricas RFM (Recency, Frequency, Monetary)
- Patrones de comportamiento del cliente
- Indicadores de riesgo de churn
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configuracion import (
    RUTA_SILVER,
    RUTA_GOLD,
    ARCHIVO_SILVER,
    ARCHIVO_GOLD,
    log_mensaje
)

from utilidades.features import generar_todas_features


def transformar_silver_a_gold() -> pd.DataFrame:
    """
    Pipeline completo de transformación Silver → Gold.
    
    Returns:
        DataFrame con features para modelado
    """
    log_mensaje("="*70, "INFO")
    log_mensaje("TRANSFORMACIÓN: SILVER → GOLD", "INFO")
    log_mensaje("="*70, "INFO")
    
    # ========================================================================
    # PASO 1: CARGAR DATOS SILVER
    # ========================================================================
    
    log_mensaje("\n[1/4] Cargando datos de capa Silver...", "INFO")
    ruta_silver = RUTA_SILVER / ARCHIVO_SILVER
    
    if not ruta_silver.exists():
        log_mensaje(f"Archivo Silver no encontrado: {ruta_silver}", "ERROR")
        log_mensaje("Ejecutar primero: bronze_a_silver.py", "ERROR")
        raise FileNotFoundError(f"No se encuentra: {ruta_silver}")
    
    df = pd.read_csv(ruta_silver)
    
    log_mensaje(f"✓ Datos Silver cargados: {len(df)} registros, {len(df.columns)} columnas", "SUCCESS")
    
    # Verificar que tengamos las columnas necesarias
    columnas_requeridas = ['customer_id', 'signup_date', 'last_purchase_date', 
                          'monthly_spend', 'total_shipments', 'churn_label']
    
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if columnas_faltantes:
        log_mensaje(f"⚠️  Columnas faltantes: {columnas_faltantes}", "ERROR")
        raise ValueError(f"Faltan columnas requeridas: {columnas_faltantes}")
    
    # Convertir fechas a datetime
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    
    # ========================================================================
    # PASO 2: GENERAR FEATURES
    # ========================================================================
    
    log_mensaje("\n[2/4] Generando features de modelado...", "INFO")
    
    columnas_iniciales = len(df.columns)
    df = generar_todas_features(df)
    columnas_finales = len(df.columns)
    features_nuevas = columnas_finales - columnas_iniciales
    
    log_mensaje(f"✓ Features generadas: {features_nuevas} nuevas columnas", "SUCCESS")
    
    # ========================================================================
    # PASO 3: PREPARAR DATASET PARA MODELADO
    # ========================================================================
    
    log_mensaje("\n[3/4] Preparando dataset para modelado...", "INFO")
    
    # Seleccionar features relevantes para el modelo
    # Excluir: datos hasheados, IDs, fechas originales, features intermedias
    
    columnas_excluir = [
        # Identificadores y datos hasheados
        'customer_id',
        'full_name_hash',
        'email_hash',
        'phone_hash',
        'home_address_hash',
        
        # Fechas (ya tenemos recencia y antigüedad)
        'signup_date',
        'last_purchase_date',
        
        # Features categóricas que vamos a codificar
        # (mantener las numéricas y crear dummies de categóricas)
    ]
    
    # Identificar columnas categóricas a codificar
    # Excluir explícitamente identificadores y datos hasheados
    todas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Columnas a excluir de encodings (IDs, Hashes, Fechas, Target)
    cols_ignorar_encoding = [
        'customer_id', 'churn_label',
        'full_name_hash', 'email_hash', 'phone_hash', 'home_address_hash',
        'signup_date', 'last_purchase_date'
    ]
    
    columnas_categoricas = [col for col in todas_categoricas if col not in cols_ignorar_encoding]
    
    log_mensaje(f"Columnas categóricas a codificar: {columnas_categoricas}", "INFO")
    
    # Crear variables dummy (one-hot encoding) para categóricas
    if columnas_categoricas:
        df_encoded = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)
        log_mensaje(f"✓ Variables categóricas codificadas", "SUCCESS")
    else:
        df_encoded = df.copy()
    
    # Eliminar columnas no necesarias
    columnas_a_eliminar = [col for col in columnas_excluir if col in df_encoded.columns]
    df_gold = df_encoded.drop(columns=columnas_a_eliminar)
    
    log_mensaje(f"✓ Dataset preparado: {len(df_gold.columns)} features finales", "SUCCESS")
    
    # ========================================================================
    # PASO 4: VALIDAR Y GUARDAR
    # ========================================================================
    
    log_mensaje("\n[4/4] Validando dataset Gold...", "INFO")
    
    # Verificar que tenemos la variable objetivo
    if 'churn_label' not in df_gold.columns:
        log_mensaje("⚠️  Variable objetivo 'churn_label' no encontrada", "ERROR")
        raise ValueError("Falta la variable objetivo")
    
    # Estadísticas de la variable objetivo
    distribucion_churn = df_gold['churn_label'].value_counts()
    log_mensaje("Distribución de churn_label:", "INFO")
    for clase, cantidad in distribucion_churn.items():
        porcentaje = (cantidad / len(df_gold)) * 100
        etiqueta = "No Churn" if clase == 0 else "Churn"
        log_mensaje(f"  - {etiqueta} ({clase}): {cantidad} ({porcentaje:.1f}%)", "INFO")
    
    # Verificar balance de clases
    tasa_churn = (distribucion_churn.get(1, 0) / len(df_gold)) * 100
    if tasa_churn < 10 or tasa_churn > 50:
        log_mensaje(f"⚠️  Dataset desbalanceado: {tasa_churn:.1f}% churn", "WARNING")
        log_mensaje("   Recomendación: Usar class_weight='balanced' en modelo", "INFO")
    
    # Verificar valores nulos en features
    nulos_features = df_gold.isnull().sum().sum()
    if nulos_features > 0:
        log_mensaje(f"⚠️  Valores nulos en features: {nulos_features}", "WARNING")
        # Imputar nulos en features numéricas si existen
        columnas_numericas = df_gold.select_dtypes(include=[np.number]).columns
        for col in columnas_numericas:
            if df_gold[col].isnull().sum() > 0 and col != 'churn_label':
                df_gold[col] = df_gold[col].fillna(df_gold[col].median())
                log_mensaje(f"    - {col}: nulos imputados con mediana", "INFO")
    
    # Guardar dataset Gold
    log_mensaje("\nGuardando dataset en capa GOLD...", "INFO")
    
    ruta_gold = RUTA_GOLD / ARCHIVO_GOLD
    df_gold.to_csv(ruta_gold, index=False)
    
    log_mensaje(f"✓ Dataset Gold guardado en: {ruta_gold}", "SUCCESS")
    
    # Reporte final
    log_mensaje("\nReporte Final - Dataset Gold:", "INFO")
    log_mensaje(f"  - Registros: {len(df_gold)}", "INFO")
    log_mensaje(f"  - Features: {len(df_gold.columns) - 1} (+ 1 target)", "INFO")
    log_mensaje(f"  - Tasa de Churn: {tasa_churn:.1f}%", "INFO")
    log_mensaje(f"  - Memoria: {df_gold.memory_usage(deep=True).sum() / 1024**2:.2f} MB", "INFO")
    
    log_mensaje("\n" + "="*70, "SUCCESS")
    log_mensaje("TRANSFORMACIÓN SILVER → GOLD COMPLETADA", "SUCCESS")
    log_mensaje("="*70, "SUCCESS")
    
    return df_gold


def main():
    """
    Función principal para ejecutar transformación Silver → Gold.
    """
    try:
        df_gold = transformar_silver_a_gold()
        
        log_mensaje("\n✅ Dataset analítico listo en capa GOLD", "SUCCESS")
        log_mensaje("➡️  Siguiente paso: Ejecutar entrenar_modelo.py", "INFO")
        
        # Mostrar preview
        log_mensaje("\nVista previa de datos Gold:", "INFO")
        print(df_gold.head(3))
        
        log_mensaje("\nColumnas en dataset Gold:", "INFO")
        print(df_gold.columns.tolist())
        
        return df_gold
        
    except Exception as e:
        log_mensaje(f"Error en transformación Gold: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    df = main()