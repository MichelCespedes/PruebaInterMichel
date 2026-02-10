"""
Utilidades para Feature Engineering.

Este módulo genera variables derivadas relevantes para el modelo de predicción de churn.
Todas las features tienen justificación de negocio clara.

Features Creadas:
-----------------
1. Recency: Días desde la última compra
2. Frequency: Total de envíos (ya existe, pero se categoriza)
3. Monetary: Gasto mensual (ya existe, pero se categoriza)
4. Tenure: Antigüedad del cliente
5. Engagement Score: Métrica combinada de actividad
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configuracion import (
    log_mensaje,
    VENTANA_RECENCIA_DIAS,
    BINS_GASTO_MENSUAL,
    ETIQUETAS_GASTO,
    BINS_FRECUENCIA_ENVIOS,
    ETIQUETAS_FRECUENCIA
)


def calcular_recencia(df: pd.DataFrame,
                     columna_fecha: str = 'last_purchase_date',
                     fecha_referencia: datetime = None) -> pd.DataFrame:
    """
    Calcula la recencia (días desde la última compra).
    
    Decisión de Negocio:
        La recencia es uno de los indicadores más fuertes de churn.
        Clientes que no compran en mucho tiempo tienen alta probabilidad de abandono.
    
    Métrica RFM:
        R = Recency (qué tan reciente)
        F = Frequency (qué tan frecuente)
        M = Monetary (cuánto gasta)
    
    Args:
        df: DataFrame con fechas de compra
        columna_fecha: Nombre de la columna de fecha
        fecha_referencia: Fecha de referencia (default: hoy)
        
    Returns:
        DataFrame con columna 'recencia_dias'
    """
    log_mensaje("Calculando recencia de clientes", "INFO")
    
    if fecha_referencia is None:
        # Usar la fecha máxima en los datos como referencia
        fecha_referencia = df[columna_fecha].max()
    
    log_mensaje(f"Fecha de referencia: {fecha_referencia.date()}", "INFO")
    
    # Calcular días desde última compra
    df['recencia_dias'] = (fecha_referencia - df[columna_fecha]).dt.days
    
    # Crear categorías de recencia
    df['categoria_recencia'] = pd.cut(
        df['recencia_dias'],
        bins=[0, 30, 90, 180, float('inf')],
        labels=['Muy_Reciente', 'Reciente', 'Inactivo', 'Muy_Inactivo']
    )
    
    log_mensaje(f"✓ Recencia calculada. Media: {df['recencia_dias'].mean():.1f} días", "SUCCESS")
    
    return df


def calcular_antiguedad(df: pd.DataFrame,
                       columna_signup: str = 'signup_date',
                       fecha_referencia: datetime = None) -> pd.DataFrame:
    """
    Calcula la antigüedad del cliente (tenure).
    
    Decisión de Negocio:
        Clientes más antiguos pueden tener mayor lealtad, pero también
        pueden estar en riesgo si su comportamiento ha decaído.
    
    Args:
        df: DataFrame con fechas de registro
        columna_signup: Columna con fecha de registro
        fecha_referencia: Fecha de referencia (default: hoy)
        
    Returns:
        DataFrame con columna 'antiguedad_dias'
    """
    log_mensaje("Calculando antigüedad de clientes", "INFO")
    
    if fecha_referencia is None:
        fecha_referencia = df['last_purchase_date'].max()
    
    # Calcular días desde registro
    df['antiguedad_dias'] = (fecha_referencia - df[columna_signup]).dt.days
    
    # Categorizar antigüedad
    df['categoria_antiguedad'] = pd.cut(
        df['antiguedad_dias'],
        bins=[0, 180, 365, 730, float('inf')],
        labels=['Nuevo', 'Establecido', 'Veterano', 'Antiguo']
    )
    
    log_mensaje(f"✓ Antigüedad calculada. Media: {df['antiguedad_dias'].mean():.1f} días", "SUCCESS")
    
    return df


def categorizar_monetary(df: pd.DataFrame,
                        columna_gasto: str = 'monthly_spend') -> pd.DataFrame:
    """
    Categoriza el gasto mensual en segmentos.
    
    Decisión de Negocio:
        Segmentar clientes por nivel de gasto permite:
        - Estrategias diferenciadas de retención
        - Priorización de recursos (clientes Premium vs Bajo)
        - Análisis de churn por segmento de valor
    
    Segmentos:
        - Bajo: < $500
        - Medio: $500 - $1,500
        - Alto: $1,500 - $5,000
        - Premium: > $5,000
    
    Args:
        df: DataFrame con gastos
        columna_gasto: Columna con gasto mensual
        
    Returns:
        DataFrame con categoría de gasto
    """
    log_mensaje("Categorizando nivel de gasto", "INFO")
    
    df['segmento_gasto'] = pd.cut(
        df[columna_gasto],
        bins=BINS_GASTO_MENSUAL,
        labels=ETIQUETAS_GASTO
    )
    
    # Distribución de segmentos
    distribucion = df['segmento_gasto'].value_counts()
    log_mensaje(f"Distribución de segmentos:", "INFO")
    for segmento, cantidad in distribucion.items():
        porcentaje = (cantidad / len(df)) * 100
        log_mensaje(f"  - {segmento}: {cantidad} ({porcentaje:.1f}%)", "INFO")
    
    return df


def categorizar_frequency(df: pd.DataFrame,
                         columna_envios: str = 'total_shipments') -> pd.DataFrame:
    """
    Categoriza la frecuencia de compras.
    
    Decisión de Negocio:
        La frecuencia de compra indica engagement del cliente.
        Clientes con alta frecuencia son más valiosos y menos propensos a churn.
    
    Segmentos:
        - Ocasional: < 10 envíos
        - Regular: 10-30 envíos
        - Frecuente: 30-100 envíos
        - VIP: > 100 envíos
    
    Args:
        df: DataFrame con envíos
        columna_envios: Columna con total de envíos
        
    Returns:
        DataFrame con categoría de frecuencia
    """
    log_mensaje("Categorizando frecuencia de compras", "INFO")
    
    df['segmento_frecuencia'] = pd.cut(
        df[columna_envios],
        bins=BINS_FRECUENCIA_ENVIOS,
        labels=ETIQUETAS_FRECUENCIA
    )
    
    distribucion = df['segmento_frecuencia'].value_counts()
    log_mensaje(f"Distribución de frecuencia:", "INFO")
    for segmento, cantidad in distribucion.items():
        porcentaje = (cantidad / len(df)) * 100
        log_mensaje(f"  - {segmento}: {cantidad} ({porcentaje:.1f}%)", "INFO")
    
    return df


def calcular_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un score de engagement combinando múltiples métricas.
    
    Decisión de Negocio:
        Un score único de engagement facilita:
        - Identificación rápida de clientes en riesgo
        - Segmentación para campañas de retención
        - Priorización de esfuerzos comerciales
    
    Fórmula:
        Engagement = (Recencia_Normalizada * 0.4) + 
                    (Frecuencia_Normalizada * 0.3) + 
                    (Gasto_Normalizado * 0.3)
    
    Pesos justificados:
        - Recencia (40%): Indicador más fuerte de churn inmediato
        - Frecuencia (30%): Indica lealtad y hábito
        - Gasto (30%): Indica valor del cliente
    
    Args:
        df: DataFrame con métricas RFM
        
    Returns:
        DataFrame con 'engagement_score' (0-100)
    """
    log_mensaje("Calculando Engagement Score", "INFO")
    
    # Normalizar recencia (invertir: menos días = mejor)
    recencia_max = df['recencia_dias'].max()
    df['recencia_norm'] = 1 - (df['recencia_dias'] / recencia_max)
    
    # Normalizar frecuencia
    frecuencia_max = df['total_shipments'].max()
    df['frecuencia_norm'] = df['total_shipments'] / frecuencia_max
    
    # Normalizar gasto
    gasto_max = df['monthly_spend'].max()
    df['gasto_norm'] = df['monthly_spend'] / gasto_max
    
    # Calcular score ponderado (0-100)
    df['engagement_score'] = (
        df['recencia_norm'] * 0.4 +
        df['frecuencia_norm'] * 0.3 +
        df['gasto_norm'] * 0.3
    ) * 100
    
    # Categorizar engagement
    df['nivel_engagement'] = pd.cut(
        df['engagement_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Bajo', 'Medio', 'Alto', 'Muy_Alto']
    )
    
    # Eliminar columnas temporales de normalización
    df = df.drop(columns=['recencia_norm', 'frecuencia_norm', 'gasto_norm'])
    
    log_mensaje(f"✓ Engagement Score calculado. Media: {df['engagement_score'].mean():.1f}", "SUCCESS")
    
    distribucion = df['nivel_engagement'].value_counts()
    for nivel, cantidad in distribucion.items():
        porcentaje = (cantidad / len(df)) * 100
        log_mensaje(f"  - {nivel}: {cantidad} ({porcentaje:.1f}%)", "INFO")
    
    return df


def crear_features_comportamiento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features adicionales de comportamiento del cliente.
    
    Features:
        - gasto_por_envio: Ticket promedio
        - dias_entre_compras: Frecuencia temporal
        - tendencia_gasto: Si gasto está aumentando/disminuyendo
    
    Args:
        df: DataFrame con datos de clientes
        
    Returns:
        DataFrame con features de comportamiento
    """
    log_mensaje("Creando features de comportamiento", "INFO")
    
    # Gasto promedio por envío
    df['gasto_por_envio'] = df['monthly_spend'] / df['total_shipments'].replace(0, 1)
    
    # Categorizar ticket promedio
    df['categoria_ticket'] = pd.cut(
        df['gasto_por_envio'],
        bins=[0, 50, 100, 200, float('inf')],
        labels=['Bajo', 'Medio', 'Alto', 'Premium']
    )
    
    # Días promedio entre compras (aproximado)
    df['dias_entre_compras'] = df['antiguedad_dias'] / df['total_shipments'].replace(0, 1)
    
    # Flag de cliente activo reciente
    df['cliente_activo_reciente'] = (df['recencia_dias'] <= 90).astype(int)
    
    # Flag de cliente de alto valor
    df['cliente_alto_valor'] = ((df['monthly_spend'] > 1500) | 
                                (df['total_shipments'] > 50)).astype(int)
    
    log_mensaje("✓ Features de comportamiento creadas", "SUCCESS")
    
    return df


def crear_features_riesgo_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features específicas para predicción de churn.
    
    Decisión de Negocio:
        Estas features son señales explícitas de riesgo de abandono
        basadas en conocimiento del negocio logístico.
    
    Señales de Riesgo:
        - Inactividad prolongada (>180 días)
        - Reducción drástica en frecuencia
        - Cliente con bajo engagement
        - Antigüedad muy baja (no estableció hábito)
    
    Args:
        df: DataFrame con features previas
        
    Returns:
        DataFrame con features de riesgo
    """
    log_mensaje("Creando features de riesgo de churn", "INFO")
    
    # Flag de inactividad prolongada
    df['riesgo_inactividad'] = (df['recencia_dias'] > VENTANA_RECENCIA_DIAS).astype(int)
    
    # Flag de bajo engagement
    df['riesgo_bajo_engagement'] = (df['engagement_score'] < 30).astype(int)
    
    # Flag de cliente nuevo sin actividad reciente
    df['riesgo_nuevo_inactivo'] = (
        (df['antiguedad_dias'] < 180) & 
        (df['recencia_dias'] > 90)
    ).astype(int)
    
    # Score de riesgo combinado
    df['score_riesgo_churn'] = (
        df['riesgo_inactividad'] * 3 +
        df['riesgo_bajo_engagement'] * 2 +
        df['riesgo_nuevo_inactivo'] * 1
    )
    
    # Categorizar nivel de riesgo
    df['nivel_riesgo'] = pd.cut(
        df['score_riesgo_churn'],
        bins=[-1, 0, 2, 4, 6],
        labels=['Bajo', 'Medio', 'Alto', 'Crítico']
    )
    
    log_mensaje("✓ Features de riesgo creadas", "SUCCESS")
    
    distribucion_riesgo = df['nivel_riesgo'].value_counts()
    for nivel, cantidad in distribucion_riesgo.items():
        porcentaje = (cantidad / len(df)) * 100
        log_mensaje(f"  - Riesgo {nivel}: {cantidad} ({porcentaje:.1f}%)", "INFO")
    
    return df


def generar_todas_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    
    Ejecuta todas las funciones de creación de features en orden.
    
    Args:
        df: DataFrame limpio (capa Silver)
        
    Returns:
        DataFrame con todas las features (capa Gold)
    """
    log_mensaje("="*60, "INFO")
    log_mensaje("INICIANDO FEATURE ENGINEERING COMPLETO", "INFO")
    log_mensaje("="*60, "INFO")
    
    # Paso 1: Métricas RFM básicas
    df = calcular_recencia(df)
    df = calcular_antiguedad(df)
    df = categorizar_monetary(df)
    df = categorizar_frequency(df)
    
    # Paso 2: Score de engagement
    df = calcular_engagement_score(df)
    
    # Paso 3: Features de comportamiento
    df = crear_features_comportamiento(df)
    
    # Paso 4: Features de riesgo
    df = crear_features_riesgo_churn(df)
    
    log_mensaje("="*60, "SUCCESS")
    log_mensaje(f"FEATURE ENGINEERING COMPLETADO: {len(df.columns)} columnas totales", "SUCCESS")
    log_mensaje("="*60, "SUCCESS")
    
    return df


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de feature engineering.
    """
    
    print("\n" + "="*60)
    print("EJEMPLO: Feature Engineering")
    print("="*60)
    
    # DataFrame de ejemplo
    df_ejemplo = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'signup_date': pd.to_datetime(['2023-01-15', '2023-06-01', '2024-01-01']),
        'last_purchase_date': pd.to_datetime(['2025-05-01', '2024-12-15', '2025-06-01']),
        'monthly_spend': [450.50, 1200.00, 890.20],
        'total_shipments': [12, 45, 22]
    })
    
    print("\nDataFrame Original:")
    print(df_ejemplo)
    
    # Generar features
    df_features = generar_todas_features(df_ejemplo)
    
    print("\nDataFrame con Features:")
    print(df_features[['customer_id', 'recencia_dias', 'engagement_score', 
                       'nivel_riesgo', 'segmento_gasto']].head())