"""
Utilidades para limpieza y validación de datos.

Este módulo contiene funciones para el tratamiento de datos sucios:
- Detección y eliminación de duplicados
- Normalización de formatos de fecha
- Tratamiento de valores nulos
- Detección y corrección de outliers

Decisiones de Negocio:
----------------------
Cada función incluye la justificación de negocio para las decisiones tomadas.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configuracion import (
    log_mensaje, 
    ESTRATEGIA_NULOS,
    UMBRAL_GASTO_MINIMO,
    UMBRAL_GASTO_MAXIMO,
    UMBRAL_ENVIOS_MAXIMO
)


def detectar_duplicados(df: pd.DataFrame, 
                       columna_id: str = 'customer_id',
                       accion: str = 'eliminar') -> pd.DataFrame:
    """
    Detecta y maneja registros duplicados.
    
    Decisión de Negocio:
        - Los duplicados completos se eliminan (información redundante)
        - Para duplicados por ID con datos diferentes, se mantiene el registro
          con la última fecha de compra (dato más reciente)
    
    Args:
        df: DataFrame a limpiar
        columna_id: Columna identificadora única
        accion: 'eliminar' o 'marcar'
        
    Returns:
        DataFrame sin duplicados
    """
    log_mensaje("Iniciando detección de duplicados", "INFO")
    
    registros_iniciales = len(df)
    
    # 1. Detectar duplicados completos (todas las columnas iguales)
    duplicados_completos = df.duplicated()
    n_duplicados_completos = duplicados_completos.sum()
    
    if n_duplicados_completos > 0:
        log_mensaje(f"Encontrados {n_duplicados_completos} duplicados completos", "WARNING")
        df = df[~duplicados_completos]
    
    # 2. Detectar duplicados por ID (mismo ID, datos diferentes)
    duplicados_id = df[columna_id].duplicated(keep=False)
    n_duplicados_id = duplicados_id.sum()
    
    if n_duplicados_id > 0:
        log_mensaje(f"Encontrados {n_duplicados_id} registros con {columna_id} duplicado", "WARNING")
        
        # Para IDs duplicados, conservar el registro con fecha más reciente
        if 'last_purchase_date' in df.columns:
            # Convertir fechas a datetime si no lo están
            df['last_purchase_date_temp'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
            
            # Ordenar por ID y fecha (más reciente primero) y mantener el primero
            df = df.sort_values([columna_id, 'last_purchase_date_temp'], 
                               ascending=[True, False])
            df = df.drop_duplicates(subset=columna_id, keep='first')
            df = df.drop(columns=['last_purchase_date_temp'])
            
            log_mensaje(f"✓ Duplicados resueltos: manteniendo registro más reciente por {columna_id}", "SUCCESS")
        else:
            # Si no hay fecha, mantener el primero
            df = df.drop_duplicates(subset=columna_id, keep='first')
    
    registros_finales = len(df)
    registros_eliminados = registros_iniciales - registros_finales
    
    log_mensaje(f"Duplicados eliminados: {registros_eliminados} registros", "SUCCESS")
    
    return df


def normalizar_fechas(df: pd.DataFrame, 
                     columnas_fecha: List[str]) -> pd.DataFrame:
    """
    Normaliza formatos de fecha inconsistentes.
    
    Problema Identificado:
        - Fechas en múltiples formatos: 'DD/MM/YYYY', 'YYYY-MM-DD', 'MM/DD/YYYY'
        - Algunos valores NULL
    
    Solución:
        - Intentar parsear con múltiples formatos
        - Convertir todo a formato estándar YYYY-MM-DD
        - Marcar valores no parseables como NaT
    
    Args:
        df: DataFrame con columnas de fecha
        columnas_fecha: Lista de columnas a normalizar
        
    Returns:
        DataFrame con fechas normalizadas
    """
    log_mensaje(f"Normalizando {len(columnas_fecha)} columnas de fecha", "INFO")
    
    for columna in columnas_fecha:
        if columna not in df.columns:
            log_mensaje(f"Columna {columna} no encontrada, saltando...", "WARNING")
            continue
        
        valores_originales = df[columna].copy()
        
        # Intentar parsear con inferencia automática
        df[columna] = pd.to_datetime(df[columna], errors='coerce', infer_datetime_format=True)
        
        # Contar valores que no se pudieron parsear
        valores_nulos = df[columna].isna().sum()
        nulos_originales = valores_originales.isna().sum()
        nulos_nuevos = valores_nulos - nulos_originales
        
        if nulos_nuevos > 0:
            log_mensaje(f"⚠️  {columna}: {nulos_nuevos} fechas no se pudieron parsear", "WARNING")
        
        log_mensaje(f"✓ {columna}: normalizada correctamente ({len(df) - valores_nulos} valores válidos)", "SUCCESS")
    
    return df


def manejar_valores_nulos(df: pd.DataFrame, 
                         estrategias: Dict[str, str] = ESTRATEGIA_NULOS) -> pd.DataFrame:
    """
    Maneja valores nulos según estrategias definidas por columna.
    
    Decisiones de Negocio por Columna:
    -----------------------------------
    
    1. phone: 'MISSING'
       Justificación: El teléfono faltante puede ser información relevante
                     (cliente no proveyó teléfono). Categoría especial.
    
    2. monthly_spend: 'mediana'
       Justificación: La mediana es robusta a outliers. Representa un cliente
                     "promedio" sin distorsionar distribución.
    
    3. total_shipments: 'mediana'
       Justificación: Similar a monthly_spend, representa comportamiento típico.
    
    4. last_purchase_date: 'ffill' (Forward fill)
       Justificación: Si falta, usar última fecha conocida en datos ordenados.
                     Conservador en modelo de churn.
    
    5. churn_label: 'eliminar'
       Justificación: Sin etiqueta no podemos entrenar. Registros no utilizables.
    
    Args:
        df: DataFrame a procesar
        estrategias: Diccionario {columna: estrategia}
        
    Returns:
        DataFrame con nulos manejados
    """
    log_mensaje("Iniciando tratamiento de valores nulos", "INFO")
    
    # Reporte inicial de nulos
    nulos_iniciales = df.isnull().sum()
    columnas_con_nulos = nulos_iniciales[nulos_iniciales > 0]
    
    if len(columnas_con_nulos) > 0:
        log_mensaje(f"Columnas con valores nulos detectadas: {len(columnas_con_nulos)}", "WARNING")
        for col, cantidad in columnas_con_nulos.items():
            porcentaje = (cantidad / len(df)) * 100
            log_mensaje(f"  - {col}: {cantidad} nulos ({porcentaje:.1f}%)", "INFO")
    
    # Aplicar estrategias
    for columna, estrategia in estrategias.items():
        if columna not in df.columns:
            continue
        
        nulos_en_columna = df[columna].isnull().sum()
        if nulos_en_columna == 0:
            continue
        
        if estrategia == 'MISSING':
            df[columna] = df[columna].fillna('MISSING')
            log_mensaje(f"✓ {columna}: {nulos_en_columna} nulos → 'MISSING'", "SUCCESS")
        
        elif estrategia == 'mediana':
            mediana = df[columna].median()
            df[columna] = df[columna].fillna(mediana)
            log_mensaje(f"✓ {columna}: {nulos_en_columna} nulos → mediana ({mediana:.2f})", "SUCCESS")
        
        elif estrategia == 'media':
            media = df[columna].mean()
            df[columna] = df[columna].fillna(media)
            log_mensaje(f"✓ {columna}: {nulos_en_columna} nulos → media ({media:.2f})", "SUCCESS")
        
        elif estrategia == 'moda':
            moda = df[columna].mode()[0]
            df[columna] = df[columna].fillna(moda)
            log_mensaje(f"✓ {columna}: {nulos_en_columna} nulos → moda ({moda})", "SUCCESS")
        
        elif estrategia == 'ffill':
            df[columna] = df[columna].fillna(method='ffill')
            log_mensaje(f"✓ {columna}: {nulos_en_columna} nulos → forward fill", "SUCCESS")
        
        elif estrategia == 'eliminar':
            registros_antes = len(df)
            df = df.dropna(subset=[columna])
            registros_despues = len(df)
            eliminados = registros_antes - registros_despues
            log_mensaje(f"✓ {columna}: {eliminados} registros eliminados (nulos no permitidos)", "SUCCESS")
    
    # Reporte final
    nulos_finales = df.isnull().sum().sum()
    log_mensaje(f"Tratamiento de nulos completado. Nulos restantes: {nulos_finales}", "SUCCESS")
    
    return df


def detectar_outliers(df: pd.DataFrame, 
                     columnas_numericas: List[str],
                     metodo: str = 'umbrales') -> Tuple[pd.DataFrame, Dict]:
    """
    Detecta y reporta outliers en columnas numéricas.
    
    Métodos:
        - 'umbrales': Usa umbrales de negocio predefinidos
        - 'iqr': Usa método estadístico IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - 'zscore': Usa desviación estándar (|z| > 3)
    
    Decisión de Negocio:
        Preferimos 'umbrales' porque reflejan conocimiento del negocio.
        Por ejemplo, un gasto mensual negativo es claramente un error,
        independiente de la distribución estadística.
    
    Args:
        df: DataFrame a analizar
        columnas_numericas: Columnas a verificar
        metodo: Método de detección
        
    Returns:
        Tuple (DataFrame marcado, Dict con estadísticas de outliers)
    """
    log_mensaje(f"Detectando outliers usando método: {metodo}", "INFO")
    
    outliers_info = {}
    
    for columna in columnas_numericas:
        if columna not in df.columns:
            continue
        
        if metodo == 'umbrales':
            # Umbrales específicos por columna
            if columna == 'monthly_spend':
                outliers = (df[columna] < UMBRAL_GASTO_MINIMO) | (df[columna] > UMBRAL_GASTO_MAXIMO)
            elif columna == 'total_shipments':
                outliers = df[columna] > UMBRAL_ENVIOS_MAXIMO
            else:
                continue
        
        elif metodo == 'iqr':
            Q1 = df[columna].quantile(0.25)
            Q3 = df[columna].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[columna] < (Q1 - 1.5 * IQR)) | (df[columna] > (Q3 + 1.5 * IQR))
        
        elif metodo == 'zscore':
            z_scores = np.abs((df[columna] - df[columna].mean()) / df[columna].std())
            outliers = z_scores > 3
        
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            outliers_info[columna] = {
                'cantidad': n_outliers,
                'porcentaje': (n_outliers / len(df)) * 100,
                'valores': df[outliers][columna].tolist()
            }
            
            log_mensaje(f"⚠️  {columna}: {n_outliers} outliers detectados ({outliers_info[columna]['porcentaje']:.1f}%)", "WARNING")
        
        # Marcar outliers en el DataFrame
        df[f'{columna}_is_outlier'] = outliers
    
    return df, outliers_info


def corregir_outliers(df: pd.DataFrame, 
                     columna: str,
                     metodo: str = 'cap') -> pd.DataFrame:
    """
    Corrige outliers en una columna específica.
    
    Métodos:
        - 'cap': Limita valores a umbrales min/max
        - 'eliminar': Elimina registros con outliers
        - 'nulo': Convierte outliers a NaN para posterior imputación
    
    Decisión de Negocio:
        - Para gastos negativos: convertir a 0 (error de registro)
        - Para gastos extremadamente altos (>15000): cap al umbral (preservar cliente)
        - Para envíos extremos: cap (pueden ser clientes VIP legítimos)
    
    Args:
        df: DataFrame a corregir
        columna: Columna a procesar
        metodo: Método de corrección
        
    Returns:
        DataFrame con outliers corregidos
    """
    if columna not in df.columns:
        return df
    
    if metodo == 'cap':
        if columna == 'monthly_spend':
            # Corregir valores negativos a 0
            n_negativos = (df[columna] < 0).sum()
            if n_negativos > 0:
                df.loc[df[columna] < 0, columna] = 0
                log_mensaje(f"✓ {columna}: {n_negativos} valores negativos → 0", "SUCCESS")
            
            # Cap valores muy altos
            n_altos = (df[columna] > UMBRAL_GASTO_MAXIMO).sum()
            if n_altos > 0:
                df.loc[df[columna] > UMBRAL_GASTO_MAXIMO, columna] = UMBRAL_GASTO_MAXIMO
                log_mensaje(f"✓ {columna}: {n_altos} valores > {UMBRAL_GASTO_MAXIMO} → {UMBRAL_GASTO_MAXIMO}", "SUCCESS")
        
        elif columna == 'total_shipments':
            n_altos = (df[columna] > UMBRAL_ENVIOS_MAXIMO).sum()
            if n_altos > 0:
                df.loc[df[columna] > UMBRAL_ENVIOS_MAXIMO, columna] = UMBRAL_ENVIOS_MAXIMO
                log_mensaje(f"✓ {columna}: {n_altos} valores > {UMBRAL_ENVIOS_MAXIMO} → {UMBRAL_ENVIOS_MAXIMO}", "SUCCESS")
    
    return df


def validar_calidad_datos(df: pd.DataFrame) -> Dict:
    """
    Genera un reporte de calidad de datos.
    
    Verifica:
        - Valores nulos por columna
        - Duplicados
        - Tipos de datos
        - Rangos de valores numéricos
        - Cardinalidad de categóricas
    
    Returns:
        Diccionario con métricas de calidad
    """
    reporte = {
        'registros_totales': len(df),
        'columnas_totales': len(df.columns),
        'nulos': df.isnull().sum().to_dict(),
        'duplicados': df.duplicated().sum(),
        'tipos_datos': df.dtypes.astype(str).to_dict(),
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Estadísticas de columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    reporte['estadisticas_numericas'] = df[columnas_numericas].describe().to_dict()
    
    # Cardinalidad de columnas categóricas
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    reporte['cardinalidad_categoricas'] = {col: df[col].nunique() for col in columnas_categoricas}
    
    return reporte


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de limpieza.
    """
    
    print("\n" + "="*60)
    print("EJEMPLO: Limpieza de Datos")
    print("="*60)
    
    # DataFrame de ejemplo con problemas
    df_ejemplo = pd.DataFrame({
        'customer_id': ['C001', 'C001', 'C002', 'C003', 'C004'],
        'monthly_spend': [450.50, 450.50, -50, 99999, 1200],
        'total_shipments': [12, 12, 15, 1000, 45],
        'last_purchase_date': ['2025-01-15', '15/01/2025', None, '2025-03-20', '20/03/2025'],
        'churn_label': [0, 0, 1, None, 0]
    })
    
    print("\nDataFrame Original:")
    print(df_ejemplo)
    
    # Limpiar
    df_limpio = detectar_duplicados(df_ejemplo)
    df_limpio = normalizar_fechas(df_limpio, ['last_purchase_date'])
    df_limpio = manejar_valores_nulos(df_limpio)
    df_limpio = corregir_outliers(df_limpio, 'monthly_spend', metodo='cap')
    
    print("\nDataFrame Limpio:")
    print(df_limpio)