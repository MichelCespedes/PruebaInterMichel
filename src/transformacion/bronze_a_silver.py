"""
Script de Transformación - BRONZE → SILVER

Este script limpia y gobierna los datos raw.
Objetivo: Datos limpios, consistentes y conformes a estándares.

Procesos aplicados:
-------------------
1. Eliminación de duplicados
2. Normalización de formatos de fecha
3. Conversión de tipos de datos
4. Tratamiento de valores nulos
5. Detección y corrección de outliers
6. Hashing de datos sensibles (PII)

Decisiones de Negocio:
---------------------
Cada transformación está justificada por requerimientos de calidad
y cumplimiento normativo (GDPR, privacidad de datos).
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from configuracion import (
    RUTA_BRONZE,
    RUTA_SILVER,
    ARCHIVO_BRONZE,
    ARCHIVO_SILVER,
    COLUMNAS_SENSIBLES,
    log_mensaje
)

from utilidades.limpieza import (
    detectar_duplicados,
    normalizar_fechas,
    manejar_valores_nulos,
    detectar_outliers,
    corregir_outliers,
    validar_calidad_datos
)

from utilidades.hashing import hashear_datos_sensibles


def transformar_bronze_a_silver() -> pd.DataFrame:
    """
    Pipeline completo de transformación Bronze → Silver.
    
    Returns:
        DataFrame limpio y governado
    """
    log_mensaje("="*70, "INFO")
    log_mensaje("TRANSFORMACIÓN: BRONZE → SILVER", "INFO")
    log_mensaje("="*70, "INFO")
    
    # ========================================================================
    # PASO 1: CARGAR DATOS BRONZE
    # ========================================================================
    
    log_mensaje("\n[1/8] Cargando datos de capa Bronze...", "INFO")
    ruta_bronze = RUTA_BRONZE / ARCHIVO_BRONZE
    
    # Cargar como string para preservar formatos originales
    # Cargar como string para preservar formatos originales
    df = pd.read_csv(ruta_bronze, dtype=str)

    # FIX: Detectar si el CSV está malformado (todo en una columna con comillas)
    # Esto pasa cuando el archivo tiene comillas al inicio y final de cada línea
    # Ajustamos la condición para ser más flexibles
    condicion_1_columna = len(df.columns) == 1
    
    if condicion_1_columna:
        log_mensaje("⚠️  Detectado posible CSV malformado (1 columna)", "WARNING")
        
        # Recargar ignorando comillas (quoting=3 es QUOTE_NONE)
        df_temp = pd.read_csv(ruta_bronze, dtype=str, quoting=3)
        
        # Si al recargar tenemos más columnas, asumimos que es el fix correcto
        if len(df_temp.columns) > 1:
            log_mensaje("⚠️  Aplicando corrección de comillas...", "WARNING")
            df = df_temp
            
            # Limpiar comillas en nombres de columnas
            df.columns = df.columns.str.replace('"', '')
            
            # Limpiar comillas en los datos
            for col in df.columns:
                df[col] = df[col].str.replace('"', '')
                
            log_mensaje("✓ Corrección de formato aplicada exitosamente", "SUCCESS")
    
    log_mensaje(f"✓ Datos cargados: {len(df)} registros, {len(df.columns)} columnas", "SUCCESS")
    
    # ========================================================================
    # PASO 2: ELIMINAR DUPLICADOS
    # ========================================================================
    
    log_mensaje("\n[2/8] Detectando y eliminando duplicados...", "INFO")
    registros_antes = len(df)
    df = detectar_duplicados(df, columna_id='customer_id', accion='eliminar')
    registros_despues = len(df)
    
    if registros_antes != registros_despues:
        log_mensaje(f"✓ Eliminados {registros_antes - registros_despues} duplicados", "SUCCESS")
    else:
        log_mensaje("✓ No se encontraron duplicados", "SUCCESS")
    
    # ========================================================================
    # PASO 3: NORMALIZAR FORMATOS DE FECHA
    # ========================================================================
    
    log_mensaje("\n[3/8] Normalizando formatos de fecha...", "INFO")
    
    columnas_fecha = ['signup_date', 'last_purchase_date']
    df = normalizar_fechas(df, columnas_fecha)
    
    log_mensaje("✓ Fechas normalizadas a formato estándar", "SUCCESS")
    
    # ========================================================================
    # PASO 4: CONVERTIR TIPOS DE DATOS
    # ========================================================================
    
    log_mensaje("\n[4/8] Convirtiendo tipos de datos...", "INFO")
    
    # Convertir 'NULL' strings a np.nan
    df = df.replace('NULL', np.nan)
    df = df.replace('', np.nan)
    df = df.replace('N/A', np.nan)
    
    # Convertir columnas numéricas
    columnas_numericas = ['monthly_spend', 'total_shipments', 'churn_label']
    
    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors='coerce')
    
    log_mensaje("✓ Tipos de datos convertidos", "SUCCESS")
    
    # ========================================================================
    # PASO 5: CORREGIR OUTLIERS
    # ========================================================================
    
    log_mensaje("\n[5/8] Detectando y corrigiendo outliers...", "INFO")
    
    # Detectar outliers en gasto y envíos
    df, outliers_info = detectar_outliers(
        df, 
        columnas_numericas=['monthly_spend', 'total_shipments'],
        metodo='umbrales'
    )
    
    # Corregir outliers
    df = corregir_outliers(df, 'monthly_spend', metodo='cap')
    df = corregir_outliers(df, 'total_shipments', metodo='cap')
    
    # Eliminar columnas temporales de outliers
    columnas_outlier = [col for col in df.columns if col.endswith('_is_outlier')]
    df = df.drop(columns=columnas_outlier)
    
    log_mensaje("✓ Outliers detectados y corregidos", "SUCCESS")
    
    # ========================================================================
    # PASO 6: MANEJAR VALORES NULOS
    # ========================================================================
    
    log_mensaje("\n[6/8] Tratando valores nulos...", "INFO")
    
    registros_antes_nulos = len(df)
    df = manejar_valores_nulos(df)
    registros_despues_nulos = len(df)
    
    if registros_antes_nulos != registros_despues_nulos:
        log_mensaje(f"⚠️  {registros_antes_nulos - registros_despues_nulos} registros eliminados por nulos en churn_label", "WARNING")
    
    log_mensaje("✓ Valores nulos manejados", "SUCCESS")
    
    # ========================================================================
    # PASO 7: HASHEAR DATOS SENSIBLES
    # ========================================================================
    
    log_mensaje("\n[7/8] Hasheando datos sensibles (PII)...", "INFO")
    log_mensaje(f"Columnas a hashear: {COLUMNAS_SENSIBLES}", "INFO")
    
    df = hashear_datos_sensibles(df, columnas=COLUMNAS_SENSIBLES, sufijo='_hash')
    
    log_mensaje("✓ Datos sensibles protegidos mediante hashing", "SUCCESS")
    
    # ========================================================================
    # PASO 8: VALIDAR CALIDAD FINAL
    # ========================================================================
    
    log_mensaje("\n[8/8] Validando calidad de datos Silver...", "INFO")
    
    reporte_calidad = validar_calidad_datos(df)
    
    log_mensaje(f"✓ Registros finales: {reporte_calidad['registros_totales']}", "SUCCESS")
    log_mensaje(f"✓ Columnas finales: {reporte_calidad['columnas_totales']}", "SUCCESS")
    log_mensaje(f"✓ Duplicados restantes: {reporte_calidad['duplicados']}", "SUCCESS")
    log_mensaje(f"✓ Memoria utilizada: {reporte_calidad['memoria_mb']:.2f} MB", "SUCCESS")
    
    # Verificar que no haya nulos críticos
    nulos_totales = sum(reporte_calidad['nulos'].values())
    if nulos_totales > 0:
        log_mensaje(f"⚠️  Nulos restantes: {nulos_totales} (en columnas no críticas)", "WARNING")
        for col, cantidad in reporte_calidad['nulos'].items():
            if cantidad > 0:
                log_mensaje(f"    - {col}: {cantidad}", "INFO")
    
    # ========================================================================
    # GUARDAR DATOS SILVER
    # ========================================================================
    
    log_mensaje("\nGuardando datos en capa SILVER...", "INFO")
    
    ruta_silver = RUTA_SILVER / ARCHIVO_SILVER
    df.to_csv(ruta_silver, index=False)
    
    log_mensaje(f"✓ Datos Silver guardados en: {ruta_silver}", "SUCCESS")
    
    log_mensaje("\n" + "="*70, "SUCCESS")
    log_mensaje("TRANSFORMACIÓN BRONZE → SILVER COMPLETADA", "SUCCESS")
    log_mensaje("="*70, "SUCCESS")
    
    return df


def main():
    """
    Función principal para ejecutar transformación Bronze → Silver.
    """
    try:
        df_silver = transformar_bronze_a_silver()
        
        log_mensaje("\n✅ Datos limpios disponibles en capa SILVER", "SUCCESS")
        log_mensaje("➡️  Siguiente paso: Ejecutar silver_a_gold.py para Feature Engineering", "INFO")
        
        # Mostrar preview
        log_mensaje("\nVista previa de datos Silver:", "INFO")
        print(df_silver.head(3))
        
        return df_silver
        
    except Exception as e:
        log_mensaje(f"Error en transformación Silver: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    df = main()