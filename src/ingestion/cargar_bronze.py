"""
Script de Ingesta de Datos - Capa BRONZE

Este script carga los datos raw sin transformaciones.
Objetivo: Preservar los datos originales tal como fueron recibidos.

Arquitectura Medallion:
-----------------------
BRONZE: Datos crudos, sin transformar
        - Preserva el estado original
        - Permite auditoría y trazabilidad
        - Base para reproducibilidad
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from configuracion import (
    RUTA_BRONZE,
    ARCHIVO_BRONZE,
    log_mensaje
)


def cargar_datos_bronze(mostrar_preview: bool = True) -> pd.DataFrame:
    """
    Carga datos raw a la capa Bronze.
    
    Proceso:
        1. Lee el CSV tal como está
        2. No aplica transformaciones
        3. Registra metadatos de carga
    
    Args:
        mostrar_preview: Si mostrar vista previa de los datos
        
    Returns:
        DataFrame con datos raw
    """
    log_mensaje("="*60, "INFO")
    log_mensaje("CAPA BRONZE - CARGA DE DATOS RAW", "INFO")
    log_mensaje("="*60, "INFO")
    
    ruta_archivo = RUTA_BRONZE / ARCHIVO_BRONZE
    
    if not ruta_archivo.exists():
        log_mensaje(f"Archivo no encontrado: {ruta_archivo}", "ERROR")
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta_archivo}")
    
    log_mensaje(f"Cargando datos desde: {ruta_archivo}", "INFO")
    
    # Cargar CSV sin transformaciones
    # Nota: Usamos dtype=str para preservar exactamente como vienen los datos
    df = pd.read_csv(ruta_archivo, dtype=str)

    # Detectar si el CSV está malformado (todo en una columna con comillas)
    condicion_1_columna = len(df.columns) == 1
    
    if condicion_1_columna:
        log_mensaje("⚠️  Detectado posible CSV malformado (1 columna)", "WARNING")
        
        # Recargar ignorando comillas (quoting=3 es QUOTE_NONE)
        df_temp = pd.read_csv(ruta_archivo, dtype=str, quoting=3)
        
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
    
    # Metadatos de carga
    metadatos = {
        'timestamp_carga': datetime.now(),
        'registros': len(df),
        'columnas': len(df.columns),
        'archivo_origen': ARCHIVO_BRONZE,
        'ruta_completa': str(ruta_archivo)
    }
    
    log_mensaje(f"✓ Datos cargados exitosamente", "SUCCESS")
    log_mensaje(f"  - Registros: {metadatos['registros']}", "INFO")
    log_mensaje(f"  - Columnas: {metadatos['columnas']}", "INFO")
    log_mensaje(f"  - Timestamp: {metadatos['timestamp_carga']}", "INFO")
    
    if mostrar_preview:
        log_mensaje("\nVista previa de datos raw:", "INFO")
        log_mensaje(f"\nColumnas: {list(df.columns)}", "INFO")
        log_mensaje(f"\nPrimeros 3 registros:", "INFO")
        print(df.head(3))
        
        log_mensaje(f"\nÚltimos 3 registros:", "INFO")
        print(df.tail(3))
    
    # Análisis inicial de calidad
    log_mensaje("\nAnálisis Inicial de Calidad:", "INFO")
    
    # Contar valores únicos por columna
    for columna in df.columns:
        valores_unicos = df[columna].nunique()
        valores_nulos = df[columna].isna().sum() + (df[columna] == 'NULL').sum()
        log_mensaje(f"  - {columna}: {valores_unicos} únicos, {valores_nulos} nulos", "INFO")
    
    # Detectar duplicados
    duplicados_completos = df.duplicated().sum()
    if duplicados_completos > 0:
        log_mensaje(f"⚠️  Duplicados completos detectados: {duplicados_completos}", "WARNING")
    
    # Detectar duplicados por customer_id
    if 'customer_id' in df.columns:
        duplicados_id = df['customer_id'].duplicated().sum()
        if duplicados_id > 0:
            log_mensaje(f"⚠️  IDs duplicados detectados: {duplicados_id}", "WARNING")
    
    log_mensaje("="*60, "SUCCESS")
    log_mensaje("CARGA BRONZE COMPLETADA", "SUCCESS")
    log_mensaje("="*60, "SUCCESS")
    
    return df


def main():
    """
    Función principal para ejecutar la carga de datos Bronze.
    """
    try:
        df_bronze = cargar_datos_bronze(mostrar_preview=True)
        
        # Los datos permanecen en la carpeta bronze sin modificar
        # El siguiente paso será procesarlos hacia Silver
        
        log_mensaje("\n✅ Datos listos en capa BRONZE", "SUCCESS")
        log_mensaje("➡️  Siguiente paso: Ejecutar bronze_a_silver.py", "INFO")
        
        return df_bronze
        
    except Exception as e:
        log_mensaje(f"Error en carga Bronze: {str(e)}", "ERROR")
        raise


if __name__ == "__main__":
    df = main()