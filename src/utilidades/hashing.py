"""
Utilidades para el hashing de datos sensibles.

Este módulo implementa funciones para anonimizar información personal identificable (PII)
usando hashing con SHA-256. Esto garantiza el cumplimiento de normativas de privacidad
mientras mantiene la trazabilidad de los datos.

Justificación de Negocio:
-------------------------
- Protección de datos personales según GDPR/CCPA
- Permite análisis sin exponer información sensible
- Mantiene unicidad de registros para joins
- Reproducible con el mismo salt
"""

import hashlib
import pandas as pd
from typing import Union, List
import sys
from pathlib import Path

# Agregar directorio padre al path para importar configuración
sys.path.append(str(Path(__file__).parent.parent))
from configuracion import SALT_HASHING, COLUMNAS_SENSIBLES, log_mensaje


def hashear_valor(valor: Union[str, None], salt: str = SALT_HASHING) -> str:
    """
    Hashea un valor individual usando SHA-256 con salt.
    
    Args:
        valor: Valor a hashear (puede ser None)
        salt: Salt para agregar seguridad al hash
        
    Returns:
        Hash hexadecimal del valor o 'NULL' si el valor es None
        
    Ejemplo:
        >>> hashear_valor("juan.perez@email.com")
        'a1b2c3d4e5f6...'
    """
    if valor is None or pd.isna(valor) or str(valor).strip() == '':
        return 'NULL'
    
    # Convertir a string, agregar salt y hashear
    valor_con_salt = f"{salt}{str(valor)}"
    hash_objeto = hashlib.sha256(valor_con_salt.encode('utf-8'))
    
    return hash_objeto.hexdigest()


def hashear_columna(serie: pd.Series, nombre_columna: str, salt: str = SALT_HASHING) -> pd.Series:
    """
    Hashea una columna completa de un DataFrame.
    
    Args:
        serie: Serie de pandas a hashear
        nombre_columna: Nombre de la columna (para logging)
        salt: Salt para hashing
        
    Returns:
        Serie con valores hasheados
        
    Decisión de Negocio:
        - Los valores nulos se mantienen como 'NULL' para distinguirlos
        - El hash es irreversible pero reproducible
    """
    log_mensaje(f"Hasheando columna: {nombre_columna}", "INFO")
    
    valores_unicos_original = serie.nunique()
    
    # Aplicar hashing a cada valor
    serie_hasheada = serie.apply(lambda x: hashear_valor(x, salt))
    
    valores_unicos_hash = serie_hasheada.nunique()
    
    # Verificación de integridad
    if valores_unicos_original == valores_unicos_hash:
        log_mensaje(f"✓ Columna {nombre_columna}: {valores_unicos_original} valores únicos preservados", "SUCCESS")
    else:
        log_mensaje(f"⚠️  Posible colisión en {nombre_columna}: {valores_unicos_original} → {valores_unicos_hash}", "WARNING")
    
    return serie_hasheada


def hashear_datos_sensibles(df: pd.DataFrame, 
                            columnas: List[str] = COLUMNAS_SENSIBLES,
                            sufijo: str = '_hash') -> pd.DataFrame:
    """
    Hashea múltiples columnas sensibles en un DataFrame.
    
    Args:
        df: DataFrame con datos originales
        columnas: Lista de columnas a hashear
        sufijo: Sufijo para las nuevas columnas hasheadas
        
    Returns:
        DataFrame con columnas hasheadas y originales eliminadas
        
    Proceso:
        1. Crea nuevas columnas con sufijo '_hash'
        2. Hashea los valores
        3. Elimina las columnas originales
        4. Renombra las columnas hasheadas
        
    Ejemplo:
        >>> df_anonimizado = hashear_datos_sensibles(df, ['email', 'phone'])
    """
    df_copia = df.copy()
    
    log_mensaje(f"Iniciando proceso de hashing de {len(columnas)} columnas sensibles", "INFO")
    
    # Verificar que todas las columnas existen
    columnas_faltantes = [col for col in columnas if col not in df_copia.columns]
    if columnas_faltantes:
        log_mensaje(f"Columnas no encontradas: {columnas_faltantes}", "WARNING")
        columnas = [col for col in columnas if col in df_copia.columns]
    
    # Hashear cada columna
    for columna in columnas:
        # Crear nombre de columna temporal
        nombre_hash = f"{columna}{sufijo}"
        
        # Hashear la columna
        df_copia[nombre_hash] = hashear_columna(df_copia[columna], columna)
    
    # Eliminar columnas originales
    df_copia = df_copia.drop(columns=columnas)
    
    # Renombrar columnas hasheadas (quitar sufijo si se desea)
    # Mantenemos el sufijo para claridad de que son valores hasheados
    
    log_mensaje(f"Proceso de hashing completado. {len(columnas)} columnas protegidas", "SUCCESS")
    
    return df_copia


def crear_mapeo_hash(df: pd.DataFrame, 
                     columna: str,
                     guardar_ruta: Union[str, Path] = None) -> pd.DataFrame:
    """
    Crea un mapeo entre valores originales y hasheados.
    
    ADVERTENCIA: Esta función solo debe usarse en desarrollo/testing.
    En producción, NO se deben guardar mapeos de valores sensibles.
    
    Args:
        df: DataFrame original
        columna: Columna a mapear
        guardar_ruta: Ruta donde guardar el mapeo (opcional)
        
    Returns:
        DataFrame con mapeo [valor_original, valor_hash]
    """
    log_mensaje("⚠️  ADVERTENCIA: Creando mapeo de hashing (solo para desarrollo)", "WARNING")
    
    if columna not in df.columns:
        raise ValueError(f"Columna {columna} no existe en el DataFrame")
    
    # Crear mapeo de valores únicos
    valores_unicos = df[columna].dropna().unique()
    mapeo = pd.DataFrame({
        'valor_original': valores_unicos,
        'valor_hash': [hashear_valor(v) for v in valores_unicos]
    })
    
    if guardar_ruta:
        mapeo.to_csv(guardar_ruta, index=False)
        log_mensaje(f"Mapeo guardado en: {guardar_ruta}", "INFO")
    
    return mapeo


def verificar_reproducibilidad_hash(valor: str, 
                                    hash_esperado: str,
                                    salt: str = SALT_HASHING) -> bool:
    """
    Verifica que el hashing sea reproducible.
    
    Útil para testing y validación de integridad.
    
    Args:
        valor: Valor original
        hash_esperado: Hash que debería producirse
        salt: Salt utilizado
        
    Returns:
        True si el hash coincide
    """
    hash_generado = hashear_valor(valor, salt)
    return hash_generado == hash_esperado


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de hashing.
    """
    
    # Crear DataFrame de ejemplo
    df_ejemplo = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'full_name': ['Juan Perez', 'Maria Garcia', 'Carlos Rodriguez'],
        'email': ['jperez@email.com', 'm.garcia@provider.net', 'c.rod@work.com'],
        'phone': ['555-0101', '555-0102', None],
        'monthly_spend': [450.50, 1200.00, 890.20]
    })
    
    print("\n" + "="*60)
    print("EJEMPLO: Hashing de Datos Sensibles")
    print("="*60)
    
    print("\nDataFrame Original:")
    print(df_ejemplo)
    
    # Hashear datos sensibles
    df_hasheado = hashear_datos_sensibles(
        df_ejemplo, 
        columnas=['full_name', 'email', 'phone']
    )
    
    print("\nDataFrame con Datos Hasheados:")
    print(df_hasheado)
    
    # Verificar reproducibilidad
    print("\n" + "="*60)
    print("Verificación de Reproducibilidad:")
    print("="*60)
    
    email_original = 'jperez@email.com'
    hash_1 = hashear_valor(email_original)
    hash_2 = hashear_valor(email_original)
    
    print(f"\nEmail original: {email_original}")
    print(f"Hash 1: {hash_1}")
    print(f"Hash 2: {hash_2}")
    print(f"¿Son iguales?: {hash_1 == hash_2}")