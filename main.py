#!/usr/bin/env python3
"""
Pipeline Principal - Proyecto de Predicción de Churn

Este script ejecuta el pipeline completo end-to-end:
1. Bronze: Carga de datos raw
2. Silver: Limpieza y gobierno de datos  
3. Gold: Feature engineering
4. Modelado: Entrenamiento y evaluación

Uso:
    python main.py                    # Ejecutar pipeline completo
    python main.py --solo-limpieza    # Solo hasta capa Silver
    python main.py --solo-features    # Solo hasta capa Gold
    python main.py --solo-modelo      # Solo entrenamiento
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from configuracion import log_mensaje

# Importar módulos del pipeline
from ingestion.cargar_bronze import main as cargar_bronze
from transformacion.bronze_a_silver import main as bronze_a_silver
from transformacion.silver_a_gold import main as silver_a_gold
from modelado.entrenar_modelo import main as entrenar_modelo


def ejecutar_pipeline_completo():
    """
    Ejecuta el pipeline completo de principio a fin.
    """
    inicio = datetime.now()
    
    log_mensaje("\n" + "="*80, "INFO")
    log_mensaje("🚀 INICIANDO PIPELINE COMPLETO - PREDICCIÓN DE CHURN", "INFO")
    log_mensaje("="*80, "INFO")
    log_mensaje(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    
    try:
        # ====================================================================
        # PASO 1: CAPA BRONZE
        # ====================================================================
        log_mensaje("\n" + "▶️  PASO 1/4: Cargando datos a capa BRONZE...", "INFO")
        df_bronze = cargar_bronze()
        log_mensaje("✅ Capa BRONZE completada", "SUCCESS")
        
        # ====================================================================
        # PASO 2: CAPA SILVER
        # ====================================================================
        log_mensaje("\n" + "▶️  PASO 2/4: Transformando BRONZE → SILVER...", "INFO")
        df_silver = bronze_a_silver()
        log_mensaje("✅ Capa SILVER completada", "SUCCESS")
        
        # ====================================================================
        # PASO 3: CAPA GOLD
        # ====================================================================
        log_mensaje("\n" + "▶️  PASO 3/4: Transformando SILVER → GOLD...", "INFO")
        df_gold = silver_a_gold()
        log_mensaje("✅ Capa GOLD completada", "SUCCESS")
        
        # ====================================================================
        # PASO 4: ENTRENAMIENTO
        # ====================================================================
        log_mensaje("\n" + "▶️  PASO 4/4: Entrenando modelo de Churn...", "INFO")
        modelo, metricas = entrenar_modelo()
        log_mensaje("✅ Entrenamiento completado", "SUCCESS")
        
        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        fin = datetime.now()
        duracion = fin - inicio
        
        log_mensaje("\n" + "="*80, "SUCCESS")
        log_mensaje("✅ PIPELINE COMPLETADO EXITOSAMENTE", "SUCCESS")
        log_mensaje("="*80, "SUCCESS")
        
        log_mensaje(f"\nFin: {fin.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
        log_mensaje(f"Duración total: {duracion}", "INFO")
        
        log_mensaje("\n📊 Resultados Finales:", "INFO")
        log_mensaje(f"  - Registros procesados: {len(df_gold)}", "INFO")
        log_mensaje(f"  - Features generadas: {len(df_gold.columns) - 1}", "INFO")
        log_mensaje(f"  - Modelo ganador: {metricas.get('modelo_ganador', 'N/A')}", "INFO")
        log_mensaje(f"  - Accuracy: {metricas.get('test_accuracy', 0):.4f}", "INFO")
        log_mensaje(f"  - F1-Score: {metricas.get('test_f1', 0):.4f}", "INFO")
        log_mensaje(f"  - ROC-AUC: {metricas.get('test_roc_auc', 0):.4f}", "INFO")
        log_mensaje(f"  - CV F1-Score: {metricas.get('cv_f1_mean', 0):.4f}", "INFO")
        
        log_mensaje("\n📁 Archivos Generados:", "INFO")
        log_mensaje("  - datos/silver/clientes_limpios.csv", "INFO")
        log_mensaje("  - datos/gold/clientes_modelado.csv", "INFO")
        log_mensaje("  - modelos/modelo_churn.pkl", "INFO")
        log_mensaje("  - resultados/metricas/metricas_modelo.json", "INFO")
        log_mensaje("  - resultados/visualizaciones/evaluacion_modelo.png", "INFO")
        
        return True
        
    except Exception as e:
        log_mensaje(f"\n❌ ERROR EN PIPELINE: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def ejecutar_solo_limpieza():
    """
    Ejecuta solo hasta la capa Silver (limpieza de datos).
    """
    log_mensaje("\n🔧 Ejecutando solo pipeline de limpieza (Bronze → Silver)", "INFO")
    
    try:
        df_bronze = cargar_bronze()
        df_silver = bronze_a_silver()
        
        log_mensaje("\n✅ Pipeline de limpieza completado", "SUCCESS")
        log_mensaje("📁 Datos limpios disponibles en: datos/silver/clientes_limpios.csv", "INFO")
        
        return True
        
    except Exception as e:
        log_mensaje(f"\n❌ ERROR: {str(e)}", "ERROR")
        return False


def ejecutar_solo_features():
    """
    Ejecuta hasta la capa Gold (incluye limpieza + feature engineering).
    """
    log_mensaje("\n🔧 Ejecutando pipeline hasta features (Bronze → Silver → Gold)", "INFO")
    
    try:
        df_bronze = cargar_bronze()
        df_silver = bronze_a_silver()
        df_gold = silver_a_gold()
        
        log_mensaje("\n✅ Pipeline de features completado", "SUCCESS")
        log_mensaje("📁 Dataset analítico disponible en: datos/gold/clientes_modelado.csv", "INFO")
        
        return True
        
    except Exception as e:
        log_mensaje(f"\n❌ ERROR: {str(e)}", "ERROR")
        return False


def ejecutar_solo_modelo():
    """
    Ejecuta solo el entrenamiento (requiere que Gold ya exista).
    """
    log_mensaje("\n🔧 Ejecutando solo entrenamiento de modelo", "INFO")
    
    try:
        modelo, metricas = entrenar_modelo()
        
        log_mensaje("\n✅ Entrenamiento completado", "SUCCESS")
        # Ajuste para estructura de métricas multi-modelo
        f1_score = metricas.get('test_f1', 0) if 'test_f1' in metricas else metricas['test']['f1_churn']
        log_mensaje(f"📊 F1-Score (Test): {f1_score:.4f}", "INFO")
        log_mensaje("📁 Modelo guardado en: modelos/modelo_churn.pkl", "INFO")
        
        return True
        
    except Exception as e:
        log_mensaje(f"\n❌ ERROR: {str(e)}", "ERROR")
        return False


def main():
    """
    Función principal con argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Pipeline de Predicción de Churn - Arquitectura Medallion'
    )
    
    parser.add_argument(
        '--solo-limpieza',
        action='store_true',
        help='Ejecutar solo limpieza de datos (Bronze → Silver)'
    )
    
    parser.add_argument(
        '--solo-features',
        action='store_true',
        help='Ejecutar hasta feature engineering (Bronze → Silver → Gold)'
    )
    
    parser.add_argument(
        '--solo-modelo',
        action='store_true',
        help='Ejecutar solo entrenamiento de modelo (requiere Gold existente)'
    )
    
    args = parser.parse_args()
    
    # Ejecutar según argumentos
    if args.solo_limpieza:
        exito = ejecutar_solo_limpieza()
    elif args.solo_features:
        exito = ejecutar_solo_features()
    elif args.solo_modelo:
        exito = ejecutar_solo_modelo()
    else:
        # Por defecto: pipeline completo
        exito = ejecutar_pipeline_completo()
    
    # Retornar código de salida
    sys.exit(0 if exito else 1)


if __name__ == "__main__":
    main()