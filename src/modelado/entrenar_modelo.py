"""
Script de Entrenamiento de Múltiples Modelos - Churn Prediction

Entrena y compara:
1. Random Forest (Baseline complejo)
2. Logistic Regression (Baseline simple)
3. XGBoost (SOTA - State of the Art)

Selecciona el mejor modelo basado en F1-Score (balance precision/recall).
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

# Silenciar advertencias de métricas cuando no hay predicciones positivas (común en clases minoritarias)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(str(Path(__file__).parent.parent))
from configuracion import (
    RUTA_GOLD,
    RUTA_MODELOS,
    RUTA_METRICAS,
    RUTA_VISUALIZACIONES,
    ARCHIVO_GOLD,
    ARCHIVO_MODELO,
    TEST_SIZE,
    RANDOM_STATE,
    MODELOS_CONFIG,  # Nueva configuración con múltiples modelos
    log_mensaje
)


def cargar_datos_gold() -> pd.DataFrame:
    """Carga el dataset Gold para entrenamiento."""
    log_mensaje("Cargando datos de capa Gold...", "INFO")
    
    ruta_gold = RUTA_GOLD / ARCHIVO_GOLD
    
    if not ruta_gold.exists():
        log_mensaje(f"Archivo Gold no encontrado: {ruta_gold}", "ERROR")
        raise FileNotFoundError(f"No se encuentra: {ruta_gold}")
    
    df = pd.read_csv(ruta_gold)
    log_mensaje(f"✓ Datos cargados: {len(df)} registros, {len(df.columns)} columnas", "SUCCESS")
    return df


def preparar_datos_entrenamiento(df: pd.DataFrame):
    """Prepara datos: split train/test y manejo de target."""
    log_mensaje("Preparando datos para entrenamiento...", "INFO")
    
    if 'churn_label' not in df.columns:
        raise ValueError("Variable objetivo 'churn_label' no encontrada")
    
    X = df.drop(columns=['churn_label'])
    y = df['churn_label']
    
    feature_names = X.columns.tolist()
    
    log_mensaje(f"Dividiendo datos: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test", "INFO")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names


def entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test):
    """
    Entrena múltiples modelos y selecciona el mejor.
    """
    resultados = {}
    mejores_metricas = None
    mejor_modelo = None
    mejor_nombre = ""
    mejor_score = -1
    
    log_mensaje("\n" + "="*70, "INFO")
    log_mensaje("INICIANDO ENTRENAMIENTO MULTI-MODELO", "INFO")
    log_mensaje("="*70, "INFO")
    
    for nombre_modelo, params in MODELOS_CONFIG.items():
        log_mensaje(f"\n🤖 Entrenando: {nombre_modelo}...", "INFO")
        
        try:
            # Seleccionar algoritmo
            if nombre_modelo == 'RandomForest':
                clf = RandomForestClassifier(**params)
            elif nombre_modelo == 'LogisticRegression':
                # LR necesita escalado de datos
                clf = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(**params))
                ])
            elif nombre_modelo == 'XGBoost':
                clf = XGBClassifier(**params)
            
            # Entrenar
            clf.fit(X_train, y_train)
            
            # Validar con Cross-Validation (5 folds) en Train
            # Usamos F1 como métrica principal para selección
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
            mean_cv_score = cv_scores.mean()
            
            log_mensaje(f"   ✓ CV F1-Score: {mean_cv_score:.4f} (+/- {cv_scores.std()*2:.4f})", "SUCCESS")
            
            # Evaluar en Test
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Detectar clave de clase positiva
            clave_pos = '1'
            if '1.0' in report: clave_pos = '1.0'
            elif 1 in report: clave_pos = 1
            
            f1_test = report[clave_pos]['f1-score']
            
            log_mensaje(f"   ✓ Test F1-Score: {f1_test:.4f}", "INFO")
            log_mensaje(f"   ✓ Test ROC-AUC: {roc_auc:.4f}", "INFO")
            
            # Guardar resultados
            resultados[nombre_modelo] = {
                'cv_f1_mean': mean_cv_score,
                'test_f1': f1_test,
                'test_roc_auc': roc_auc,
                'modelo': clf
            }
            
            # Verificar si es el mejor modelo (basado en CV para ser robustos)
            if mean_cv_score > mejor_score:
                mejor_score = mean_cv_score
                mejor_modelo = clf
                mejor_nombre = nombre_modelo
                mejores_metricas = {
                    'timestamp': datetime.now().isoformat(),
                    'modelo_ganador': nombre_modelo,
                    'test_accuracy': report['accuracy'],
                    'test_f1': f1_test,
                    'test_precision': report[clave_pos]['precision'],
                    'test_recall': report[clave_pos]['recall'],
                    'test_roc_auc': roc_auc,
                    'cv_f1_mean': mean_cv_score,
                    'cv_f1_std': cv_scores.std(),
                    'comparativa': {k: v['cv_f1_mean'] for k, v in resultados.items() if 'cv_f1_mean' in v}
                }
                
        except Exception as e:
            log_mensaje(f"❌ Error entrenando {nombre_modelo}: {str(e)}", "ERROR")
            continue

    log_mensaje("\n" + "="*70, "SUCCESS")
    log_mensaje(f"🏆 MEJOR MODELO: {mejor_nombre} (CV F1: {mejor_score:.4f})", "SUCCESS")
    log_mensaje("="*70, "SUCCESS")
    
    return mejor_modelo, mejores_metricas, resultados


def generar_visualizaciones_todos_modelos(resultados, X_test, y_test):
    """Genera visualizaciones para TODOS los modelos entrenados."""
    log_mensaje("\n📊 Generando visualizaciones para todos los modelos...", "INFO")
    
    sns.set_style("whitegrid")
    
    for nombre_modelo, datos in resultados.items():
        if 'modelo' not in datos:
            continue
            
        modelo = datos['modelo']
        log_mensaje(f"  → Generando gráficos para {nombre_modelo}...", "INFO")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Matriz de Confusión
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Matriz de Confusión - {nombre_modelo}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Real')
        axes[0].set_xlabel('Predicción')
        
        # 2. Curva ROC
        y_proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_title(f'Curva ROC - {nombre_modelo}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Tasa de Falsos Positivos')
        axes[1].set_ylabel('Tasa de Verdaderos Positivos')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # Guardar con nombre específico del modelo
        nombre_archivo = f'evaluacion_{nombre_modelo.lower()}.png'
        ruta_viz = RUTA_VISUALIZACIONES / nombre_archivo
        plt.tight_layout()
        plt.savefig(ruta_viz, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_mensaje(f"    ✓ Guardado: {nombre_archivo}", "SUCCESS")


def guardar_modelo_y_metricas(modelo, metricas_ganador, todos_resultados):
    """Guarda el mejor modelo y las métricas de TODOS los modelos."""
    log_mensaje("\n💾 Guardando modelo ganador y métricas...", "INFO")
    
    # Guardar modelo ganador
    with open(RUTA_MODELOS / ARCHIVO_MODELO, 'wb') as f:
        pickle.dump(modelo, f)
    log_mensaje(f"  ✓ Modelo ganador guardado: {ARCHIVO_MODELO}", "SUCCESS")
    
    # Preparar métricas completas de todos los modelos
    metricas_completas = {
        'timestamp': metricas_ganador['timestamp'],
        'modelo_ganador': metricas_ganador['modelo_ganador'],
        'razon_seleccion': metricas_ganador.get('razon_seleccion', ''),
        'metricas_ganador': {
            'test_accuracy': metricas_ganador['test_accuracy'],
            'test_f1': metricas_ganador['test_f1'],
            'test_precision': metricas_ganador['test_precision'],
            'test_recall': metricas_ganador['test_recall'],
            'test_roc_auc': metricas_ganador['test_roc_auc'],
            'cv_f1_mean': metricas_ganador['cv_f1_mean'],
            'cv_f1_std': metricas_ganador['cv_f1_std']
        },
        'comparativa_todos_modelos': {}
    }
    
    # Agregar métricas de todos los modelos
    for nombre_modelo, datos in todos_resultados.items():
        if 'cv_f1_mean' in datos:
            metricas_completas['comparativa_todos_modelos'][nombre_modelo] = {
                'cv_f1_mean': datos['cv_f1_mean'],
                'test_f1': datos['test_f1'],
                'test_roc_auc': datos['test_roc_auc']
            }
    
    # Guardar métricas completas
    with open(RUTA_METRICAS / 'metricas_todos_modelos.json', 'w') as f:
        json.dump(metricas_completas, f, indent=2)
    log_mensaje(f"  ✓ Métricas de todos los modelos guardadas", "SUCCESS")


def main():
    try:
        log_mensaje("\nPIPELINE DE ENTRENAMIENTO MULTI-MODELO", "INFO")
        
        df = cargar_datos_gold()
        X_train, X_test, y_train, y_test, _ = preparar_datos_entrenamiento(df)
        
        mejor_modelo, metricas, resultados = entrenar_y_evaluar_modelos(
            X_train, X_test, y_train, y_test
        )
        
        if mejor_modelo:
            # Generar visualizaciones para TODOS los modelos
            generar_visualizaciones_todos_modelos(resultados, X_test, y_test)
            
            # Generar explicación dinámica del porqué ganó este modelo
            def construir_razon_ganador(nombre_ganador, resultados, metricas):
                cv_ganador = metricas['cv_f1_mean']
                f1_test_ganador = metricas['test_f1']
                
                razon = f"El modelo {nombre_ganador} fue seleccionado como ganador debido a que obtuvo "
                razon += f"el mayor CV F1-Score promedio ({cv_ganador:.4f}) entre todos los competidores. "
                razon += f"Además, demostró una excelente capacidad de generalización con un F1-Score en el set de prueba de {f1_test_ganador:.4f}. "
                
                # Comparativa técnica
                competidores = [m for m in resultados.keys() if m != nombre_ganador]
                if competidores:
                    comparaciones = []
                    for comp in competidores:
                        cv_comp = resultados[comp]['cv_f1_mean']
                        diferencia_puntos = cv_ganador - cv_comp
                        comparaciones.append(f"{comp} (CV F1: {cv_comp:.4f}, Dif: {diferencia_puntos:+.4f})")
                    
                    razon += "\n\nComparativa detallada contra otros modelos:\n- " + "\n- ".join(comparaciones)
                
                return razon

            # Asignar la razon real
            nombre_ganador = metricas['modelo_ganador']
            metricas['razon_seleccion'] = construir_razon_ganador(nombre_ganador, resultados, metricas)
            
            # Mostrar mensaje explicativo
            log_mensaje("\n" + "="*70, "INFO")
            log_mensaje("🏆 RAZÓN TÉCNICA DE SELECCIÓN (MÉTRICAS REALES)", "INFO")
            log_mensaje("="*70, "INFO")
            log_mensaje(f"\n{metricas['razon_seleccion']}\n", "SUCCESS")
            
            # Guardar modelo y métricas completas
            guardar_modelo_y_metricas(mejor_modelo, metricas, resultados)
            
            return mejor_modelo, metricas
        else:
            log_mensaje("❌ No se pudo entrenar ningún modelo correctamente.", "ERROR")
            return None, None
            
    except Exception as e:
        log_mensaje(f"Error crítico: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()