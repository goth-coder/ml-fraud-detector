"""
Otimização de Hiperparâmetros XGBoost

Grid Search para otimizar hiperparâmetros do modelo XGBoost.
Compara resultados com versões anteriores automaticamente.

Outputs:
- models/xgboost_v{version}.pkl
- reports/grid_search_results_v{version}.json
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.ml.processing.loader import load_from_postgresql
from src.ml.models.configs import config
from src.services.database.connection import get_engine


def load_previous_versions():
    """
    Carrega métricas de versões anteriores do archive/
    
    Returns:
        dict: {version: {pr_auc, precision, fp, features, ...}}
    """
    data_dir = Path(__file__).parent.parent.parent.parent / 'data'
    archive_dir = data_dir / 'archive'
    
    versions = {}
    
    if not archive_dir.exists():
        return versions
    
    for json_file in archive_dir.glob('xgboost_hyperparameters_v*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            version = data.get('model_version', 'unknown')
            
            if 'metrics' in data and 'test_set' in data['metrics']:
                metrics = data['metrics']['test_set']
                versions[version] = {
                    'pr_auc': metrics.get('pr_auc', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'fp': metrics['confusion_matrix'].get('fp', 0) if 'confusion_matrix' in metrics else 0,
                    'features': data.get('features', {}).get('total', 0),
                    'file': json_file.name
                }
        except Exception as e:
            continue
    
    return versions


def load_training_data(engine):
    """Carrega dados de treino e teste do PostgreSQL"""
    print("\n" + "="*80)
    print("CARREGAR DADOS")
    print("="*80)
    
    train_df = load_from_postgresql(engine, 'train_data')
    test_df = load_from_postgresql(engine, 'test_data')
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    print(f"✅ Train: {len(X_train):,} linhas, {len(X_train.columns)} features")
    print(f"✅ Test: {len(X_test):,} linhas, {len(X_test.columns)} features")
    print(f"✅ Fraudes train: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"✅ Fraudes test: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")
    
    return X_train, y_train, X_test, y_test


def define_param_grid():
    """Define grid de hiperparâmetros para otimização"""
    print("\n" + "="*80)
    print("GRID DE HIPERPARÂMETROS")
    print("="*80)
    
    param_grid = {
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 3, 5],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'scale_pos_weight': [577],  # Ratio fraud:legit
    }
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    print("📊 Parâmetros a testar:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")
    
    print(f"\n🔢 Total de combinações: {total_combinations:,}")
    print(f"⏱️  Tempo estimado: {total_combinations * 2 / 60:.1f} minutos")
    
    return param_grid


def run_grid_search(X_train, y_train, param_grid):
    """
    Executa Grid Search com Cross-Validation
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        param_grid: Grid de hiperparâmetros
        
    Returns:
        GridSearchCV: Objeto com resultados
    """
    print("\n" + "="*80)
    print("EXECUTAR GRID SEARCH (PR-AUC)")
    print("="*80)
    
    base_model = XGBClassifier(
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("🔄 Iniciando Grid Search...")
    print(f"   - CV Folds: 3 (StratifiedKFold)")
    print(f"   - Scoring: PR-AUC (average_precision)")
    print(f"   - n_jobs: -1 (todos os cores)")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='average_precision',  # PR-AUC scorer built-in
        cv=skf,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Grid Search completo em {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
    
    return grid_search


def analyze_grid_results(grid_search):
    """
    Analisa resultados do Grid Search e compara com versões anteriores
    
    Args:
        grid_search: GridSearchCV fitted
    """
    print("\n" + "="*80)
    print("ANÁLISE DE RESULTADOS")
    print("="*80)
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    print(f"\n🏆 Melhores Parâmetros:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    best_score = grid_search.best_score_
    print(f"\n📊 Melhor PR-AUC (CV): {best_score:.4f}")
    
    # Comparar com versões anteriores dinamicamente
    previous_versions = load_previous_versions()
    
    if previous_versions:
        sorted_versions = sorted(
            previous_versions.items(), 
            key=lambda x: x[1].get('pr_auc', 0), 
            reverse=True
        )
        
        print(f"\n� Versões anteriores:")
        for version, metrics in sorted_versions:
            pr_auc_old = metrics.get('pr_auc', 0)
            print(f"   {version}: PR-AUC {pr_auc_old:.4f}")
        
        # Comparar com melhor versão anterior
        best_old_version = sorted_versions[0]
        best_old_pr_auc = best_old_version[1].get('pr_auc', 0)
        improvement = ((best_score - best_old_pr_auc) / best_old_pr_auc) * 100
        
        if improvement > 0:
            print(f"✅ Melhoria sobre {best_old_version[0]}: +{improvement:.2f}%")
        else:
            print(f"⚠️  Resultado vs {best_old_version[0]}: {improvement:.2f}%")
    
    top_5 = results_df.nsmallest(5, 'rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'mean_train_score']
    ]
    
    print(f"\n📊 Top 5 Configurações:")
    print("-" * 80)
    for idx, row in top_5.iterrows():
        train_score = row['mean_train_score']
        test_score = row['mean_test_score']
        overfitting = train_score - test_score
        
        print(f"\nRank {row.name + 1}:")
        print(f"  Params: {row['params']}")
        print(f"  CV PR-AUC: {test_score:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Train PR-AUC: {train_score:.4f}")
        print(f"  Overfitting: {overfitting:.4f}")
    
    return results_df


def evaluate_best_model(grid_search, X_test, y_test):
    """
    Avalia melhor modelo no test set
    
    Args:
        grid_search: GridSearchCV fitted
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        dict: Métricas no test set
    """
    print("\n" + "="*80)
    print("AVALIAÇÃO NO TEST SET")
    print("="*80)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"📊 PR-AUC (Test): {pr_auc:.4f}")
    print(f"📊 ROC-AUC (Test): {roc_auc:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude']))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def save_optimized_model(grid_search, version="2.1.0"):
    """
    Salva modelo otimizado
    
    Args:
        grid_search: GridSearchCV fitted
        version: Versão do modelo
        
    Returns:
        str: Caminho do arquivo salvo
    """
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = models_dir / f"xgboost_v{version}.pkl"
    joblib.dump(grid_search.best_estimator_, filepath)
    
    print(f"\n💾 Modelo otimizado salvo: {filepath}")
    return str(filepath)


def save_grid_search_report(grid_search, test_metrics, results_df, elapsed_time, excluded_features):
    """
    Salva relatório completo do Grid Search
    
    Args:
        grid_search: GridSearchCV fitted
        test_metrics: Métricas no test set
        results_df: DataFrame com todos os resultados
        elapsed_time: Tempo total de execução
        excluded_features: Lista de features removidas
    """
    reports_dir = Path(__file__).parent.parent.parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    filepath = reports_dir / 'grid_search_results_v2.1.json'
    
    top_5_configs = []
    top_5_df = results_df.nsmallest(5, 'rank_test_score')
    for _, row in top_5_df.iterrows():
        top_5_configs.append({
            'rank': int(row['rank_test_score']),
            'params': row['params'],
            'cv_pr_auc_mean': float(row['mean_test_score']),
            'cv_pr_auc_std': float(row['std_test_score']),
            'train_pr_auc_mean': float(row['mean_train_score']),
            'overfitting': float(row['mean_train_score'] - row['mean_test_score'])
        })
    
    # Carregar versões anteriores dinamicamente
    previous_versions = load_previous_versions()
    
    report = {
        'model': 'XGBoost',
        'model_version': '2.1.0',
        'optimization_date': datetime.now().isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'features_excluded': excluded_features,
        'previous_versions': previous_versions,
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'test_set_metrics': test_metrics,
        'top_5_configurations': top_5_configs,
        'total_configurations_tested': len(results_df)
    }
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Relatório salvo: {filepath}")


def main():
    """Pipeline principal de otimização"""
    print("\n" + "="*80)
    print("🚀 OTIMIZAÇÃO DE HIPERPARÂMETROS - XGBOOST v2.1.0")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = ModelConfig.load_from_json()
    
    print(f"\n📋 Features REMOVIDAS nesta versão:")
    excluded = config.feature_selection.excluded_features
    print(f"   {excluded}")
    
    # Carregar e exibir versões anteriores
    previous_versions = load_previous_versions()
    
    if previous_versions:
        print(f"\n📌 Versões anteriores disponíveis:")
        sorted_versions = sorted(
            previous_versions.items(), 
            key=lambda x: x[1].get('pr_auc', 0), 
            reverse=True
        )
        
        for version, metrics in sorted_versions:
            pr_auc = metrics.get('pr_auc', 0)
            precision = metrics.get('precision', 0) * 100
            fp = metrics.get('fp', 'N/A')
            features = metrics.get('features', 'N/A')
            symbol = "✅" if version == sorted_versions[0][0] else "  "
            
            print(f"   {symbol} {version}: PR-AUC {pr_auc:.4f}, "
                  f"Precision {precision:.2f}%, FP {fp}, Features {features}")
    
    start_time = time.time()
    
    engine = get_engine()
    
    X_train, y_train, X_test, y_test = load_training_data(engine)
    
    param_grid = define_param_grid()
    
    grid_search = run_grid_search(X_train, y_train, param_grid)
    
    results_df = analyze_grid_results(grid_search)
    
    test_metrics = evaluate_best_model(grid_search, X_test, y_test)
    
    model_path = save_optimized_model(grid_search, version="2.1.0")
    
    elapsed_time = time.time() - start_time
    
    save_grid_search_report(
        grid_search, 
        test_metrics, 
        results_df, 
        elapsed_time,
        config.feature_selection.excluded_features
    )
    
    print("\n" + "="*80)
    print("✅ OTIMIZAÇÃO COMPLETA!")
    print("="*80)
    
    print(f"\n🏆 Modelo Otimizado (v2.1.0):")
    print(f"   PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1-Score: {test_metrics['f1']:.4f}")
    print(f"   Falsos Positivos: {test_metrics['confusion_matrix']['fp']}")
    
    # Comparação final com melhor versão anterior
    if previous_versions:
        sorted_versions = sorted(
            previous_versions.items(), 
            key=lambda x: x[1].get('pr_auc', 0), 
            reverse=True
        )
        best_old_version = sorted_versions[0]
        best_old_pr_auc = best_old_version[1].get('pr_auc', 0)
        best_old_fp = best_old_version[1].get('fp', 'N/A')
        
        print(f"\n📊 Comparação com {best_old_version[0]}:")
        print(f"   PR-AUC: {best_old_pr_auc:.4f} → {test_metrics['pr_auc']:.4f}")
        print(f"   FP: {best_old_fp} → {test_metrics['confusion_matrix']['fp']}")
        
        improvement = ((test_metrics['pr_auc'] - best_old_pr_auc) / best_old_pr_auc) * 100
        if improvement > 0:
            print(f"\n✅ Melhoria: +{improvement:.2f}%")
        elif improvement > -1:
            print(f"\n⚠️  Performance similar: {improvement:.2f}%")
        else:
            print(f"\n❌ Piora: {improvement:.2f}%")
    
    print(f"\n⏱️  Tempo total: {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")


if __name__ == "__main__":
    main()
