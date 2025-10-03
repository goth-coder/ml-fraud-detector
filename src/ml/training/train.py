"""
Script de treinamento do modelo XGBoost para detecção de fraudes

Treina XGBoost com hiperparâmetros otimizados após análise comparativa
que demonstrou superioridade em todas as métricas.

Outputs:
- models/xgboost_v{version}.pkl
- reports/xgboost_v{version}_metrics.json
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
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


def load_training_data(engine):
    """
    Carrega dados de treino e teste do PostgreSQL
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print("\n" + "="*80)
    print("CARREGAR DADOS DE TREINO")
    print("="*80)
    
    try:
        train_df = load_from_postgresql(engine, 'train_data')
        test_df = load_from_postgresql(engine, 'test_data')
        
        X_train = train_df.drop('Class', axis=1)
        y_train = train_df['Class']
        
        X_test = test_df.drop('Class', axis=1)
        y_test = test_df['Class']
        
        print(f"✅ Train: {len(X_train):,} linhas, {len(X_train.columns)} features")
        print(f"✅ Test: {len(X_test):,} linhas, {len(X_test.columns)} features")
        print(f"✅ Train fraudes: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
        print(f"✅ Test fraudes: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        raise


def create_xgboost_model():
    """
    Cria instância do XGBoost com configuração otimizada
    
    Configuração baseada em:
    - scale_pos_weight=577 (ratio desbalanceamento 1:578)
    - max_depth=6 (previne overfitting)
    - learning_rate=0.1 (balanço velocidade/precisão)
    - n_estimators=100 (baseline, será otimizado em Grid Search)
    
    Returns:
        XGBClassifier: Modelo configurado
    """
    print("\n" + "="*80)
    print("CRIAR MODELO XGBOOST")
    print("="*80)
    
    model = XGBClassifier(
        scale_pos_weight=577,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )
    
    print(f"✅ xgboost: {type(model).__name__}")
    print(f"✅ Parâmetros: scale_pos_weight=577, max_depth=6, n_estimators=100")
    
    return model


def train_with_cross_validation(model, X_train, y_train):
    """
    Treina modelo com StratifiedKFold cross-validation
    
    Args:
        model: XGBClassifier
        X_train: Features de treino
        y_train: Target de treino
        
    Returns:
        dict: Métricas de cross-validation
    """
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION (STRATIFIEDKFOLD K=5)")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'pr_auc': 'average_precision',
        'roc_auc': 'roc_auc',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1'
    }
    
    print("🔄 Executando 5-fold cross-validation...")
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    print(f"⏱️  Tempo de treinamento: {training_time:.2f}s")
    print(f"\n📊 Resultados Cross-Validation:")
    print(f"{'Métrica':<15} {'Train':>12} {'Test':>12} {'Std':>10}")
    print("-" * 55)
    
    results = {}
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        train_mean = train_scores.mean()
        test_mean = test_scores.mean()
        test_std = test_scores.std()
        
        results[metric] = {
            'train_mean': train_mean,
            'test_mean': test_mean,
            'test_std': test_std
        }
        
        print(f"{metric:<15} {train_mean:>12.4f} {test_mean:>12.4f} {test_std:>10.4f}")
    
    overfitting = results['pr_auc']['train_mean'] - results['pr_auc']['test_mean']
    if overfitting > 0.1:
        print(f"\n⚠️  Possível overfitting: {overfitting:.4f}")
    else:
        print(f"\n✅ Sem overfitting significativo: {overfitting:.4f}")
    
    return results, training_time


def train_final_model(model, X_train, y_train):
    """
    Treina modelo final com todos os dados de treino
    
    Args:
        model: XGBClassifier
        X_train: Features de treino
        y_train: Target de treino
        
    Returns:
        model: Modelo treinado
    """
    print(f"\n{'='*80}")
    print(f"TREINAR MODELO FINAL")
    print(f"{'='*80}")
    
    print(f"🔧 Treinando com {len(X_train):,} amostras...")
    model.fit(X_train, y_train)
    print("✅ Modelo treinado!")
    
    return model


def evaluate_on_test(model, X_test, y_test):
    """
    Avalia modelo no conjunto de teste
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        dict: Métricas de teste
    """
    print(f"\n{'='*80}")
    print(f"AVALIAÇÃO EM TEST SET")
    print(f"{'='*80}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
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


def extract_feature_importance(model, feature_names):
    """
    Extrai feature importance do XGBoost
    
    Args:
        model: XGBoost treinado
        feature_names: Lista de nomes das features
        
    Returns:
        pd.DataFrame: Ranking de features
    """
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE")
    print(f"{'='*80}")
    
    importances = model.feature_importances_
    
    feature_ranking = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Features Mais Importantes:")
    print(feature_ranking.head(10).to_string(index=False))
    
    print(f"\n📊 Top 5 Features Menos Importantes:")
    print(feature_ranking.tail(5).to_string(index=False))

    low_importance = feature_ranking[feature_ranking['importance'] < 0.001]
    if len(low_importance) > 0:
        print(f"\n⚠️  Features com importance < 0.001 (candidatas a remoção):")
        print(low_importance.to_string(index=False))
    else:
        print(f"\n✅ Todas as features têm importance >= 0.001")
    
    return feature_ranking


def save_model(model, model_version="1.1.0"):
    """
    Salva modelo treinado em models/
    
    Args:
        model: Modelo treinado
        model_version: Versão do modelo
        
    Returns:
        str: Caminho do arquivo salvo
    """
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = models_dir / f"xgboost_v{model_version}.pkl"
    joblib.dump(model, filepath)
    
    print(f"\n💾 Modelo salvo: {filepath}")
    return str(filepath)


def save_metrics_report(cv_results, test_results, feature_importance, training_time):
    """
    Salva relatório completo de métricas
    
    Args:
        cv_results: Resultados cross-validation
        test_results: Resultados test set
        feature_importance: DataFrame com ranking de features
        training_time: Tempo de treinamento
    """
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    filepath = reports_dir / 'xgboost_metrics.json'
    
    report = {
        'model': 'XGBoost',
        'model_version': config.feature_selection.model_version,
        'training_date': datetime.now().isoformat(),
        'training_time_seconds': training_time,
        'n_features': len(feature_importance),
        'cross_validation': cv_results,
        'test_set': test_results,
        'feature_importance': {
            'top_10': feature_importance.head(10).to_dict('records'),
            'low_importance': feature_importance[feature_importance['importance'] < 0.001].to_dict('records')
        }
    }
    
    import json
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Relatório salvo: {filepath}")


def main():
    """
    Pipeline principal de treinamento XGBoost
    """
    print("\n" + "="*80)
    print("🚀 TREINAMENTO DO MODELO FINAL - XGBOOST")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Version: {config.feature_selection.model_version}")
    print(f"\n📌 Modelo: XGBoost (escolhido após análise comparativa)")
    print(f"📌 Features: 38 (Time removido)")
    engine = get_engine()
    
    X_train, y_train, X_test, y_test = load_training_data(engine)
    
    model = create_xgboost_model()
    
    cv_results, training_time = train_with_cross_validation(model, X_train, y_train)
    
    trained_model = train_final_model(model, X_train, y_train)
    
    test_results = evaluate_on_test(trained_model, X_test, y_test)
    
    feature_importance = extract_feature_importance(trained_model, X_train.columns.tolist())
    
    model_path = save_model(trained_model, config.feature_selection.model_version)
    
    save_metrics_report(cv_results, test_results, feature_importance, training_time)
    
    print("\n" + "="*80)
    print("✅ TREINAMENTO COMPLETO!")
    print("="*80)
    print(f"\n🏆 XGBoost - Métricas Finais:")
    print(f"   PR-AUC: {test_results['pr_auc']:.4f}")
    print(f"   Recall: {test_results['recall']:.4f}")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   F1-Score: {test_results['f1']:.4f}")
    print(f"   Falsos Positivos: {test_results['confusion_matrix']['fp']}")
    print(f"   Tempo: {training_time:.2f}s")


if __name__ == "__main__":
    main()
