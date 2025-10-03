"""
CLI Principal - Detecção de Fraude em Cartão de Crédito

Modos de Operação:
1. pipeline: Executa pipeline completo de dados (7 steps)
2. train: Treina modelo com hiperparâmetros do configs.py
3. tune: Grid Search com atualização automática de configs.py
4. predict: Inferência em novas transações (CSV ou JSON)
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
import subprocess
import re

sys.path.append(str(Path(__file__).parent))

from src.ml.processing.loader import load_from_postgresql
from src.ml.models.configs import config
from src.services.database.connection import get_engine


def update_model_artifacts(model_path, metrics_path, model_version, overwrite=False):
    """
    Salva/atualiza modelo e métricas com versionamento
    
    Args:
        model_path: Path do arquivo .pkl do modelo
        metrics_path: Path do arquivo .json de métricas
        model_version: Versão do modelo
        overwrite: Se True, move arquivos antigos para archive/ e cria novos
    
    Returns:
        tuple: (final_model_path, final_metrics_path)
    """
    from datetime import datetime
    
    models_dir = Path(__file__).parent / 'models'
    archive_dir = models_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n📝 Salvando modelo e métricas...")
    
    # Se arquivos existem e não é overwrite, criar versão alternativa em archive
    if model_path.exists() and not overwrite:
        alt_model_path = archive_dir / f'xgboost_v{model_version}_{timestamp}.pkl'
        alt_metrics_path = archive_dir / f'xgboost_v{model_version}_{timestamp}_metrics.json'
        
        print(f"   ⚠️  Modelo v{model_version} já existe!")
        print(f"   💾 Salvando versão alternativa em archive/:")
        print(f"      - {alt_model_path.name}")
        print(f"      - {alt_metrics_path.name}")
        
        return alt_model_path, alt_metrics_path
    
    # Se é overwrite, mover antigos para archive
    elif model_path.exists() and overwrite:
        backup_model_path = archive_dir / f'xgboost_v{model_version}_{timestamp}.pkl'
        backup_metrics_path = archive_dir / f'xgboost_v{model_version}_{timestamp}_metrics.json'
        
        print(f"   📦 Movendo versão antiga (v{model_version}) para archive/:")
        print(f"      - {backup_model_path.name}")
        
        model_path.rename(backup_model_path)
        
        if metrics_path.exists():
            print(f"      - {backup_metrics_path.name}")
            metrics_path.rename(backup_metrics_path)
        
        return model_path, metrics_path
    
    # Primeira vez
    else:
        print(f"   ✅ Salvando modelo v{model_version} (primeira vez)")
        return model_path, metrics_path


def train_model(model_version='2.1.0', overwrite=False):
    """Treina modelo XGBoost com hiperparâmetros do configs.py"""
    print("\n" + "="*80)
    print("🚀 MODO: TREINAMENTO (Configs)")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Versão: {model_version}")
    
    engine = get_engine()
    
    print("\n📥 Verificando dados no PostgreSQL...")
    try:
        train_df = load_from_postgresql(engine, 'train_data')
        test_df = load_from_postgresql(engine, 'test_data')
        print(f"✅ Train: {len(train_df):,} linhas ({train_df['Class'].sum()} fraudes)")
        print(f"✅ Test: {len(test_df):,} linhas ({test_df['Class'].sum()} fraudes)")
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("Execute: python -m src.ml.pipelines.data_pipeline")
        return
    
    print("\n⚙️  Hiperparâmetros (configs.py):")
    for key, value in config.models.xgboost_params.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 Treinando...")
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix
        import pickle
        
        X_train = train_df.drop(columns=['Class'])
        y_train = train_df['Class']
        X_test = test_df.drop(columns=['Class'])
        y_test = test_df['Class']
        
        model = XGBClassifier(**config.models.xgboost_params)
        
        print(f"📊 Cross-Validation (k={config.models.cv_folds})...")
        cv = StratifiedKFold(n_splits=config.models.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=['precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=-1
        )
        
        print(f"   Precision: {cv_scores['test_precision'].mean():.4f}")
        print(f"   Recall: {cv_scores['test_recall'].mean():.4f}")
        
        print(f"\n🔧 Treinando modelo final...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        print(f"\n   PR-AUC: {pr_auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude'])}")
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {tn:,}  FP: {fp}")
        print(f"   FN: {fn}  TP: {tp}")
        
        # Preparar paths antes de salvar
        base_model_path = config.paths.models_dir / f"xgboost_v{model_version}.pkl"
        base_metrics_path = config.paths.reports_dir / f"xgboost_v{model_version}_metrics.json"
        
        # Aplicar versionamento
        final_model_path, final_metrics_path = update_model_artifacts(
            base_model_path, base_metrics_path, model_version, overwrite
        )
        
        # Salvar modelo
        with open(final_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✅ Modelo salvo: {final_model_path}")
        
        # Salvar métricas
        metrics = {
            'model_version': model_version,
            'pr_auc': pr_auc,
            'precision': float(tp / (tp + fp)),
            'recall': float(tp / (tp + fn)),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'hyperparameters': config.models.xgboost_params,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(final_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Métricas: {final_metrics_path}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


def update_hyperparams_json(best_params, test_metrics, model_version, overwrite=False):
    """
    Salva/atualiza hiperparâmetros em JSON com versionamento
    
    Args:
        best_params: Melhores hiperparâmetros do Grid Search
        test_metrics: Métricas do test set
        model_version: Versão do modelo
        overwrite: Se True, move JSON antigo para archive/ e cria novo
    """
    from datetime import datetime
    
    data_dir = Path(__file__).parent / 'data'
    json_path = data_dir / 'xgboost_hyperparameters.json'
    archive_dir = data_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)
    
    print("\n📝 Salvando hiperparâmetros em JSON...")
    
    # Se arquivo existe e não é overwrite, criar versão alternativa
    if json_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_json_path = archive_dir / f'xgboost_hyperparameters_v{model_version}_{timestamp}.json'
        
        print(f"   ⚠️  {json_path.name} já existe!")
        print(f"   💾 Salvando versão alternativa: {alt_json_path.name}")
        
        final_path = alt_json_path
    
    # Se é overwrite, mover antigo para archive
    elif json_path.exists() and overwrite:
        # Ler versão antiga do JSON antes de mover
        import json
        with open(json_path, 'r') as f:
            old_config = json.load(f)
        old_version = old_config.get('model_version', 'unknown')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = archive_dir / f'xgboost_hyperparameters_v{old_version}_{timestamp}.json'
        
        print(f"   📦 Movendo versão antiga (v{old_version}) para: {backup_path.name}")
        json_path.rename(backup_path)
        
        final_path = json_path
    
    # Primeira vez
    else:
        final_path = json_path
    
    # Construir JSON completo
    config_data = {
        "model_name": "XGBoost",
        "model_version": model_version,
        "description": f"XGBoost v{model_version} - Grid Search optimized",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "optimization_method": "GridSearchCV",
            "cv_folds": config.models.grid_search_cv,
            "cv_metric": config.models.cv_scoring
        },
        "hyperparameters": best_params,
        "metrics": {
            "test_set": test_metrics
        }
    }
    
    # Salvar JSON
    with open(final_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"   ✅ Hiperparâmetros salvos: {final_path}")
    
    if final_path == json_path:
        print(f"   🎯 configs.py lerá automaticamente este arquivo!")
    else:
        print(f"   ℹ️  Para usar estes parâmetros, renomeie para: xgboost_hyperparameters.json")


def update_configs_with_best_params(best_params, model_version):
    """DEPRECATED: Use update_hyperparams_json() instead"""
    print("⚠️  Função deprecada. Use update_hyperparams_json().")
    pass


def tune_model(model_version='2.2.0', overwrite=False):
    """
    Executa Grid Search e salva hiperparâmetros em JSON
    
    Args:
        model_version: Versão do modelo otimizado
        overwrite: Se True, sobrescreve xgboost_hyperparameters.json (move antigo para archive/)
    """
    print("\n" + "="*80)
    print("🔧 MODO: TUNING (Grid Search)")
    print("="*80)
    print(f"Versão: {model_version}")
    print(f"Overwrite JSON: {overwrite}")
    print(f"⚠️  ~20 minutos")
    
    engine = get_engine()
    
    try:
        train_df = load_from_postgresql(engine, 'train_data')
        test_df = load_from_postgresql(engine, 'test_data')
        print(f"✅ Train: {len(train_df):,}")
        print(f"✅ Test: {len(test_df):,}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    script_path = Path(__file__).parent / 'src' / 'ml' / 'training' / 'tune.py'
    
    if not script_path.exists():
        print(f"❌ Script não encontrado: {script_path}")
        return
    
    print(f"\n🚀 Executando Grid Search...")
    print(f"   Script: {script_path.name}")
    print(f"   Versão: {model_version}")
    print(f"   Overwrite: {overwrite}\n")
    
    try:
        # Executar script mostrando output em tempo real
        result = subprocess.run(
            ['python', str(script_path)],
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"\n✅ Grid Search completo!")
            
            results_path = config.paths.reports_dir / f"grid_search_results_v{model_version}.json"
            
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                best_params = results['best_params']
                test_metrics = results.get('test_set_metrics', {})
                best_score = results.get('best_pr_auc', test_metrics.get('pr_auc', 0))
                
                print(f"\n🏆 Melhores Parâmetros:")
                for key, value in best_params.items():
                    print(f"   {key}: {value}")
                print(f"\n   PR-AUC: {best_score:.4f}")
                
                # Atualizar JSON de hiperparâmetros
                update_hyperparams_json(best_params, test_metrics, model_version, overwrite)
            
            print(f"\n   Modelo: models/xgboost_v{model_version}.pkl")
            print(f"   Hiperparâmetros: data/xgboost_hyperparameters.json")
        else:
            print(f"\n❌ Grid Search falhou (exit code {result.returncode})")
    
    except Exception as e:
        print(f"❌ Erro ao executar Grid Search: {e}")


def predict_from_csv(filepath):
    """Predição em CSV"""
    print("\n" + "="*80)
    print("🔮 MODO: PREDIÇÃO (CSV)")
    print("="*80)
    
    if not Path(filepath).exists():
        print(f"❌ Arquivo não encontrado: {filepath}")
        return
    
    models_dir = Path(__file__).parent / 'models'
    model_files = sorted(models_dir.glob('xgboost_v*.pkl'), reverse=True)
    
    if not model_files:
        print(f"❌ Nenhum modelo. Execute: python main.py train")
        return
    
    model_path = model_files[0]
    
    print(f"📥 Carregando: {model_path.name}")
    model = joblib.load(model_path)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ {len(df):,} transações")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    try:
        X = df.drop(columns=['Class']) if 'Class' in df.columns else df
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        df['Prediction'] = predictions
        df['Fraud_Probability'] = probabilities
        
        output_path = Path(filepath).parent / f"{Path(filepath).stem}_predictions.csv"
        df.to_csv(output_path, index=False)
        
        n_frauds = predictions.sum()
        print(f"✅ Completo!")
        print(f"\n📊 Resumo:")
        print(f"   Total: {len(df):,}")
        print(f"   Fraudes: {n_frauds:,} ({n_frauds/len(df)*100:.2f}%)")
        print(f"\n💾 Resultados: {output_path}")
        
        if n_frauds > 0:
            print(f"\n⚠️  Top 5 Fraudes:")
            top_frauds = df[df['Prediction'] == 1].nlargest(5, 'Fraud_Probability')
            for idx, row in top_frauds.iterrows():
                print(f"   Linha {idx}: {row['Fraud_Probability']:.2%}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


def predict_from_json(json_input):
    """Predição em JSON"""
    print("\n" + "="*80)
    print("🔮 MODO: PREDIÇÃO (JSON)")
    print("="*80)
    
    if Path(json_input).exists():
        with open(json_input, 'r') as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            print(f"❌ JSON inválido: {e}")
            return
    
    models_dir = Path(__file__).parent / 'models'
    model_files = sorted(models_dir.glob('xgboost_v*.pkl'), reverse=True)
    
    if not model_files:
        print(f"❌ Nenhum modelo. Execute: python main.py train")
        return
    
    model_path = model_files[0]
    model = joblib.load(model_path)
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
        print(f"✅ {len(df):,} transações")
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
        print(f"✅ 1 transação")
    else:
        print(f"❌ Formato inválido")
        return
    
    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        results = []
        for i in range(len(df)):
            result = {
                'transaction_id': i,
                'prediction': 'FRAUDE' if predictions[i] == 1 else 'LEGÍTIMA',
                'fraud_probability': float(probabilities[i]),
                'confidence': 'Alta' if probabilities[i] > 0.8 or probabilities[i] < 0.2 else 'Média'
            }
            results.append(result)
        
        n_frauds = predictions.sum()
        print(f"✅ Completo!")
        print(f"\n📊 Resumo:")
        print(f"   Total: {len(df):,}")
        print(f"   Fraudes: {n_frauds:,}")
        
        print(f"\n📋 Resultados:")
        for result in results:
            emoji = "🚨" if result['prediction'] == 'FRAUDE' else "✅"
            print(f"   {emoji} {result['transaction_id']}: {result['prediction']} ({result['fraud_probability']:.2%})")
        
        output_path = Path('prediction_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Salvos: {output_path}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


def run_pipeline():
    """Executa pipeline completo de dados (Steps 01-07)"""
    print("\n" + "="*80)
    print("🚀 MODO: PIPELINE DE DADOS")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Tempo estimado: ~62 segundos")
    
    script_path = Path(__file__).parent / 'src' / 'ml' / 'pipelines' / 'data_pipeline.py'
    
    if not script_path.exists():
        print(f"❌ Script não encontrado: {script_path}")
        return
    
    print(f"\n🚀 Executando pipeline completo...")
    print(f"   Script: {script_path.name}")
    print(f"   Steps: 01-07\n")
    
    try:
        result = subprocess.run(
            ['python', '-m', 'src.ml.pipelines.data_pipeline'],
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline completo!")
            print(f"\n📊 Resultado esperado:")
            print(f"   ✅ raw_transactions: 284,807 linhas")
            print(f"   ✅ cleaned_transactions: 284,804 linhas")
            print(f"   ✅ imputed_transactions: 284,804 linhas")
            print(f"   ✅ normalized_transactions: 284,804 linhas")
            print(f"   ✅ engineered_transactions: 284,804 linhas (40 features)")
            print(f"   ✅ train_features: ~227,843 linhas (394 fraudes)")
            print(f"   ✅ test_features: ~56,961 linhas (98 fraudes)")
            print(f"   ✅ scalers.pkl salvo em models/")
            print(f"\n💡 Próximo passo: python main.py train")
        else:
            print(f"\n❌ Pipeline falhou (exit code {result.returncode})")
    
    except Exception as e:
        print(f"❌ Erro ao executar pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='CLI Detecção de Fraude',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Modo de operação')
    
    # Modo 1: Pipeline
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Executa pipeline completo de dados (Steps 01-07, ~62s)'
    )
    
    # Modo 2: Train
    train_parser = subparsers.add_parser(
        'train', 
        help='Treina modelo XGBoost. [-ow] para sobrescrever'
    )
    train_parser.add_argument('--model-version', default='2.1.0', help='Versão do modelo (default: 2.1.0)')
    train_parser.add_argument('-ow', '--overwrite', action='store_true', help='Sobrescreve modelo existente (move antigo para models/archive/)')
    
    # Modo 3: Tune
    tune_parser = subparsers.add_parser(
        'tune', 
        help='Grid Search p/ hiperparâmetros. [-ow] para sobrescrever'
    )
    tune_parser.add_argument('--model-version', default='2.2.0', help='Versão do modelo (default: 2.2.0)')
    tune_parser.add_argument('-ow', '--overwrite', action='store_true', help='Sobrescreve xgboost_hyperparameters.json (move antigo para data/archive/)')
    
    # Modo 4: Predict
    predict_parser = subparsers.add_parser('predict', help='Predições em transações')
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('--file', help='Path do arquivo CSV')
    predict_group.add_argument('--json-file', help='Path do arquivo JSON')
    predict_group.add_argument('--json', help='JSON string direto')
    
    args = parser.parse_args()
    
    if args.mode == 'pipeline':
        run_pipeline()
    
    elif args.mode == 'train':
        train_model(args.model_version, args.overwrite)
    
    elif args.mode == 'tune':
        tune_model(args.model_version, args.overwrite)
    
    elif args.mode == 'predict':
        if args.file:
            predict_from_csv(args.file)
        elif args.json_file:
            predict_from_json(args.json_file)
        elif args.json:
            predict_from_json(args.json)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
