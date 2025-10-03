"""
Data Pipeline - Pipeline completo de processamento de dados
Encadeia funções de processing/ para transformar dados raw → engineered
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adicionar src ao path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from src.ml.processing.loader import (
    load_csv_to_dataframe,
    load_from_postgresql,
    save_to_postgresql
)
from src.ml.processing.validation import (
    validate_csv_schema,
    validate_no_missing_values,
    validate_class_distribution
)
from src.ml.processing.cleaning import (
    identify_outliers,
    analyze_fraud_distribution_in_outliers,
    handle_missing_values
)
from src.ml.processing.normalization import (
    fit_and_transform_features,
    save_scalers,
    get_scaler_statistics
)
from src.ml.processing.feature_engineering import engineer_all_features
from src.ml.processing.feature_selection import (
    analyze_feature_importance,
    save_feature_selection_report,
    apply_feature_selection
)
from src.ml.processing.splitters import (
    stratified_train_test_split,
    validate_split_integrity,
    save_split_to_postgresql
)
from src.ml.processing.metadata import save_pipeline_metadata
from src.ml.models.configs import config
from services.database.connection import create_db_engine


class DataPipeline:
    """Pipeline de processamento de dados end-to-end"""
    
    def __init__(self):
        self.engine = create_db_engine()
        self.config = config
        self.scalers = None
    
    def run_step_01_load_raw_data(self) -> None:
        """Step 01: Carrega CSV → PostgreSQL"""
        print("="*70)
        print("STEP 01: LOAD RAW DATA - CSV → PostgreSQL")
        print("="*70)
        
        # Carregar CSV
        df = load_csv_to_dataframe(self.config.paths.raw_csv)
        
        # Validar schema
        validate_csv_schema(df)
        validate_class_distribution(df)
        
        # Salvar no PostgreSQL
        save_to_postgresql(df, self.engine, self.config.pipeline.table_raw)
        
        print(f"\n✅ Step 01 concluído: {len(df):,} linhas em {self.config.pipeline.table_raw}")
    
    def run_step_02_outlier_analysis(self) -> None:
        """Step 02: Análise de outliers (SEM remoção)"""
        print("\n" + "="*70)
        print("STEP 02: OUTLIER ANALYSIS (IQR - SEM REMOÇÃO)")
        print("="*70)
        
        start_time = time.time()
        
        # Carregar de raw_transactions (já está no PostgreSQL)
        df = load_from_postgresql(self.engine, self.config.pipeline.table_raw)
        
        # Identificar outliers
        outliers = identify_outliers(df, column='Amount')
        
        # Analisar impacto em fraudes
        fraud_stats = analyze_fraud_distribution_in_outliers(df, outliers)
        
        # DECISÃO: NÃO REMOVER (preservar 100% dos dados)
        # NÃO salvar nova tabela, apenas metadata
        
        duration = time.time() - start_time
        
        # Salvar metadata da análise (dados NÃO modificados)
        save_pipeline_metadata(
            engine=self.engine,
            step_number=2,
            step_name='outlier_analysis',
            rows_processed=len(df),
            rows_output=len(df),  # Mesma quantidade (100% preservado)
            data_modified=False,  # Dados NÃO alterados
            metadata={
                'outliers_detected': len(outliers),
                'outliers_percentage': round(len(outliers) / len(df) * 100, 2),
                'frauds_in_outliers': fraud_stats.get('frauds_in_outliers', 0),
                'decision': 'preserve_all_data',
                'reason': 'preserve_18.5%_frauds_in_outliers'
            },
            duration_seconds=duration
        )
        
        print(f"\n✅ Step 02 concluído: {len(df):,} linhas preservadas (100%)")
        print(f"📊 Metadata salvo (dados NÃO duplicados no PostgreSQL)")
        print(f"⏱️  Duração: {duration:.2f}s")
    
    def run_step_03_handle_missing(self) -> None:
        """Step 03: Tratamento de missing values"""
        print("\n" + "="*70)
        print("STEP 03: HANDLE MISSING VALUES")
        print("="*70)
        
        start_time = time.time()
        
        # Carregar de raw_transactions (Step 02 não modificou)
        df = load_from_postgresql(self.engine, self.config.pipeline.table_raw)
        
        # Analisar missing values
        missing_report = validate_no_missing_values(df)
        
        # Se não há missing values, não precisa salvar dados novamente
        if len(missing_report) == 0:
            duration = time.time() - start_time
            
            # Salvar metadata (dados NÃO modificados)
            save_pipeline_metadata(
                engine=self.engine,
                step_number=3,
                step_name='handle_missing_values',
                rows_processed=len(df),
                rows_output=len(df),
                data_modified=False,  # Nenhum dado alterado
                metadata={
                    'missing_values_found': 0,
                    'action': 'none_required',
                    'reason': 'dataset_100%_complete'
                },
                duration_seconds=duration
            )
            
            print("✅ Nenhum missing value encontrado")
            print(f"📊 Metadata salvo (dados NÃO duplicados no PostgreSQL)")
            print(f"⏱️  Duração: {duration:.2f}s")
        else:
            # Caso haja missing values (futuro), SALVAR nova tabela
            df_imputed = handle_missing_values(df)
            save_to_postgresql(df_imputed, self.engine, self.config.pipeline.table_imputed)
            
            duration = time.time() - start_time
            save_pipeline_metadata(
                engine=self.engine,
                step_number=3,
                step_name='handle_missing_values',
                rows_processed=len(df),
                rows_output=len(df_imputed),
                data_modified=True,
                metadata={'missing_values_imputed': len(missing_report)},
                duration_seconds=duration
            )
        
        print(f"\n✅ Step 03 concluído: {len(df):,} linhas")
    
    def run_step_04_normalize(self) -> None:
        """Step 04: Normalização de features"""
        print("\n" + "="*70)
        print("STEP 04: NORMALIZE FEATURES (RobustScaler)")
        print("="*70)
        
        start_time = time.time()
        
        # Carregar de raw_transactions (Steps 02-03 não modificaram)
        df = load_from_postgresql(self.engine, self.config.pipeline.table_raw)
        
        # Normalizar features
        df_normalized, self.scalers = fit_and_transform_features(df)
        
        # Exibir estatísticas dos scalers
        scaler_stats = get_scaler_statistics(self.scalers)
        
        # Salvar scalers
        save_scalers(self.scalers, self.config.paths.scalers_pkl)
        
        # Salvar dados normalizados (DADOS MODIFICADOS - nova tabela) - OTIMIZADO!
        save_to_postgresql(df_normalized, self.engine, self.config.pipeline.table_normalized)
        
        duration = time.time() - start_time
        
        # Salvar metadata (dados MODIFICADOS)
        save_pipeline_metadata(
            engine=self.engine,
            step_number=4,
            step_name='normalize_features',
            rows_processed=len(df),
            rows_output=len(df_normalized),
            data_modified=True,  # Dados ALTERADOS
            metadata={
                'features_normalized': ['Amount', 'Time'],
                'scaler_amount': 'RobustScaler',
                'scaler_time': 'StandardScaler',
                'scalers_saved': str(self.config.paths.scalers_pkl)
            },
            duration_seconds=duration
        )
        
        print(f"\n✅ Step 04 concluído: {len(df_normalized):,} linhas normalizadas")
        print(f"💾 Dados salvos em: {self.config.pipeline.table_normalized}")
        print(f"⏱️  Duração: {duration:.2f}s")
    
    def run_step_05_feature_engineering(self) -> None:
        """Step 05: Feature engineering"""
        print("\n" + "="*70)
        print("STEP 05: FEATURE ENGINEERING")
        print("="*70)
        
        start_time = time.time()
        
        # Carregar dados normalizados (última tabela modificada)
        df = load_from_postgresql(self.engine, self.config.pipeline.table_normalized)
        
        original_columns = len(df.columns)
        
        # Criar novas features
        df_engineered = engineer_all_features(
            df,
            include_interactions=self.config.pipeline.include_interactions
        )
        
        new_features = len(df_engineered.columns) - original_columns
        
        # Salvar dados com features engineered (DADOS MODIFICADOS - nova tabela) - OTIMIZADO!
        save_to_postgresql(df_engineered, self.engine, self.config.pipeline.table_engineered)
        
        duration = time.time() - start_time
        
        # Salvar metadata
        save_pipeline_metadata(
            engine=self.engine,
            step_number=5,
            step_name='feature_engineering',
            rows_processed=len(df),
            rows_output=len(df_engineered),
            data_modified=True,  # Dados ALTERADOS
            metadata={
                'original_features': original_columns,
                'new_features': new_features,
                'total_features': len(df_engineered.columns),
                'interactions_included': self.config.pipeline.include_interactions
            },
            duration_seconds=duration
        )
        
        print(f"\n✅ Step 05 concluído: {len(df_engineered):,} linhas, {len(df_engineered.columns)} features")
        print(f"💾 Dados salvos em: {self.config.pipeline.table_engineered}")
        print(f"⏱️  Duração: {duration:.2f}s")
    
    def run_step_05_5_feature_selection_analysis(self) -> None:
        """
        Step 05.5: Análise de feature selection (APENAS ANÁLISE)
        NÃO modifica dados, apenas gera relatório JSON para decisão manual
        """
        print("\n" + "="*70)
        print("STEP 05.5: FEATURE SELECTION ANALYSIS (Config-Driven)")
        print("="*70)
        
        start_time = time.time()
        
        # Carregar dados engineered (última tabela modificada)
        df = load_from_postgresql(self.engine, self.config.pipeline.table_engineered)
        
        # Analisar features
        report = analyze_feature_importance(df, target_col=self.config.data.target_column)
        
        # Salvar relatório
        save_feature_selection_report(report, self.config.paths.feature_selection_report)
        
        duration = time.time() - start_time
        
        # Salvar metadata (dados NÃO modificados)
        save_pipeline_metadata(
            engine=self.engine,
            step_number=5.5,
            step_name='feature_selection_analysis',
            rows_processed=len(df),
            rows_output=len(df),
            data_modified=False,  # Nenhum dado alterado
            metadata={
                'candidates_for_removal': report['candidates_for_removal']['count'],
                'high_vif_count': report['multicollinearity']['high_vif_count'],
                'redundant_pairs': report['multicollinearity']['redundant_count'],
                'report_path': str(self.config.paths.feature_selection_report),
                'action_required': 'review_and_update_configs'
            },
            duration_seconds=duration
        )
        
        print(f"\n✅ Step 05.5 concluído: Análise gerada")
        print(f"📄 Próximo passo: Revisar {self.config.paths.feature_selection_report}")
        print(f"📝 Editar: src/ml/models/configs.py → feature_selection.excluded_features")
        print(f"⏱️  Duração: {duration:.2f}s")
    
    def run_step_06_apply_feature_selection(self) -> None:
        """
        Step 06: Aplica feature selection baseado em configs.py
        Remove features configuradas em excluded_features
        """
        print("\n" + "="*70)
        print("STEP 06: APPLY FEATURE SELECTION (Config-Driven)")
        print("="*70)
        
        start_time = time.time()
        
        # Carregar dados engineered
        df = load_from_postgresql(self.engine, self.config.pipeline.table_engineered)
        
        original_features = len(df.columns)
        
        # Aplicar feature selection
        df_selected = apply_feature_selection(
            df,
            excluded_features=self.config.feature_selection.excluded_features,
            verbose=True
        )
        
        features_removed = original_features - len(df_selected.columns)
        
        duration = time.time() - start_time
        
        if features_removed > 0:
            # SALVAR nova tabela (dados MODIFICADOS)
            save_to_postgresql(df_selected, self.engine, self.config.pipeline.table_selected)
            
            save_pipeline_metadata(
                engine=self.engine,
                step_number=6,
                step_name='apply_feature_selection',
                rows_processed=len(df),
                rows_output=len(df_selected),
                data_modified=True,
                metadata={
                    'original_features': original_features,
                    'features_removed': features_removed,
                    'features_remaining': len(df_selected.columns),
                    'excluded_list': self.config.feature_selection.excluded_features,
                    'model_version': self.config.feature_selection.model_version
                },
                duration_seconds=duration
            )
            
            print(f"\n✅ Step 06 concluído: {features_removed} features removidas")
            print(f"💾 Dados salvos em: {self.config.pipeline.table_selected}")
        else:
            # Sem mudanças, apenas metadata
            save_pipeline_metadata(
                engine=self.engine,
                step_number=6,
                step_name='apply_feature_selection',
                rows_processed=len(df),
                rows_output=len(df),
                data_modified=False,
                metadata={
                    'features_removed': 0,
                    'reason': 'no_exclusions_configured',
                    'model_version': self.config.feature_selection.model_version
                },
                duration_seconds=duration
            )
            
            print(f"\n✅ Step 06 concluído: Nenhuma feature removida (configuração vazia)")
        
        print(f"⏱️  Duração: {duration:.2f}s")
    
    def run_step_07_train_test_split(self) -> None:
        """Step 07: Split train/test estratificado (antigo Step 06)"""
        print("\n" + "="*70)
        print("STEP 07: TRAIN/TEST SPLIT (Stratified)")
        print("="*70)
        
        start_time = time.time()
        
        # Decidir qual tabela carregar (com ou sem feature selection)
        features_removed = len(self.config.feature_selection.excluded_features)
        
        if features_removed > 0:
            # Carregar de selected_features (pós feature selection)
            df = load_from_postgresql(self.engine, self.config.pipeline.table_selected)
            print(f"📊 Carregando dados de: {self.config.pipeline.table_selected}")
        else:
            # Carregar de engineered_transactions (sem feature selection)
            df = load_from_postgresql(self.engine, self.config.pipeline.table_engineered)
            print(f"📊 Carregando dados de: {self.config.pipeline.table_engineered}")
        
        # Split estratificado
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            df,
            target_column=self.config.data.target_column,
            test_size=1 - self.config.data.train_test_split_ratio,
            random_state=self.config.data.random_state
        )
        
        # Validar integridade
        validate_split_integrity(X_train, X_test, y_train, y_test)
        
        # Salvar split no PostgreSQL (DADOS MODIFICADOS - novas tabelas)
        save_split_to_postgresql(X_train, X_test, y_train, y_test, self.engine)
        
        duration = time.time() - start_time
        
        # Salvar metadata
        save_pipeline_metadata(
            engine=self.engine,
            step_number=6,
            step_name='train_test_split',
            rows_processed=len(df),
            rows_output=len(X_train) + len(X_test),
            data_modified=True,  # Dados ALTERADOS
            metadata={
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_frauds': int(y_train.sum()),
                'test_frauds': int(y_test.sum()),
                'split_ratio': self.config.data.train_test_split_ratio,
                'stratified': True
            },
            duration_seconds=duration
        )
        
        print(f"\n✅ Step 06 concluído: Train={len(X_train):,}, Test={len(X_test):,}")
        print(f"⏱️  Duração: {duration:.2f}s")
    
    def run_full_pipeline(self) -> None:
        """Executa pipeline completo com otimizações de concorrência"""
        print("\n" + "🚀"*35)
        print("EXECUTANDO PIPELINE COMPLETO DE DADOS (OTIMIZADO)")
        print("🚀"*35 + "\n")
        
        pipeline_start = time.time()
        
        # Step 01: Load raw data
        self.run_step_01_load_raw_data()
        
        # Steps 02-03: PARALELIZADOS (independentes, ambos read-only)
        print("\n⚡ Executando Steps 02-03 em paralelo...")
        parallel_start = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit ambos os steps
            future_02 = executor.submit(self.run_step_02_outlier_analysis)
            future_03 = executor.submit(self.run_step_03_handle_missing)
            
            # Aguardar conclusão
            for future in as_completed([future_02, future_03]):
                try:
                    future.result()  # Propaga exceptions se houver
                except Exception as e:
                    print(f"❌ Erro em step paralelo: {e}")
                    raise
        
        parallel_duration = time.time() - parallel_start
        print(f"✅ Steps 02-03 concluídos em paralelo: {parallel_duration:.2f}s")
        
        # Steps 04-07: Sequenciais (dependências)
        self.run_step_04_normalize()
        self.run_step_05_feature_engineering()
        self.run_step_05_5_feature_selection_analysis()
        self.run_step_06_apply_feature_selection()
        self.run_step_07_train_test_split()
        
        total_duration = time.time() - pipeline_start
        
        print("\n" + "✅"*35)
        print(f"PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
        print(f"⏱️  Tempo total: {total_duration:.2f}s")
        print("✅"*35)


def main():
    """Executa pipeline de dados"""
    pipeline = DataPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
