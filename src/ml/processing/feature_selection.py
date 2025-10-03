"""
Feature Selection - Análise e remoção de features
Estratégia: Config-driven com análise automatizada
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import json
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'Class') -> Dict[str, Any]:
    """
    Analisa importância de features usando múltiplos métodos
    
    Args:
        df: DataFrame com features + target
        target_col: Nome da coluna target
    
    Returns:
        dict: Relatório completo com candidatos a remoção
        
    Métodos:
        1. Pearson Correlation (linear)
        2. Spearman Correlation (monotônica)
        3. Mutual Information (não-linear geral)
        4. VIF - Variance Inflation Factor (multicolinearidade)
        5. Correlation Matrix (redundância entre features)
    """
    print("\n🔍 Analisando importância de features...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Verificar se há colunas não-numéricas (safety check)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"   ⚠️  Removendo {len(non_numeric)} colunas não-numéricas: {list(non_numeric)}")
        X = X.select_dtypes(include=[np.number])
    
    # 1. Correlação Pearson (linear)
    print("   • Calculando Pearson correlation...")
    pearson_corr = X.corrwith(y).abs().sort_values(ascending=True)
    
    # 2. Correlação Spearman (monotônica)
    print("   • Calculando Spearman correlation...")
    spearman_corr = X.apply(lambda col: abs(spearmanr(col, y)[0])).sort_values(ascending=True)
    
    # 3. Mutual Information (não-linear)
    print("   • Calculando Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)
    
    # 4. Identificar candidatos a remoção (baixos em TODOS os 3 métodos)
    low_pearson = pearson_corr[pearson_corr < 0.01].index.tolist()
    low_spearman = spearman_corr[spearman_corr < 0.01].index.tolist()
    low_mi = mi_series[mi_series < 0.001].index.tolist()
    
    # Features baixas em TODOS os 3 métodos
    candidates_removal = list(set(low_pearson) & set(low_spearman) & set(low_mi))
    
    # 5. Multicolinearidade (VIF)
    print("   • Calculando VIF (Variance Inflation Factor)...")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) 
        for i in range(len(X.columns))
    ]
    high_vif = vif_data[vif_data["VIF"] > 10].sort_values("VIF", ascending=False)
    
    # 6. Correlação entre features (redundância)
    print("   • Analisando redundância (correlação entre features)...")
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    redundant_pairs = []
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > 0.95):
            max_corr_feature = upper_triangle[column].idxmax()
            max_corr_value = upper_triangle[column].max()
            redundant_pairs.append((column, max_corr_feature, max_corr_value))
    
    # Criar relatório completo
    report = {
        "total_features": len(X.columns),
        "analysis_date": pd.Timestamp.now().isoformat(),
        
        "correlation_analysis": {
            "top_10_features": pearson_corr.tail(10).to_dict(),
            "bottom_10_features": pearson_corr.head(10).to_dict()
        },
        
        "candidates_for_removal": {
            "low_correlation_all_methods": sorted(candidates_removal),
            "count": len(candidates_removal),
            "features_detail": {
                feat: {
                    "pearson": float(pearson_corr[feat]),
                    "spearman": float(spearman_corr[feat]),
                    "mutual_info": float(mi_series[feat])
                }
                for feat in candidates_removal
            }
        },
        
        "multicollinearity": {
            "high_vif_features": high_vif.to_dict('records') if len(high_vif) > 0 else [],
            "high_vif_count": len(high_vif),
            "redundant_pairs": [
                {"feature1": f1, "feature2": f2, "correlation": float(corr)}
                for f1, f2, corr in redundant_pairs
            ],
            "redundant_count": len(redundant_pairs)
        },
        
        "recommendation": {
            "action": "Review and update configs.py",
            "path": "src/ml/models/configs.py",
            "config_key": "feature_selection.excluded_features",
            "suggested_removals": sorted(candidates_removal)
        }
    }
    
    print("✅ Análise concluída!")
    return report


def save_feature_selection_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Salva relatório de feature selection em JSON
    
    Args:
        report: Dicionário com análise completa
        output_path: Caminho para salvar JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Relatório de Feature Selection salvo: {output_path}")
    print_feature_selection_summary(report)


def print_feature_selection_summary(report: Dict[str, Any]) -> None:
    """
    Imprime resumo visual do relatório de feature selection
    
    Args:
        report: Dicionário com análise completa
    """
    print("\n" + "="*70)
    print("📊 RESUMO - ANÁLISE DE FEATURE SELECTION")
    print("="*70)
    
    candidates = report['candidates_for_removal']
    print(f"\n🔍 Total de features analisadas: {report['total_features']}")
    print(f"⚠️  Candidatas a remoção: {candidates['count']}")
    
    if candidates['count'] > 0:
        print(f"\n📋 Features com baixa correlação em TODOS os métodos:")
        for feat in candidates['low_correlation_all_methods']:
            details = candidates['features_detail'][feat]
            print(f"   - {feat}:")
            print(f"     • Pearson:     {details['pearson']:.6f}")
            print(f"     • Spearman:    {details['spearman']:.6f}")
            print(f"     • Mutual Info: {details['mutual_info']:.6f}")
    else:
        print("✅ Nenhuma feature com baixa correlação em todos os métodos!")
    
    multicollinearity = report['multicollinearity']
    
    if multicollinearity['high_vif_count'] > 0:
        print(f"\n⚠️  Features com alta multicolinearidade (VIF > 10): {multicollinearity['high_vif_count']}")
        for feat_info in multicollinearity['high_vif_features'][:5]:
            print(f"   - {feat_info['Feature']}: VIF = {feat_info['VIF']:.2f}")
    else:
        print("✅ Nenhuma feature com VIF > 10 (boa independência)")
    
    if multicollinearity['redundant_count'] > 0:
        print(f"\n🔗 Pares de features redundantes (correlação > 0.95): {multicollinearity['redundant_count']}")
        for pair in multicollinearity['redundant_pairs'][:5]:
            print(f"   - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("✅ Nenhum par com correlação > 0.95 (boa diversidade)")
    
    print(f"\n💡 Recomendação:")
    print(f"   1. Revise o relatório completo em: reports/feature_selection_analysis.json")
    print(f"   2. Edite: {report['recommendation']['path']}")
    print(f"   3. Atualize: {report['recommendation']['config_key']}")
    print(f"   4. Re-execute o pipeline para aplicar mudanças")
    
    print("="*70)


def apply_feature_selection(
    df: pd.DataFrame, 
    excluded_features: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Remove features configuradas em configs.py
    
    Args:
        df: DataFrame com todas as features
        excluded_features: Lista de features a serem removidas
        verbose: Se True, imprime informações
    
    Returns:
        DataFrame sem as features excluídas
        
    Example:
        >>> excluded = ['V13', 'V15', 'V22']
        >>> df_filtered = apply_feature_selection(df, excluded)
    """
    if not excluded_features:
        if verbose:
            print("✅ Nenhuma feature configurada para remoção")
        return df
    
    # Verificar se features existem
    existing_excluded = [f for f in excluded_features if f in df.columns]
    missing = [f for f in excluded_features if f not in df.columns]
    
    if missing and verbose:
        print(f"⚠️  Features não encontradas (ignoradas): {missing}")
    
    if existing_excluded:
        df_filtered = df.drop(columns=existing_excluded)
        
        if verbose:
            print(f"\n🗑️  Features removidas ({len(existing_excluded)}):")
            for feat in existing_excluded:
                print(f"   - {feat}")
            print(f"\n📊 Features antes: {len(df.columns)}")
            print(f"📊 Features depois: {len(df_filtered.columns)}")
            print(f"📊 Features removidas: {len(existing_excluded)}")
        
        return df_filtered
    
    return df
