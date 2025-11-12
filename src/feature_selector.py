import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression, VarianceThreshold
from typing import List, Dict

class FeatureSelector:    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.X, self.y = self._prepare_data()
        self.selection_summary: Dict[str, List[str]] = {}

    def _prepare_data(self) -> (pd.DataFrame, pd.Series):
        """
        Подготовка данных.
        Returns: X_scaled, y
        """
        df_for_selection = self.df[self.df[self.target_col].notna()].copy()
        df_processed = pd.get_dummies(df_for_selection, dummy_na=True)
        
        y = df_processed[self.target_col]
        X = df_processed.drop(columns=[self.target_col])
        
        id_columns = [col for col in X.columns if 'id' in col.lower() or col.endswith('_id')]
        if id_columns:
            X = X.drop(columns=id_columns)

        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X_scaled, y

    def remove_low_variance(self, threshold: float = 0.01) -> List[str]:
        """Удаляет признаки с малой дисперсией."""
        variances = self.X.var()
        to_keep = variances[variances > threshold].index.tolist()
        
        if len(to_keep) < 5:
            to_keep = variances.nlargest(min(20, len(self.X.columns))).index.tolist()
        
        removed = [col for col in self.X.columns if col not in to_keep]
        self.X = self.X[to_keep]
        return removed

    def remove_correlated(self, threshold: float = 0.95) -> List[str]:
        """Удаляет коррелирующие признаки."""
        corr_matrix = self.X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
        self.X = self.X.drop(columns=to_drop)
        return to_drop

    def select_with_l1(self, **kwargs) -> List[str]:
        """L1-регуляризация (Lasso)."""
        lasso = LassoCV(cv=5, random_state=42, **kwargs)
        lasso.fit(self.X, self.y)
        
        selected = list(self.X.columns[lasso.coef_ != 0])
        self.selection_summary['L1'] = selected
        return selected

    def select_with_trees(self, top_n: int = 20) -> List[str]:
        """RandomForest."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        importances = pd.Series(rf.feature_importances_, index=self.X.columns)
        selected = list(importances.nlargest(top_n).index)
        self.selection_summary['RF'] = selected
        return selected

    def select_with_rfe(self, n_features: int = 15) -> List[str]:
        """Recursive Feature Elimination"""
        rfe = RFE(RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=n_features)
        rfe.fit(self.X, self.y)
        
        selected = list(self.X.columns[rfe.support_])
        self.selection_summary['RFE'] = selected
        return selected

    def run(self, top_n: int = 10, var_thresh: float = 0.01, 
            corr_tresh: float = 0.95) -> (pd.DataFrame, pd.DataFrame):
        """
        Полный отбор признаков.

        Args:
            top_n: количество финальных признаков
            var_thresh: порог дисперсии
            corr_tresh: порог корреляции
            
        Returns:
            df_selected, summary - датасет и важность признаков
        """
        removed_variance = self.remove_low_variance(threshold=var_thresh)
        removed_corr = self.remove_correlated(threshold=corr_tresh)
        
        if removed_variance:
            print(f"Удалено (дисперсия): {len(removed_variance)}")
        if removed_corr:
            print(f"Удалено (корреляция): {len(removed_corr)}")
        
        if len(self.X.columns) < 5:
            print(f"Осталось мало признаков: {len(self.X.columns)}")
            selected_features = list(self.X.columns[:top_n])
            summary_df = pd.DataFrame(index=selected_features)
        else:
            self.select_with_trees(top_n=top_n)
            
            if self.y.std() > 0.01:
                try:
                    self.select_with_l1()
                except:
                    print("L1 пропущен")
            
            if len(self.X.columns) <= 50:
                try:
                    self.select_with_rfe(n_features=min(top_n, len(self.X.columns)))
                except:
                    print("RFE пропущен")
            
            summary_df = pd.DataFrame(index=self.X.columns)
            
            for method, features in self.selection_summary.items():
                summary_df[method] = summary_df.index.isin(features)
            
            summary_df['Total Votes'] = summary_df.sum(axis=1)
            summary_df = summary_df.sort_values('Total Votes', ascending=False)
            
            selected_features = summary_df[summary_df['Total Votes'] >= 2].index.tolist()
            if len(selected_features) < top_n:
                selected_features = summary_df.head(top_n).index.tolist()
        
        print(f"Отобрано: {len(selected_features)} признаков")
        
        original_feature_names = [col for col in self.df.columns if col in selected_features or any(col.startswith(f) for f in selected_features)]
        return (self.df[[self.target_col] + [col for col in self.df.columns if col in original_feature_names and col != self.target_col]].copy()), summary_df
