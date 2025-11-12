import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from typing import Dict, Any
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
import joblib
import logging

from config import MODEL_SAVE_PATH, RANDOM_STATE
from hyperparameter_tuner import HyperparameterTuner

log = logging.getLogger(__name__)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        weights = torch.softmax(self.attention_weights(x), dim=-1)
        return x * weights

class DeepRegressionNet(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(DeepRegressionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            SelfAttention(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class ModelTrainer:
    """Обучение и оценка нескольких моделей регрессии."""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.df_train = self.df[self.df[target_col].notna()].copy()
        self.df_test = self.df[self.df[target_col].isna()].copy()
        self.X_train = self.df_train.drop(columns=[self.target_col])
        self.y_train = self.df_train[self.target_col].values.astype(np.float32).reshape(-1, 1)
        self.X_test = self.df_test.drop(columns=[self.target_col]) if len(self.df_test) > 0 else pd.DataFrame()

    def _prepare_data(self):
        """
        Подготовка данных к обучению.
        Train: все записи с delta != null
        Test: все записи с delta == null
        
        Returns:
            X_train, X_test, y_train, y_test (y_test будет None для записей с delta=null)
        """
        for col in self.X_train.select_dtypes(include=np.number).columns: 
            self.X_train[col].fillna(self.X_train[col].median(), inplace=True)
        for col in self.X_train.select_dtypes(include=['object', 'category']).columns: 
            self.X_train[col].fillna('missing', inplace=True)
        
        if len(self.X_test) > 0:
            for col in self.X_test.select_dtypes(include=np.number).columns:
                if self.X_test[col].isnull().any():
                    self.X_test[col].fillna(self.X_train[col].median() if col in self.X_train.columns else 0, inplace=True)
            for col in self.X_test.select_dtypes(include=['object', 'category']).columns:
                if self.X_test[col].isnull().any():
                    self.X_test[col].fillna('missing', inplace=True)
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=RANDOM_STATE
        )
        
        return X_train_split, X_val, self.X_test, y_train_split, y_val, None

    def run(self, tune_hyperparams=True):
        """
        Обучение моделей.
        Args: tune_hyperparams: запустить подбор гиперпараметров
        Returns: лучшая модель
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data()
        
        log.info(f"Train: {len(X_train)} записей, Val: {len(X_val)} записей, Test: {len(X_test)} записей")
        
        categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns
        numerical_features = self.X_train.select_dtypes(include=np.number).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test) if len(X_test) > 0 else None
        
        tuned_params = {}
        if tune_hyperparams:
            log.info("Подбор гиперпараметров")
            tuner = HyperparameterTuner(X_train_transformed, y_train, cv=5, random_state=RANDOM_STATE)
            tuned_params = tuner.tune_all(quick_mode=True)
        
        ridge_params = tuned_params.get('Ridge', {}).get('params', {'alpha': 1.0})
        ridge_params['random_state'] = RANDOM_STATE
        
        lasso_params = tuned_params.get('Lasso', {}).get('params', {'alpha': 0.1})
        lasso_params['random_state'] = RANDOM_STATE
        
        rf_params = tuned_params.get('RandomForest', {}).get('params', {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5
        })
        rf_params['random_state'] = RANDOM_STATE
        
        gb_params = tuned_params.get('GradientBoosting', {}).get('params', {
            'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5
        })
        gb_params['random_state'] = RANDOM_STATE

        input_size = X_train_transformed.shape[1]
        neural_net = NeuralNetRegressor(
            DeepRegressionNet,
            module__input_size=input_size,
            module__dropout_rate=0.3,
            max_epochs=50,
            lr=0.001,
            batch_size=32,
            optimizer=torch.optim.Adam,
            device='cpu',
            verbose=0
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(**ridge_params),
            "Lasso": Lasso(**lasso_params),
            "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(**rf_params),
            "Gradient Boosting": GradientBoostingRegressor(**gb_params),
            "Neural Network": neural_net
        }
        
        best_model = None
        best_rmse = float('inf')
        best_model_name = ""
        results = []

        for name, model in models.items():
            log.info(f"Модель: {name}")
            
            try:
                model.fit(X_train_transformed, y_train.ravel())
                
                preds_train = model.predict(X_train_transformed)
                preds_val = model.predict(X_val_transformed)
                
                # Метрики на валидационной выборке
                rmse = np.sqrt(mean_squared_error(y_val, preds_val))
                mae = mean_absolute_error(y_val, preds_val)
                r2 = r2_score(y_val, preds_val)
                
                rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
                r2_train = r2_score(y_train, preds_train)
                
                log.info(f"Val: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                log.info(f"Train: RMSE={rmse_train:.4f}, R2={r2_train:.4f}")
                
                overfit_ratio = rmse / rmse_train if rmse_train > 0 else float('inf')
                log.info(f"Переобучение: {overfit_ratio:.2f}")
                
                results.append({
                    'Model': name,
                    'RMSE_val': rmse,
                    'MAE_val': mae,
                    'R2_val': r2,
                    'RMSE_train': rmse_train,
                    'R2_train': r2_train,
                    'Overfit_ratio': overfit_ratio
                })
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_model_name = name
                    best_preprocessor = preprocessor
                    
            except Exception as e:
                log.error(f"Ошибка {name}: {e}")
        
        if results:
            results_df = pd.DataFrame(results).sort_values('RMSE_val')
            log.info("Результаты:")
            log.info(results_df.to_string(index=False))
        
        if best_model:
            log.info(f"Лучшая модель: {best_model_name}")
            log.info(f"RMSE: {best_rmse:.4f}")
            MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
            
            pipeline = Pipeline(steps=[('preprocessor', best_preprocessor), ('regressor', best_model)])
            joblib.dump(pipeline, MODEL_SAVE_PATH)
            log.info(f"Сохранено: {MODEL_SAVE_PATH}")
            
            if len(self.df_test) > 0 and X_test_transformed is not None:
                log.info(f"Делаем предсказания на {len(self.df_test)} записях с delta=null")
                predictions = best_model.predict(X_test_transformed)
                
                self.df_test[self.target_col] = predictions
                log.info(f"Предсказания для тестовых записей: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
                
                predictions_path = MODEL_SAVE_PATH.parent / "test_predictions.csv"
                self.df_test.to_csv(predictions_path, index=False)
                log.info(f"Сохранены предсказания: {predictions_path}")
            else:
                log.info("Нет записей с delta=null для предсказания")

        return best_model
