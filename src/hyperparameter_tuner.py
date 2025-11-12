import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error
import logging

log = logging.getLogger(__name__)

class HyperparameterTuner:
    """Подбор гиперпараметров с кросс-валидацией."""
    
    def __init__(self, X_train, y_train, cv=5, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.random_state = random_state
        self.rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))

    def tune_random_forest(self, n_iter=20):
        """Подбор для Random Forest."""
        log.info("Подбор гиперпараметров: Random Forest")
        
        param_distributions = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        search = RandomizedSearchCV(RandomForestRegressor(random_state=self.random_state), param_distributions, n_iter=n_iter, cv=self.cv, scoring=self.rmse_scorer, random_state=self.random_state, n_jobs=-1, verbose=0)
        search.fit(self.X_train, self.y_train.ravel())
        
        log.info(f"Лучшие параметры: {search.best_params_}")
        log.info(f"CV RMSE: {-search.best_score_:.4f}")
        
        return search.best_params_, -search.best_score_

    def tune_gradient_boosting(self):
        """Подбор для GB."""
        log.info("Подбор гиперпараметров: Gradient Boosting")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        
        gb = GradientBoostingRegressor(random_state=self.random_state)
        
        search = GridSearchCV(
            gb,
            param_grid,
            cv=self.cv,
            scoring=self.rmse_scorer,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(self.X_train, self.y_train.ravel())
        
        log.info(f"Лучшие параметры: {search.best_params_}")
        log.info(f"CV RMSE: {-search.best_score_:.4f}")
        
        return search.best_params_, -search.best_score_

    def tune_ridge(self):
                
        ridge = Ridge(random_state=self.random_state)
        search = GridSearchCV(ridge, {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, cv=self.cv, scoring=self.rmse_scorer, n_jobs=-1, verbose=0)
        search.fit(self.X_train, self.y_train.ravel())
        
        log.info(f"Лучшие параметры: {search.best_params_}")
        
        return search.best_params_, -search.best_score_

    def tune_lasso(self):
        log.info("Подбор гиперпараметров: Lasso")
        
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        search = GridSearchCV(Lasso(random_state=self.random_state), {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}, cv=self.cv, scoring=self.rmse_scorer, n_jobs=-1, verbose=0)
        search.fit(self.X_train, self.y_train.ravel())
        
        log.info(f"Лучшие параметры: {search.best_params_}")
        log.info(f"CV RMSE: {-search.best_score_:.4f}")
        
        return search.best_params_, -search.best_score_

    def cross_validate_model(self, model, model_name):
        """
        Кросс-валидация модели.
        
        Args:
            model: модель sklearn
            model_name: название модели
            
        Returns:
            cv_scores - массив RMSE по фолдам
        """
        log.info(f"Кросс-валидация: {model_name}")
        rmse_scores = -cross_val_score(model, self.X_train, self.y_train.ravel(), cv=self.cv, scoring=self.rmse_scorer, n_jobs=-1)
        log.info(f"CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
        log.info(f"Фолды: {', '.join([f'{s:.4f}' for s in rmse_scores])}")
        
        return rmse_scores

    def tune_all(self, quick_mode=True):
        """
        Подбор гиперпараметров для всех моделей.
        
        Args:
            quick_mode: быстрый режим (меньше итераций)
            
        Returns:
            словарь с лучшими параметрами для каждой модели
        """
        results = {}
        
        try:
            params, score = self.tune_ridge()
            results['Ridge'] = {'params': params, 'cv_score': score}
        except Exception as e:
            log.error(f"Ошибка Ridge: {e}")
        
        try:
            params, score = self.tune_lasso()
            results['Lasso'] = {'params': params, 'cv_score': score}
        except Exception as e:
            log.error(f"Ошибка Lasso: {e}")
        
        if quick_mode:
            n_iter = 10
        else:
            n_iter = 20
        
        try:
            params, score = self.tune_random_forest(n_iter=n_iter)
            results['RandomForest'] = {'params': params, 'cv_score': score}
        except Exception as e:
            log.error(f"Ошибка RF: {e}")
        
        try:
            params, score = self.tune_gradient_boosting()
            results['GradientBoosting'] = {'params': params, 'cv_score': score}
        except Exception as e:
            log.error(f"Ошибка GB: {e}")
        
        if results:
            log.info("Сводка по подбору гиперпараметров:")
            for model_name, data in results.items():
                log.info(f"  {model_name}: CV RMSE = {data['cv_score']:.4f}")
        
        return results

