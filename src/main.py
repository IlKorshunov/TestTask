from model_trainer import ModelTrainer
from data_processor import DataProcessor
from feature_selector import FeatureSelector
import pandas as pd
import logging
import config
import numpy as np

def setup_logging():
    """Настраивает базовую конфигурацию логирования."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def augment_data(df, target_col='delta'):
    """
    Создает аугментированные примеры при низкой вариации данных.
    """
    log = logging.getLogger(__name__)
    log.info("Дополнение данных")
    
    df_with_target = df[df[target_col].notna()].copy()
    original_size = len(df)
    
    if len(df_with_target) == 0:
        log.warning("Нет записей с delta для аугментации")
        return df
    
    if df_with_target[target_col].std() < 0.01:
        log.warning(f"Низкая вариация '{target_col}': {df_with_target[target_col].mean():.2f}")
        log.warning("Создание синтетических данных")
        
        n_synthetic = int(len(df_with_target) * 0.2)
        df_synthetic = df_with_target.sample(n=n_synthetic, replace=True, random_state=42).copy()
        
        base_delta = df_with_target[target_col].mean()
        df_synthetic[target_col] = base_delta + np.random.normal(0, base_delta * 0.15, n_synthetic)
        df_synthetic[target_col] = df_synthetic[target_col].clip(lower=base_delta * 0.5, upper=base_delta * 2)
        
        numeric_cols = df_synthetic.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col:
                col_mean = df_synthetic[col].mean()
                col_std = df_synthetic[col].std()
                
                if col_std > 0: noise = np.random.normal(0, col_std * 0.05, n_synthetic)
                elif col_mean != 0: noise = np.random.normal(0, abs(col_mean) * 0.02, n_synthetic)
                else: noise = np.random.normal(0, 0.1, n_synthetic)
                df_synthetic[col] = df_synthetic[col] + noise
        
        df_augmented = pd.concat([df, df_synthetic], ignore_index=True)
        
        log.info(f"Создано примеров: {len(df_augmented) - original_size}")
        log.info(f"Размер: {original_size} -> {len(df_augmented)}")
        
        return df_augmented
    
    return df

def main():
    """Пайплайн обработки обучения"""
    setup_logging()
    log = logging.getLogger(__name__)
    
    log.info("Запуск пайплайна и обработки данных")
    DataProcessor(data_path=config.RAW_DATA_PATH).process(output_path=config.PROCESSED_DATA_PATH)
    
    df_original = pd.read_csv(config.PROCESSED_DATA_PATH)
    log.info(f"Загружено {len(df_original)} записей")
    
    log.info("Отбор признаков")
    df_selected, _ = FeatureSelector(df_original, target_col=config.TARGET_COLUMN).run(top_n=config.TOP_N_FEATURES, var_thresh=0.05, corr_tresh=0.75)
    log.info(f"После отбора: {df_selected.shape}")
    
    df_with_target = df_original[df_original[config.TARGET_COLUMN].notna()]
    if len(df_original) < 100 or (len(df_with_target) > 0 and df_with_target[config.TARGET_COLUMN].std() < 0.01):
        df_augmented = augment_data(df_selected, target_col=config.TARGET_COLUMN)
        augmented_path = config.DATA_DIR / "augmented_data.csv"
        df_augmented.to_csv(augmented_path, index=False)
        log.info(f"Сохранено: {augmented_path}")
        df_selected = df_augmented
        log.info(f"Финальный датасет: {df_selected.shape}")
    
    df_selected.to_csv(config.FINAL_DATA_PATH, index=False)
    log.info(f"Сохранено: {config.FINAL_DATA_PATH}")
    
    log.info("Обучение моделей")
    ModelTrainer(df_selected, target_col=config.TARGET_COLUMN).run(tune_hyperparams=(len(df_selected) > 500))
    log.info("Пайплайн завершен")

if __name__ == '__main__':
    main()

