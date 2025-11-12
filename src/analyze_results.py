"""Скрипт для анализа результатов и создания отчета"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config
import joblib
import numpy as np
import os

def create_report():
    """Отчет по результатам моделирования."""
    
    print("1. Исходные данные")
    processed_data = pd.read_csv(config.PROCESSED_DATA_PATH)
    print(f"Размер: {processed_data.shape}")
    print(f"Delta: mean={processed_data['delta'].mean():.3f}, std={processed_data['delta'].std():.3f}")
    
    augmented_path = config.DATA_DIR / "augmented_data.csv"
    if augmented_path.exists():
        print(os.linesep)
        print("2. Дополненные данные")
        augmented_data = pd.read_csv(augmented_path)
        print(f"Размер: {augmented_data.shape}")
        print(f"Delta: mean={augmented_data['delta'].mean():.3f}, std={augmented_data['delta'].std():.3f}")
    
    print(os.linesep)
    print("3. Финальный датасет")
    final_data = pd.read_csv(config.FINAL_DATA_PATH)
    print(f"Размер: {final_data.shape}")
    print(f"Признаков: {len(final_data.columns) - 1}")
    
    print(os.linesep)
    print("4. Модель")
    if config.MODEL_SAVE_PATH.exists():
        model = joblib.load(config.MODEL_SAVE_PATH)
        print(f"Тип: {type(model.named_steps['regressor']).__name__}")
    
    print(os.linesep)

def create_visualizations():
    """Визуализация данных."""
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    augmented_path = config.DATA_DIR / "augmented_data.csv"
    if not augmented_path.exists():
        data = pd.read_csv(config.PROCESSED_DATA_PATH)
    else:
        data = pd.read_csv(augmented_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ данных по толщине элементов резервуаров', fontsize=16, y=1.02)
    
    ax1 = axes[0, 0]
    ax1.hist(data['delta'], bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Толщина (delta), мм')
    ax1.set_ylabel('Частота')
    ax1.set_title('Распределение толщины элементов')
    mean_val = data['delta'].mean()
    ax1.axvline(mean_val, color='red', linestyle='--', label=f'Среднее: {mean_val:.2f}')
    ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.boxplot(data['delta'])
    ax2.set_ylabel('Толщина (delta), мм')
    ax2.set_title('Box plot толщины элементов')
    ax2.set_xticklabels(['Delta'])
    
    ax3 = axes[1, 0]
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        feature = [col for col in numeric_cols if col != 'delta'][0]
        ax3.scatter(data[feature], data['delta'], alpha=0.5)
        ax3.set_xlabel(feature)
        ax3.set_ylabel('Толщина (delta), мм')
        ax3.set_title(f'Зависимость толщины от {feature}')
    else:
        ax3.text(0.5, 0.5, 'Недостаточно признаков\nдля визуализации', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Scatter plot')
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""    
    Размер выборки: {len(data)}
    
    Толщина (delta):
     Среднее: {data['delta'].mean():.3f} мм
     Медиана: {data['delta'].median():.3f} мм
     Стд. откл.: {data['delta'].std():.3f} мм
     Мин: {data['delta'].min():.3f} мм
     Макс: {data['delta'].max():.3f} мм
    
    Признаков: {len(data.columns) - 1}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    
    output_path = config.BASE_DIR / "analysis_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Сохранено: {output_path}")
    plt.close()

if __name__ == '__main__':
    create_report()
    
    try:
        create_visualizations()
    except Exception as e:
        print(f"Ошибка визуализации: {e}")
