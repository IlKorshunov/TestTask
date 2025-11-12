import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import config


class DataVisualizer:
    
    def __init__(self, output_dir='plots'):
        self.output_dir = Path(config.BASE_DIR) / output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def plot_data_overview(self, df, stage_name, filename):
        """
        Args:
            stage_name: название этапа
            filename: имя файла для сохранения
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Обзор данных: {stage_name}', fontsize=16, y=0.995)
        
        ax1 = axes[0, 0]
        if 'delta' in df.columns:
            ax1.hist(df['delta'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.axvline(df['delta'].mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {df["delta"].mean():.2f}')
            ax1.axvline(df['delta'].median(), color='green', linestyle='--', linewidth=2, label=f'Медиана: {df["delta"].median():.2f}')
            ax1.set_xlabel('Толщина (delta), мм')
            ax1.set_ylabel('Частота')
            ax1.set_title('Распределение толщины')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_stats = [(col, df[col].isna().sum() / len(df) * 100) for col in numeric_cols]
        missing_stats = sorted(missing_stats, key=lambda x: x[1], reverse=True)[:10]
        
        if missing_stats:
            cols, missing_pcts = zip(*missing_stats)
            ax2.barh(range(len(cols)), missing_pcts, color='coral', alpha=0.7)
            ax2.set_yticks(range(len(cols)))
            ax2.set_yticklabels(cols, fontsize=8)
            ax2.set_xlabel('Процент пропусков')
            ax2.set_title('Топ-10 признаков с пропусками')
            ax2.grid(True, alpha=0.3, axis='x')
        
        ax3 = axes[1, 0]
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        
        ax3.bar(['Числовые', 'Категориальные'], [numeric_count, categorical_count], 
                color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Количество')
        ax3.set_title('Типы признаков')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate([numeric_count, categorical_count]):
            ax3.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Статистика датасета:
        
        Размер: {len(df)} записей × {len(df.columns)} признаков
        
        Целевая переменная (delta):
          Среднее:    {df['delta'].mean():.3f} мм
          Медиана:    {df['delta'].median():.3f} мм
          Стд. откл.: {df['delta'].std():.3f} мм
          Мин:        {df['delta'].min():.3f} мм
          Макс:       {df['delta'].max():.3f} мм
          Пропуски:   {df['delta'].isna().sum()}
        
        Признаки:
          Числовые:        {numeric_count}
          Категориальные:  {categorical_count}
          Всего:           {len(df.columns)}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Сохранено: {output_path}")
        
    def plot_feature_importance(self, df_before, df_after, filename):
        """
        Сравнение данных до и после отбора признаков.
        
        Args:
            df_before: датасет до отбора
            df_after: датасет после отбора
            filename: имя файла
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Сравнение данных: до и после отбора признаков', fontsize=16)
        
        ax1 = axes[0]
        numeric_before = len(df_before.select_dtypes(include=[np.number]).columns)
        categorical_before = len(df_before.select_dtypes(include=['object']).columns)
        total_before = len(df_before.columns)
        
        colors = ['steelblue', 'coral', 'lightgray']
        sizes = [numeric_before, categorical_before, total_before - numeric_before - categorical_before]
        labels = [f'Числовые\n{numeric_before}', f'Категориальные\n{categorical_before}', 
                 f'Другие\n{sizes[2]}' if sizes[2] > 0 else '']
        
        sizes_filtered = [s for s in sizes if s > 0]
        labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
        colors_filtered = [c for c, s in zip(colors, sizes) if s > 0]
        
        wedges, texts, autotexts = ax1.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered,
                                           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax1.set_title(f'ДО отбора признаков\nВсего: {total_before} признаков', fontsize=12, fontweight='bold')
        
        ax2 = axes[1]
        numeric_after = len(df_after.select_dtypes(include=[np.number]).columns) - 1
        categorical_after = len(df_after.select_dtypes(include=['object']).columns)
        total_after = len(df_after.columns) - 1
        
        sizes = [numeric_after, categorical_after, total_after - numeric_after - categorical_after]
        labels = [f'Числовые\n{numeric_after}', f'Категориальные\n{categorical_after}', 
                 f'Другие\n{sizes[2]}' if sizes[2] > 0 else '']
        
        sizes_filtered = [s for s in sizes if s > 0]
        labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
        colors_filtered = [c for c, s in zip(colors, sizes) if s > 0]
        
        wedges, texts, autotexts = ax2.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered,
                                           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax2.set_title(f'ПОСЛЕ отбора признаков\nВсего: {total_after} признаков', fontsize=12, fontweight='bold')
        
        reduction_pct = (1 - total_after / total_before) * 100
        fig.text(0.5, 0.02, f'Сокращение: {total_before} → {total_after} признаков ({reduction_pct:.1f}%)',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Сохранено: {output_path}")
        
    def plot_correlation_heatmap(self, df, stage_name, filename, top_n=15):
        """
        Тепловая карта корреляций.
        
        Args:
            df: датасет
            stage_name: название этапа
            filename: имя файла
            top_n: количество топ признаков
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print(f"Недостаточно числовых признаков для корреляции ({len(numeric_df.columns)})")
            return
        
        if 'delta' in numeric_df.columns and len(numeric_df.columns) > top_n:
            correlations = numeric_df.corr()['delta'].abs().sort_values(ascending=False)
            top_features = correlations.head(top_n).index.tolist()
            numeric_df = numeric_df[top_features]
        
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title(f'Корреляционная матрица: {stage_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Сохранено: {output_path}")
        
    def plot_delta_analysis(self, df, stage_name, filename):
        """
        Детальный анализ целевой переменной.
        
        Args:
            df: датасет
            stage_name: название этапа
            filename: имя файла
        """
        if 'delta' not in df.columns:
            print("Целевая переменная 'delta' не найдена")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Анализ толщины (delta): {stage_name}', fontsize=16, y=0.995)
        
        delta = df['delta'].dropna()
        
        ax1 = axes[0, 0]
        ax1.hist(delta, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(delta.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {delta.mean():.2f}')
        ax1.axvline(delta.median(), color='green', linestyle='--', linewidth=2, label=f'Медиана: {delta.median():.2f}')
        ax1.set_xlabel('Толщина (мм)')
        ax1.set_ylabel('Частота')
        ax1.set_title('Распределение')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        bp = ax2.boxplot(delta, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('Толщина (мм)')
        ax2.set_title('Box plot')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = axes[1, 0]
        from scipy import stats
        stats.probplot(delta, dist="norm", plot=ax3)
        ax3.set_title('Q-Q график (проверка нормальности)')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Статистика delta:
        
        Основные:
          Среднее:      {delta.mean():.4f} мм
          Медиана:      {delta.median():.4f} мм
          Мода:         {delta.mode().values[0] if len(delta.mode()) > 0 else 'N/A':.4f} мм
          Стд. откл.:   {delta.std():.4f} мм
          Дисперсия:    {delta.var():.4f}
        
        Диапазон:
          Мин:          {delta.min():.4f} мм
          25%:          {delta.quantile(0.25):.4f} мм
          50%:          {delta.quantile(0.50):.4f} мм
          75%:          {delta.quantile(0.75):.4f} мм
          Макс:         {delta.max():.4f} мм
          Размах:       {delta.max() - delta.min():.4f} мм
        
        Форма:
          Асимметрия:   {delta.skew():.4f}
          Эксцесс:      {delta.kurtosis():.4f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Сохранено: {output_path}")


def generate_all_plots():    
    visualizer = DataVisualizer()
    
    processed_data = pd.read_csv(config.PROCESSED_DATA_PATH)
    visualizer.plot_data_overview(processed_data, 'Обработанные данные', '01_processed_overview.png')
    visualizer.plot_delta_analysis(processed_data, 'Обработанные данные', '02_processed_delta.png')
    visualizer.plot_correlation_heatmap(processed_data, 'Обработанные данные', '03_processed_correlation.png')
    
    final_data = pd.read_csv(config.FINAL_DATA_PATH)
    visualizer.plot_data_overview(final_data, 'После отбора признаков', '04_final_overview.png')
    visualizer.plot_delta_analysis(final_data, 'После отбора признаков', '05_final_delta.png')
    
    visualizer.plot_feature_importance(processed_data, final_data, '06_feature_selection_comparison.png')
    
    print(f"Сохранено в: {visualizer.output_dir}")


if __name__ == '__main__':
    generate_all_plots()
