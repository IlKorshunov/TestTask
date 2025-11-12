import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class DataProcessor:
    """Обработка исходных данных."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
    def safe_json_parse(self, json_str):
        """Парсинг JSON."""
        if pd.isna(json_str) or json_str == '' or json_str == '""': return {}
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def load_and_parse_tank(self):
        """Резервуар."""
        log.info("Загрузка tank.csv")
        df = pd.read_csv(self.data_path / 'tank.csv')
        
        df['obj_parsed'] = df['obj'].apply(self.safe_json_parse)
        
        tank_features = []
        for _, row in df.iterrows():
            obj = row['obj_parsed']
            
            features = {
                'tank_id': row['tank_id'],
                'diameter': obj.get('d'),
                'yearStart': obj.get('yearStart'),
                'beltQuantity': obj.get('beltQuantity'),
                'tankType': obj.get('tankType'),
                'tankFluid': obj.get('tankFluid'),
                'fluidCorr': obj.get('fluidCorr'),
                'steelGradeWall': obj.get('steelGradeWall'),
                'baseType': obj.get('baseType'),
                'roof_type': obj.get('roof_type'),
                'maxWallT': obj.get('maxWallT'),
                'minWallT': obj.get('minWallT'),
            }
            
            load_blocks = obj.get('loadBlocks', [])
            if load_blocks and isinstance(load_blocks, list):
                h_maxs = [b.get('H_max_j') for b in load_blocks if b.get('H_max_j') is not None]
                h_mins = [b.get('H_min_j') for b in load_blocks if b.get('H_min_j') is not None]
                cycles = [b.get('load_cycle_quantity') for b in load_blocks if b.get('load_cycle_quantity') is not None]
                
                features['load_h_max'] = max(h_maxs) if h_maxs else None
                features['load_h_min'] = min(h_mins) if h_mins else None
                features['load_amplitude'] = (max(h_maxs) - min(h_mins)) if h_maxs and h_mins else None
                features['load_cycles_total'] = sum(cycles) if cycles else None
                features['load_blocks_count'] = len(load_blocks)
            else:
                features.update({'load_h_max': None, 'load_h_min': None, 'load_amplitude': None, 
                               'load_cycles_total': None, 'load_blocks_count': 0})
            
            soil_layers = obj.get('soilLayers', [])
            if soil_layers and isinstance(soil_layers, list):
                soil_types = [s.get('soilType') for s in soil_layers if s.get('soilType')]
                base_temps = [s.get('baseSoilT') for s in soil_layers if s.get('baseSoilT') is not None]
                
                features['soil_layers_count'] = len(soil_layers)
                features['soil_min_temp'] = min(base_temps) if base_temps else None
                features['soil_avg_temp'] = np.mean(base_temps) if base_temps else None
                features['soil_has_peat'] = 1 if 'peat' in soil_types else 0
                features['soil_has_salted'] = 1 if any(s.get('soilSalinity') == 'salted' for s in soil_layers) else 0
            else:
                features.update({'soil_layers_count': 0, 'soil_min_temp': None, 'soil_avg_temp': None,
                               'soil_has_peat': 0, 'soil_has_salted': 0})
            
            install_data = obj.get('installData', {})
            if install_data and isinstance(install_data, dict):
                features['rejDintK'] = install_data.get('rejDintK')
                features['rejFound'] = install_data.get('rejFound')
                features['rejWallDev'] = install_data.get('rejWallDev')
            else:
                features.update({'rejDintK': None, 'rejFound': None, 'rejWallDev': None})
            
            diag_data = obj.get('diagnosticData', {})
            if diag_data and isinstance(diag_data, dict):
                min_deltas = [diag_data.get(f'minDeltaWall{i}') for i in range(1, 13) if diag_data.get(f'minDeltaWall{i}') is not None]
                features['min_delta_wall_avg'] = np.mean(min_deltas) if min_deltas else None
                features['min_delta_wall_min'] = min(min_deltas) if min_deltas else None
                features['min_delta_bottom'] = diag_data.get('minDeltaBottom')
                features['rejIncreaseDevK'] = diag_data.get('rejIncreaseDevK')
            else:
                features.update({'min_delta_wall_avg': None, 'min_delta_wall_min': None,
                               'min_delta_bottom': None, 'rejIncreaseDevK': None})
            
            tank_features.append(features)
        
        tank_features = pd.DataFrame(tank_features)
        
        log.info(f"Резервуаров: {len(tank_features)}")
        return tank_features
    
    def load_and_parse_element(self):
        """Элемент."""
        log.info("Загрузка element.csv")
        df = pd.read_csv(self.data_path / 'element.csv')
        
        df['obj_parsed'] = df['obj'].apply(self.safe_json_parse)
        
        element_features = pd.DataFrame([
            {
                'element_id': row['element_id'],
                'tank_id': row['tank_id'],
                'tag': row['tag'],
                'r_part': obj.get('rPart', ''),
                'r_subpart': obj.get('rSubpart', ''),
                'controlType': obj.get('controlType', ''),
                'weldFinish': obj.get('weldFinish', ''),
                'weldMethod': obj.get('weldMethod', ''),
                'weldType': obj.get('weldType', ''),
            }
            for _, row in df.iterrows()
            for obj in [row['obj_parsed']]
        ])
        
        log.info(f"Элементов: {len(element_features)}")
        return element_features
    
    def load_and_parse_element_data(self):
        """Измерение элемента."""
        log.info("Загрузка ElementData.csv")
        df = pd.read_csv(self.data_path / 'ElementData.csv')
        
        df['obj_parsed'] = df['obj'].apply(self.safe_json_parse)
        
        measurements = pd.DataFrame([
            {
                'id': row['id'],
                'element_id': row['element_id'],
                'activityId': row['Activity_id'],
                'delta': obj.get('delta'),
                'createdAt': obj.get('createdAt'),
                'defect_H': obj.get('H'),
                'defect_L': obj.get('L'),
                'defect_W': obj.get('W'),
                'repair': obj.get('repair', False),
                'defectKind': obj.get('defectKind', ''),
                'defectType': obj.get('defectType', ''),
                'geometryDefect': obj.get('geometryDefect', ''),
            }
            for _, row in df.iterrows()
            for obj in [row['obj_parsed']]
        ])
        

        measurements = measurements[
            (measurements['delta'].isna()) | 
            ((measurements['delta'] > 0))
        ]
        
        measurements['createdAt'] = pd.to_datetime(measurements['createdAt'], errors='coerce', utc=True)
        measurements['createdAt'] = measurements['createdAt'].dt.tz_localize(None)
        measurements = measurements[measurements['createdAt'].notna()]
        
        log.info(f"Измерений: {len(measurements)}")
        return measurements
    
    def load_and_parse_activity(self):
        """Активность."""
        log.info("Загрузка Activity.csv")
        df = pd.read_csv(self.data_path / 'Activity.csv')
        
        df['obj_parsed'] = df['obj_new'].apply(self.safe_json_parse)
        
        activities = pd.DataFrame([
            {
                'id': row['id'],
                'tank_id': row['tank_id'],
                'activity_description': obj.get('description', ''),
                'activityType': obj.get('activityType', ''),
                'act_subtype': obj.get('act_subtype', ''),
            }
            for _, row in df.iterrows()
            for obj in [row['obj_parsed']]
        ])
        
        log.info(f"Активностей: {len(activities)}")
        return activities
    
    def create_time_features(self, df):
        """Временные признаки."""
        log.info("Создание временных признаков")
        
        df['measurement_year'] = df['createdAt'].dt.year
        df['measurement_month'] = df['createdAt'].dt.month
        df['measurement_quarter'] = df['createdAt'].dt.quarter
        df['days_since_epoch'] = (df['createdAt'] - pd.Timestamp('2000-01-01')).dt.days
        
        df['tank_age'] = df['measurement_year'] - df['yearStart'].fillna(df['measurement_year'] - 5)
        
        df = df.sort_values(['element_id', 'createdAt'])
        
        df['prev_delta'] = df.groupby('element_id')['delta'].shift(1)
        df['prev_measurement_date'] = df.groupby('element_id')['createdAt'].shift(1)
        
        df['days_since_prev_measurement'] = (df['createdAt'] - df['prev_measurement_date']).dt.days
        
        df['corrosion_rate'] = np.where(
            df['days_since_prev_measurement'] > 0,
            (df['delta'] - df['prev_delta']) / df['days_since_prev_measurement'],
            None
        )
        
        df['has_defect'] = (~df['defect_H'].isna()).astype(int)
        df['defect_volume'] = df['defect_H'].fillna(0) * df['defect_L'].fillna(0) * df['defect_W'].fillna(0)
        
        element_agg = df.groupby('element_id').agg({
            'delta': ['mean', 'std', 'min', 'max', 'count'],
            'measurement_year': ['min', 'max']
        }).reset_index()
        element_agg.columns = ['element_id', 'delta_mean_elem', 'delta_std_elem', 
                               'delta_min_elem', 'delta_max_elem', 'measurement_count',
                               'first_measurement_year', 'last_measurement_year']
        
        df = df.merge(element_agg, on='element_id', how='left', suffixes=('', '_agg'))
        df['element_lifetime'] = df['last_measurement_year'] - df['first_measurement_year']
        
        return df
    
    def process(self, output_path: Path):
        """
        Обработка данных.
        Args: output_path: путь для сохранения обработанных данных
        Returns: обработанный датасет
        """
        log.info("Обработка данных")
        
        tanks = self.load_and_parse_tank()
        elements = self.load_and_parse_element()
        measurements = self.load_and_parse_element_data()
        activities = self.load_and_parse_activity()
        
        log.info("Приведение типов")
        for col in ['tank_id']:
            if col in tanks.columns:
                tanks[col] = pd.to_numeric(tanks[col], errors='coerce')
        for col in ['tank_id', 'element_id']:
            if col in elements.columns:
                elements[col] = pd.to_numeric(elements[col], errors='coerce')
        for col in ['element_id', 'activityId']:
            if col in measurements.columns:
                measurements[col] = pd.to_numeric(measurements[col], errors='coerce')
        for col in ['id', 'tank_id']:
            if col in activities.columns:
                activities[col] = pd.to_numeric(activities[col], errors='coerce')
        
        log.info("Объединение отношений")
        df = measurements.copy()
        
        df = df.merge(elements, on='element_id', how='left')
        df = df.merge(tanks, on='tank_id', how='left')
        df = df.merge(activities, left_on='activityId', right_on='id', how='left', suffixes=('', '_activity'))
        
        df = self.create_time_features(df)
        
        cols_to_drop = ['id', 'id_activity', 'activityId', 'createdAt', 'prev_measurement_date']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        log.info(f"Датасет: {len(df)} x {len(df.columns)}")
        log.info(f"Delta: min={df['delta'].min():.2f}, max={df['delta'].max():.2f}, mean={df['delta'].mean():.2f}")
        
        df.to_csv(output_path, index=False)
        log.info(f"Сохранено: {output_path}")
        
        return df
