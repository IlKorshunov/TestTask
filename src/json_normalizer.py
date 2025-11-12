import pandas as pd
import json
import os
import logging

log = logging.getLogger(__name__)

class JSONNormalizer:
    """
    Преобразование к реляционной бд
    """
    def __init__(self, raw_path: str, normalized_path: str):
        self.raw_path = raw_path
        self.normalized_path = normalized_path
        os.makedirs(self.normalized_path, exist_ok=True)

    def _safe_json_loads(self, x, row_info):
        """Парсинг JSON."""
        if not isinstance(x, str) or not x.strip():
            return {}
        try:
            return json.loads(x)
        except json.JSONDecodeError as e:
            log.warning(f"Ошибка парсинга JSON в {row_info}: {e}")
            return {}

    def _normalize_and_save(self, filename: str, json_col: str, record_path: list = None, meta: list = None):
        """Нормализация и сохранение."""
        log.info(f"Нормализация: {filename}")
        df = pd.read_csv(os.path.join(self.raw_path, filename))
        
        data = [self._safe_json_loads(row[json_col], f"file={filename}, index={i}") for i, row in df.iterrows()]
        
        flat_data = [d for d in data if not record_path or record_path[0] not in d]
        df_flat_normalized = pd.json_normalize(flat_data, errors='ignore')
        
        flat_cols = [c for c in df.columns if c != json_col]
        
        df_flat = df[flat_cols].join(df_flat_normalized, rsuffix='_json')
        
        df_flat.to_csv(os.path.join(self.normalized_path, f"{filename.split('.')[0]}_flat.csv"), index=False)
        log.info(f"Сохранено: {filename.split('.')[0]}_flat.csv")

        if record_path:
            nested_data = [d for d in data if record_path[0] in d]
            if nested_data:
                df_nested = pd.json_normalize(nested_data, record_path=record_path, meta=meta or [], errors='ignore')
                nested_filename = f"{filename.split('.')[0]}_{record_path[0]}.csv"
                df_nested.to_csv(os.path.join(self.normalized_path, nested_filename), index=False)
                log.info(f"Сохранено: {nested_filename}")

    def run(self):
        log.info("Нормализация JSON")
        self._normalize_and_save('tank.csv', 'obj', record_path=['soilLayers'], meta=['name', 'd'])
        self._normalize_and_save('element.csv', 'obj')
        self._normalize_and_save('ElementData.csv', 'obj')
        self._normalize_and_save('Activity.csv', 'obj_new')
        log.info("Готов json_normalizer.csv")
