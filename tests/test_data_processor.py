import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor
import config

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor(data_path=config.RAW_DATA_PATH)
    
    def test_safe_json_parse_valid(self):
        valid_json = '{"key": "value", "number": 42}'
        result = self.processor.safe_json_parse(valid_json)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['key'], 'value')
        self.assertEqual(result['number'], 42)
    
    def test_safe_json_parse_invalid(self):
        invalid_json = '{"key": invalid}'
        result = self.processor.safe_json_parse(invalid_json)
        self.assertEqual(result, {})
    
    def test_safe_json_parse_empty(self):
        for empty in [None, '', '""', np.nan]:
            result = self.processor.safe_json_parse(empty)
            self.assertEqual(result, {})
    
    def test_load_and_parse_tank(self):
        tanks = self.processor.load_and_parse_tank()
        self.assertIsInstance(tanks, pd.DataFrame)
        self.assertIn('tank_id', tanks.columns)
        self.assertIn('diameter', tanks.columns)
        self.assertIn('load_h_max', tanks.columns)
        self.assertIn('load_amplitude', tanks.columns)
        self.assertIn('soil_layers_count', tanks.columns)
        self.assertIn('soil_min_temp', tanks.columns)
        self.assertIn('rejDintK', tanks.columns)
        self.assertGreater(len(tanks), 0)
    
    def test_load_and_parse_element(self):
        elements = self.processor.load_and_parse_element()
        self.assertIsInstance(elements, pd.DataFrame)
        self.assertIn('element_id', elements.columns)
        self.assertIn('tank_id', elements.columns)
        self.assertGreater(len(elements), 0)
    
    def test_load_and_parse_element_data(self):
        measurements = self.processor.load_and_parse_element_data()
        self.assertIsInstance(measurements, pd.DataFrame)
        self.assertIn('delta', measurements.columns)
        self.assertIn('element_id', measurements.columns)
        self.assertGreater(len(measurements), 0)
        delta_not_null = measurements['delta'].notna()
        if delta_not_null.any():
            self.assertTrue(all(measurements.loc[delta_not_null, 'delta'] > 0))

if __name__ == '__main__':
    unittest.main()

