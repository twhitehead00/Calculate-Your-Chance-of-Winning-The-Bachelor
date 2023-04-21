import json
import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np

from app import app
from model_export import preprocess, predict, clean_column_names

class TestApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_predict_endpoint(self):
        with patch('app.model.preprocess') as mock_preprocess, \
             patch('app.model.predict') as mock_predict:
            # mock the model preprocess and predict methods
            mock_preprocess.return_value = 'processed_data'
            mock_predict.return_value = 1
            
            # prepare a sample input
            input_data = {'feature1': 0.5, 'feature2': 0.8}
            input_json = json.dumps(input_data)
            
            # send a POST request to the /predict endpoint
            response = self.app.post('/predict', 
                                     data=input_json, 
                                     content_type='application/json')
            
            # check the response
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), 1)
            
            # check that the model methods were called with the correct data
            mock_preprocess.assert_called_once_with(input_data)
            mock_predict.assert_called_once_with('processed_data')




class TestPredictAndPreprocess(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.predict.return_value = [5] # Mock prediction value
        
    def test_predict(self):
        # Test prediction when prediction is greater than 1_on_1_week
        data = {'name': '', 'season': 1, 'age': 25, '1_on_1_week': 3, 'hometown': 'NE', 'note': '0', 'job_category': 'CORPORATE',
                'race': 'White', 'FIR': 'No', 'joke_entrance': 'Regular'}
        #data_array = np.array(list(data.values())).reshape(1, -1)  # Convert data to 2D array
        data = preprocess(data) # Preprocess data
        self.assertEqual(predict(data), 5)

        # Test prediction when prediction is less than 1_on_1_week
        self.mock_model.predict.return_value = [2] # Mock prediction value
        self.assertEqual(predict(data), 3)
            
    def test_preprocess(self):
        data = {'name': '', 'season': 1, 'age': 25, '1_on_1_week': 3, 'hometown': 'NE', 'note': '0', 'job_category': 'CORPORATE',
                'race': 'White', 'FIR': 'No', 'joke_entrance': 'Regular'}
        expected_data = pd.DataFrame({'season': 1, 'age': 25, '1_on_1': 3, 'INTERNATIONAL': 0, 'NE': 1, 'NW': 0, 'SE': 0, 'SW': 0,
                         '0': 1, 'DQ': 0, 'quit': 0, 'CORPORATE': 1, 'OTHER': 0, 'POLITICS': 0, 'TRADITIONAL': 0, 
                         'TRADES': 0, 'Asian': 0, 'Black': 0, 'Hispanic': 0, 'Middle eastern': 0, 'White': 1, 
                         'No': 1, 'Yes': 0, 'Gimmick': 0, 'Regular': 1})
        self.assertEqual(preprocess(data), expected_data)