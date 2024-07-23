import unittest
import numpy as np
import pandas as pd
from disease_prediction import load_model, pred_result

class TestDiseasePrediction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the model
        cls.model = load_model('../notebooks/ExtraTrees.pkl')
        
        # Load the symptom description and precaution datasets
        cls.sd = pd.read_csv('../data/symptom_Description.csv')
        cls.sp = pd.read_csv('../data/symptom_precaution.csv')
        
        # Sample input data
        cls.sample_symptoms = ["chest_pain", "phlegm", "runny_nose", "high_fever", "throat_irritation", "congestion", "redness_of_eyes"]
        cls.sample_input = pd.Series([0] * len(cls.sample_symptoms), index=cls.sample_symptoms)
        cls.sample_input.loc[cls.sample_symptoms] = 1
        cls.sample_input = cls.sample_input.to_numpy().reshape(1, -1)

    def test_prediction_results(self):
        results = pred_result(self.model, self.sample_input, self.sd, self.sp)
        
        # Check if the results contain 5 predictions
        self.assertEqual(len(results), 5)
        
        for result in results:
            # Check if each result has the necessary keys
            self.assertIn("Disease Name", result)
            self.assertIn("Probability", result)
            self.assertIn("Disease Description", result)
            self.assertIn("Recommended Things to do at home", result)
            
            # Check if probability is between 0 and 1
            self.assertGreaterEqual(result["Probability"], 0)
            self.assertLessEqual(result["Probability"], 1)
    
    def test_model_loading(self):
        model = load_model('../notebooks/ExtraTrees.pkl')
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
