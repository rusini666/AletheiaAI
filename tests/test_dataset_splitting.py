import unittest
from detector import get_train_val_test_data
import pandas as pd
import numpy as np

class TestDatasetSplitting(unittest.TestCase):

    def setUp(self):
        self.dataset_path = "/Users/rusinitharagunarathne/Documents/AletheiaAI/data/subtaskA_dataset_monolingual.jsonl"
    
    def test_split_ratios(self):
        train, val, test = get_train_val_test_data(self.dataset_path)
        total = len(train) + len(val) + len(test)
        self.assertAlmostEqual(len(train) / total, 0.7, places=1, msg="Train split should be ~70%")
        self.assertAlmostEqual(len(val) / total, 0.2, places=1, msg="Validation split should be ~20%")
        self.assertAlmostEqual(len(test) / total, 0.1, places=1, msg="Test split should be ~10%")

    def test_label_balance(self):
        train, val, test = get_train_val_test_data(self.dataset_path)
        train_ratio = train['label'].value_counts(normalize=True)
        val_ratio = val['label'].value_counts(normalize=True)
        test_ratio = test['label'].value_counts(normalize=True)

        self.assertAlmostEqual(train_ratio[1], val_ratio[1], places=2, msg="Class distribution should be similar across splits")
        self.assertAlmostEqual(train_ratio[1], test_ratio[1], places=2, msg="Class distribution should be similar across splits")

if __name__ == '__main__':
    unittest.main()

