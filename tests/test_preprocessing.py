import unittest
from detector import PreprocessDataset
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = PreprocessDataset
    
    def test_remove_amp(self):
        text = "This &amp; that"
        expected = "This   that"
        self.assertEqual(self.preprocessor._remove_amp(self, text), expected)

    def test_remove_mentions(self):
        text = "Hello @user how are you?"
        expected = "Hello  how are you?"
        self.assertEqual(self.preprocessor._remove_mentions(self, text), expected)

    def test_remove_links(self):
        text = "Check this https://example.com"
        expected = "Check this  "
        self.assertEqual(self.preprocessor._remove_links(self, text), expected)

    def test_remove_punctuation(self):
        text = "Hello, world!"
        expected = "Hello world"
        self.assertEqual(self.preprocessor._remove_punctuation(self, text), expected)

    def test_tokenization(self):
        text = "This is a test sentence."
        expected = ["This", "is", "a", "test", "sentence", "."]
        self.assertEqual(self.preprocessor._tokenize(self, text), expected)

    def test_stopword_filtering(self):
        text_tokens = ["this", "is", "a", "test", "sentence"]
        expected = ["test", "sentence"]
        self.assertEqual(self.preprocessor._stopword_filtering(self, text_tokens), expected)

if __name__ == '__main__':
    unittest.main()
