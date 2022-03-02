import unittest

from sacremoses import MosesTokenizer
from source.features import ControlDivisionByZero, \
    WordLengthRatio, \
    CharLengthRatio, \
    LevenshteinRatio, \
    DependencyTreeDepthRatio, \
    WordRankRatio



class TestWordLengthRatioFeature(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MosesTokenizer(lang='en')
        self.original_text = "For example , King Bhumibol was born on Monday , " \
                             "so on his birthday throughout Thailand will be decorated with yellow color ."
        self.simple_text = "All over Thailand the color yellow will be used to celebrate King Bhumibal."


    def test_control_division_by_zero(self):
        self.assertEqual(ControlDivisionByZero(4, 0), 0)
        self.assertEqual(ControlDivisionByZero(2, 1), 2)
        self.assertEqual(ControlDivisionByZero(3.2, 2), 1.6)

    def test_word_length_ratio(self):
        result_ratio = WordLengthRatio().calculate_ratio(self.simple_text, self.original_text)

        self.assertEqual(result_ratio, 0.61)


class TestCharLengthRatioFeature(unittest.TestCase):

    def setUp(self):
        self.original_text = "For example , King Bhumibol was born on Monday , " \
                             "so on his birthday throughout Thailand will be decorated with yellow color ."
        self.simple_text = "All over Thailand the color yellow will be used to celebrate King Bhumibal."

    def test_char_length_ratio(self):
        result_ratio = CharLengthRatio().calculate_ratio(self.simple_text, self.original_text)
        self.assertEqual(result_ratio, 0.6)


# Currently only testing a method from Levenshtein library. Could change in the future.
class TestLevenshteinRatioFeature(unittest.TestCase):

    def setUp(self):
        self.original_text = "For example , King Bhumibol was born on Monday , " \
                             "so on his birthday throughout Thailand will be decorated with yellow color ."
        self.simple_text = "All over Thailand the color yellow will be used to celebrate King Bhumibal."

    def test_levenshtein_ratio(self):
        result_ratio = LevenshteinRatio().calculate_ratio(self.simple_text, self.original_text)
        self.assertEqual(result_ratio, 0.37)


class TestDependencyTreeDepthRatio(unittest.TestCase):

    def setUp(self):

        self.feature_model = DependencyTreeDepthRatio()
        self.simple_text = "All over Thailand the color yellow will be used to celebrate King Bhumibal."

    def test_dependency_tree_depth_ratio(self):
        tree_depth = self.feature_model.get_dependency_tree_depth(self.simple_text)
        self.assertEqual(tree_depth, 4)


class TestWordRankRatio(unittest.TestCase):

    def setUp(self):
        self.feature_model = WordRankRatio()
        self.simple_text = "All over Thailand the color yellow will be used to celebrate King Bhumibal."

    def test_get_lexical_complexity_score(self):
        lexical_score = self.feature_model.get_lexical_complexity_score(self.simple_text)
        self.assertAlmostEqual(lexical_score, 7.6, 2)

    def tearDown(self):
        self.feature_model = None




if __name__ == '__main__':
    unittest.main()
