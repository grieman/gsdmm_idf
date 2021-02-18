from unittest import TestCase
from gsdmm.mgp import MovieGroupProcess
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

class TestGSDMM(TestCase):
    '''This class tests the Panel data structures needed to support the RSK model'''

    def setUp(self):
        numpy.random.seed(47)

    def tearDown(self):
        numpy.random.seed(None)

    def test_grades(self):

        grades = list(map(list, [
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "D",
            "D",
            "F",
            "F",
            "P",
            "W"
        ]))

        grades = grades + grades + grades + grades + grades
        mgp = MovieGroupProcess(K=100, n_iters=100, alpha=0.001, beta=0.01)
        y = mgp.fit(grades)
        self.assertEqual(len(set(y)), 7)
        for words in mgp.cluster_word_distribution:
            self.assertTrue(len(words) in {0,1}, "More than one grade ended up in a cluster!")

    def test_short_text(self):
        # there is no perfect segmentation of this text data:
        texts = [
            "where the red dog lives",
            "red dog lives in the house",
            "blue cat eats mice",
            "monkeys hate cat but love trees",
            "green cat eats mice",
            "orange elephant never forgets",
            "orange elephant must forget",
            "monkeys eat banana",
            "monkeys live in trees",
            "elephant",
            "cat",
            "dog",
            "monkeys"
        ]

        texts = [text.split() for text in texts]
        mgp = MovieGroupProcess(K=30, n_iters=100, alpha=0.2, beta=0.01)
        y = mgp.fit(texts)
        self.assertTrue(len(set(y))<10)
        self.assertTrue(len(set(y))>3)

    def test_short_text_idf(self):
        # there is no perfect segmentation of this text data:
        texts = [
            "where the red dog lives",
            "red dog lives in the house",
            "blue cat eats mice",
            "monkeys hate cat but love trees",
            "green cat eats mice",
            "orange elephant never forgets",
            "orange elephant must forget",
            "monkeys eat banana",
            "monkeys live in trees",
            "elephant",
            "cat",
            "dog",
            "monkeys"
        ]

        texts = [text.split() for text in texts]
        mgp = MovieGroupProcess(K=30, n_iters=100, alpha=0.2, beta=0.01, vectorizer=TfidfVectorizer())
        y = mgp.fit(texts)
        self.assertTrue(len(set(y))<10)
        self.assertTrue(len(set(y))>3)

        # make sure that cluster word calculations work
        lst_idf = [[(k, v*mgp.idf_dict[k]) if k in mgp.idf_dict.keys() else (k, 0) for k, v in mgp.cluster_word_distribution[i].items()] for i in range(len(mgp.cluster_word_distribution))]
        lst_idf = [sorted(group, key = lambda x: x[1], reverse=True)[0:5] for group in lst_idf]
        self.assertTrue(sum([len(x) > 0 for x in lst_idf]) == len(set(y)))

        # The vectorizer should put more emphasis on unique words like must and never, and keep the orange elephant lines seperate
        mgp_novect = MovieGroupProcess(K=30, n_iters=100, alpha=0.2, beta=0.01)
        y_novect = mgp_novect.fit(texts)
        self.assertTrue(len(set(y))>=len(set(y_novect)))