import json
import unittest
from parameterized import parameterized
from nltk.tokenize import word_tokenize

from taxon_parser import TaxonTagger, TaxonFormatter, TaxonParser


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand([
        ("Stiboges nymphidia Butler, 1876", "Stiboges nymphidia Butler, 1876", "Stiboges"),
        ("Takashia nana", "Takashia nana", "Takashia"),
        ("Zemeros flegyas Cramer, 1780", "Zemeros flegyas Cramer, 1780", "Zemeros"),
        ("Anthocharis bambusarum Oberthür, 1876", "Anthocharis bambusarum Oberthür, 1876", "Anthocharis"),
        ("Anthocharis bieti Oberthür, 1884", "Anthocharis bieti Oberthür, 1884", "Anthocharis"),
        ("Anthocharis cardamines (Linnaeus, 1758)", "Anthocharis cardamines (Linnaeus, 1758)", "Anthocharis"),
        ("Anthocharis scolymus Butler, 1866", "Anthocharis scolymus Butler, 1866", "Anthocharis"),
        ("Aporia acraea (Oberthür, 1885)", "Aporia acraea (Oberthür, 1885)", "Aporia"),
        ("Aporia agathon (Gray, 1831)", "Aporia agathon (Gray, 1831)", "Aporia"),
        ("Aporia bernardi Koiwaya, 1989", "Aporia bernardi Koiwaya, 1989", "Aporia"),
        ("Aporia bieti Oberthür, 1884", "Aporia bieti Oberthür, 1884", "Aporia"),
        ("Aporia chunhaoi Hu, Zhang & Yang, 2021", "Aporia chunhaoi Hu, Zhang & Yang, 2021", "Aporia"),
        ("Aporia crataegi (Linnaeus, 1758)", "Aporia crataegi (Linnaeus, 1758)", "Aporia"),
        ("Aporia delavayi (Oberthür, 1890)", "Aporia delavayi (Oberthür, 1890)", "Aporia"),
        ("Aporia goutellei (Oberthür, 1886)", "Aporia goutellei (Oberthür, 1886)", "Aporia"),
        ("Aporia hippia (Bremer, 1861)", "Aporia hippia (Bremer, 1861)", "Aporia"),
        ("Aporia kamei Koiwaya, 1989", "Aporia kamei Koiwaya, 1989", "Aporia"),
        ("Aporia kanekoi Koiwaya, 1989", "Aporia kanekoi Koiwaya, 1989", "Aporia"),
        ("Papilio karna carnatus Rothschild, 1895", "Papilio karna carnatus Rothschild, 1895", "Papilio"),
        ("Papilio (Achillides) karna carnatus Rothschild, 1895", "Papilio (Achillides) karna carnatus Rothschild, 1895",
         "Papilio"),
        ("Papilio karna Felder & Felder, 1864", "Papilio karna Felder & Felder, 1864", "Papilio"),
        ("Papilio (Achillides) karna Felder & Felder, 1864", "Papilio (Achillides) karna Felder & Felder, 1864",
         "Papilio"),
        ("Papilio karna C. & R. Felder, 1864", "Papilio karna C. & R. Felder, 1864", "Papilio"),
        ("Papilio karna discordia De_nicéville, [1893]", "Papilio karna discordia De_nicéville, [1893]", "Papilio"),
        ("Papilio karna discordia de Nicéville, [1893]", "Papilio karna discordia de Nicéville, [1893]", "Papilio"),
        ("Papilio (Achillides) karna discordia De_nicéville, [1893]",
         "Papilio (Achillides) karna discordia De_nicéville, [1893]", "Papilio"),
        ("Papilio (Achillides) karna discordia de Nicéville, [1893]",
         "Papilio (Achillides) karna discordia de Nicéville, [1893]", "Papilio"),
    ])
    def test_genus(self, name, taxon, expected):
        tagger = TaxonTagger()
        parser = TaxonParser()
        fmt = TaxonFormatter()

        tagged_text = tagger.tag(word_tokenize(parser.ligate(taxon)))
        trees = parser.parse(tagged_text)
        json_string = fmt.json(list(trees)[0])
        self.assertEqual(json.loads(json_string)["genus"], expected)
