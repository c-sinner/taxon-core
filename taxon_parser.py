import json
import re
import sys

import nltk
from nltk.tokenize import word_tokenize

tests = [
    "Stiboges nymphidia Butler, 1876",
    "Takashia nana",
    "Zemeros flegyas Cramer, 1780",
    "Anthocharis bambusarum Oberthür, 1876",
    "Anthocharis bieti Oberthür, 1884",
    "Anthocharis cardamines (Linnaeus, 1758)",
    "Anthocharis scolymus Butler, 1866",
    "Aporia acraea (Oberthür, 1885)",
    "Aporia agathon (Gray, 1831)",
    "Aporia bernardi Koiwaya, 1989",
    "Aporia bieti Oberthür, 1884",
    "Aporia chunhaoi Hu, Zhang & Yang, 2021",
    "Aporia crataegi (Linnaeus, 1758)",
    "Aporia delavayi (Oberthür, 1890)",
    "Aporia goutellei (Oberthür, 1886)",
    "Aporia hippia (Bremer, 1861)",
    "Aporia kamei Koiwaya, 1989",
    "Aporia kanekoi Koiwaya, 1989",
    "Papilio karna carnatus Rothschild, 1895",
    "Papilio (Achillides) karna carnatus Rothschild, 1895",
    "Papilio karna Felder & Felder, 1864",
    "Papilio (Achillides) karna Felder & Felder, 1864",
    "Papilio karna C. & R. Felder, 1864",
    "Papilio karna discordia De_nicéville, [1893]",
    "Papilio karna discordia de Nicéville, [1893]",
    "Papilio (Achillides) karna discordia De_nicéville, [1893]",
    "Papilio (Achillides) karna discordia de Nicéville, [1893]",
#    "Gulliveria D'Abrera & Bálint, 2001",
]

taxon_grammar = nltk.CFG.fromstring("""
T -> N A | N | N PA
N -> G | S | G S | G S SSP | S SSP
G -> GG | GG PL GS PR
GG -> 'CAP'
GS -> 'CAP'
S -> 'LOW'
SSP -> 'LOW'
PA -> PL A PR
A -> AA C Y
AA -> NAME | NAME CF NAME | NAME C NAME CF NAME | NAME C NAME C NAME CF NAME | NAME C NAME C NAME C NAME C NAME CF NAME
NAME -> CAP | CAP CAP | CAP CAP CAP
CAP -> 'CAP'
C -> 'COMB'
CF -> 'FCOMB'
Y -> YY | YB
YB -> BL YY BR
YY -> 'YEAR'
PL -> 'PARL'
PR -> 'PARR'
BL -> 'BRAL'
BR -> 'BRAR'
""")

regexp_tagger = nltk.RegexpTagger(
    [
        (r"^1|2|3[0-9]{3}$", "YEAR"),
        (r"^,$", "COMB"),
        (r"^&$", "FCOMB"),
        (r"^[A-Z][\'\u00C0-\u1FFF\u2C00-\uD7FF\w]*(\.)?$", "CAP"),
        (r"^[\u00C0-\u1FFF\u2C00-\uD7FF\w]+$", "LOW"),
        (r"^[A-Z][a-z]+oidea$", "SUPERFAMILY"),
        (r"^[A-Z][a-z]+dae$", "FAMILY"),
        (r"^[A-Z][a-z]+nae$", "SUBFAMILY"),
        (r"^[A-Z][a-z]+ini$", "TRIBE"),
        (r"^[A-Z][a-z]+ina$", "SUBTRIBE"),
        (r"^\($", "PARL"),
        (r"^\)$", "PARR"),
        (r"^\[$", "BRAL"),
        (r"^\]$", "BRAR"),
    ])

parser = nltk.ChartParser(taxon_grammar)


def ligate(input: str):
    input = re.sub(r"de ([A-Z])", r"De_\g<1>", input)
    input = re.sub(r"d'([A-Z])", r"D___\g<1>", input)
    return input


def to_json(t):
    data = {
        "genus": "",
        "subgenus": "",
        "species": "",
        "subspecies": "",
        "authors": [],
        "year": "",
        "yearBracketed": False,
        "originalGenus": None
    }

    def get_by_name(name, default=""):
        l = list(t.subtrees(filter=lambda x: x.label() == name))
        return "".join(l[0]) if l else default

    data["genus"] = get_by_name('GG')
    data["subgenus"] = get_by_name('GS')
    data["species"] = get_by_name('S')
    data["subspecies"] = get_by_name('SSP')
    data["year"] = get_by_name('YY')
    data["authors"] = [name[0] for node in list(t.subtrees(filter=lambda x: x.label() == 'NAME')) for name in node]
    data["yearBracketed"] = len(list(t.subtrees(filter=lambda x: x.label() == 'YB'))) > 0
    data["originalGenus"] = len(list(t.subtrees(filter=lambda x: x.label() == 'PA'))) < 1 if list(t.subtrees(filter=lambda x: x.label() == 'A')) else None
    return json.dumps(data, default=str, indent=4)


def replace_values(tree, new_values):
    assert len(new_values) == len(tree.leaves())

    for idx, new_val in enumerate(new_values):
        pos = tree.leaf_treeposition(idx)
        exec("tree" + "".join(f"[{_}]" for _ in pos) + f"='{new_val}'")
    return tree


# for species in tests:
#     tagged_text = regexp_tagger.tag(word_tokenize(ligate(species)))
#     only_tags = [tag for text, tag in tagged_text]
#     only_text = [text for text, tag in tagged_text]
#     print(f"SPECIES: {species}")
#     # print(only_tags)
#     for tree in parser.parse(only_tags):
#         tree = replace_values(tree, only_text)
#         tree.pretty_print(highlight=[("GG", "red")])
#         print("")
#         print("TXT:")
#         print(tree)
#         print("")
#         print("JSON:")
#         print(to_json(tree))


from nltk.tokenize.treebank import TreebankWordDetokenizer

tagged_text = regexp_tagger.tag(word_tokenize(ligate(TreebankWordDetokenizer().detokenize(sys.argv[1:]))))
only_tags = [tag for text, tag in tagged_text]
only_text = [text for text, tag in tagged_text]
print(f"SPECIES: {TreebankWordDetokenizer().detokenize(sys.argv[1:])}")
# print(only_tags)
for tree in parser.parse(only_tags):
    tree = replace_values(tree, only_text)
    tree.pretty_print(highlight=[("GG", "red")])
    print("")
    print("TXT:")
    print(tree)
    print("")
    print("JSON:")
    print(to_json(tree))


# sent = ['Mus', 'musculus', 'arctos', 'Linnaeus', '1758']
# parser = nltk.ChartParser(groucho_grammar)
# for tree in parser.parse(sent):
#     print(tree)
