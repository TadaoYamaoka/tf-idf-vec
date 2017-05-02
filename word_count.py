import MeCab
import zenhan
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("--dictionary", "-d", type=str, help="mecab dictionary")
args = parser.parse_args()

mecab = MeCab.Tagger("" if not args.dictionary else "-d " + args.dictionary)

def wakati(str):
    words = []
    for line in mecab.parse(zenhan.z2h(str, mode=3).lower()).split("\n"):
        cols = line.split("\t")
        if len(cols) >= 2:
            c = cols[1].split(",")
            if not c[0] in ["助詞", "助動詞", "副詞", "記号"] and not c[1] in ["非自立", "代名詞"]:
                words.append(cols[0])
    return words

questions = []
for line in open(args.input, "r", encoding="utf-8"):
    cols = line.strip().split('\t')
    questions.append(wakati(cols[0]))

words = defaultdict(lambda: 0)
for question in questions:
    for word in question:
        words[word] += 1

for k,v in sorted(words.items(), key=lambda x:x[1]):
    print(k, v)
