from gensim.models.wrappers.fasttext import FastText
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import zenhan
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("model", type=str)
parser.add_argument("--dictionary", "-d", type=str, help="mecab dictionary")
args = parser.parse_args()

mecab = MeCab.Tagger("-Owakati" + ("" if not args.dictionary else " -d " + args.dictionary))

model = FastText.load_fasttext_format(args.model)

questions = []
answers = []
for line in open(args.input, "r", encoding="utf-8"):
    cols = line.strip().split('\t')
    questions.append(mecab.parse(zenhan.z2h(cols[0], mode=3).lower()).strip().split(" "))
    answers.append(cols[1])

questions_vec = []
tf_vecs = []
df_vec = np.zeros(300)
for question in questions:
    vec = np.zeros(300)
    maxvec = np.zeros(300)
    for word in question:
        try:
            vec += model[word]
        except:
            continue
        maxvec = np.maximum(abs(model[word]), maxvec)
    tf_vecs.append(vec / sum(abs(vec)))
    df_vec += maxvec

idf_vec = np.log(len(questions) / df_vec)
tfidf_vecs = []
for tf_vec in tf_vecs:
    tfidf_vecs.append(tf_vec * idf_vec)

while True:
    line = input("> ")
    if not line:
        break

    words = mecab.parse(zenhan.z2h(line, mode=3).lower()).strip().split(" ")
    vec = np.zeros(300)
    for word in words:
        try:
            vec += model[word]
        except:
            continue
    tf_vec = vec / sum(abs(vec))

    sims = cosine_similarity(tf_vec * idf_vec, tfidf_vecs)
    index = np.argmax(sims)
    print(questions[index], sims[0, index])
    #print()
    #print(answers[index])
    #print()
    print(questions[index-2], sims[0, index-2])
    print(questions[index-3], sims[0, index-3])
    print()