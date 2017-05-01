from gensim.models.wrappers.fasttext import FastText
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import zenhan
import numpy as np
import argparse

DIM = 300

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

def part_minus(v):
    # 正と負で別のベクトルにする
    tmp_v = np.zeros(DIM*2)
    for i in range(DIM):
        if v[i] >= 0:
            tmp_v[i] = v[i]
        else:
            tmp_v[i*2] = -v[i]
    return tmp_v

questions_vec = []
tf_vecs = []
df_vec = np.zeros(DIM*2)
for question in questions:
    vec = np.zeros(DIM*2)
    maxvec = np.zeros(DIM*2)
    n = 0
    for word in question:
        try:
            word_vec = part_minus(model[word])
            vec += word_vec
            n += 1
        except:
            continue
        maxvec = np.maximum(word_vec, maxvec)
    tf_vecs.append(vec / n)
    df_vec += maxvec

idf_vec = np.log(len(questions) / (df_vec + 1))
tfidf_vecs = []
for tf_vec in tf_vecs:
    tfidf_vecs.append(tf_vec * idf_vec)

while True:
    line = input("> ")
    if not line:
        break

    words = mecab.parse(zenhan.z2h(line, mode=3).lower()).strip().split(" ")
    vec = np.zeros(DIM*2)
    n = 0
    for word in words:
        try:
            vec += part_minus(model[word])
            n += 1
        except:
            continue
    tf_vec = vec / n

    sims = cosine_similarity([tf_vec * idf_vec], tfidf_vecs)
    index = np.argmax(sims)
    print(">", words)
    print(questions[index], sims[0, index])
    #print()
    #print(answers[index])
    #print()
    print(questions[index-2], sims[0, index-2])
    print(questions[index-3], sims[0, index-3])
    print(questions[index-4], sims[0, index-4])
    print(questions[index-5], sims[0, index-5])
    print()