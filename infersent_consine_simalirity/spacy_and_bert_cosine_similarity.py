from test import compute_train_test_split
import spacy
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")
import sklearn.linear_model as lm
import sklearn.svm as svm

def spacy_is_trigger(s, t):
    s = nlp(s)
    t = nlp(t)
    return s.similarity(t) > 0.92
    # 4 / 50 false positives
    # 46 / 50 true negatives
    # 13 / 14 true positives
    # 1 / 14 false negatives

compute_train_test_split(spacy_is_trigger)

print()
print()

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('stsb-bert-large')
def bert_large_is_trigger(s, t):
    s = model.encode(s)
    t = model.encode(t)
    cosine_score = util.pytorch_cos_sim(s, t)
    return cosine_score > 0.8
    # 0 / 50 false positives
    # 50 / 50 true negatives
    # 13 / 14 true positives
    # 1 / 14 false negatives


compute_train_test_split(bert_large_is_trigger)