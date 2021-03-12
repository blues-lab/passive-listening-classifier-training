from test import compute_train_test_split
import spacy
# python -m spacy download en_core_web_lg

nlp = spacy.load("en_core_web_lg")
import sklearn.linear_model as lm
import sklearn.svm as svm

def spacy_is_trigger(s, t):
    s = nlp(s)
    t = nlp(t)
    return s.similarity(t) > 0.93

compute_train_test_split(spacy_is_trigger)

print()
print()

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('stsb-distilbert-base')
def stsb_distilbert_is_trigger(s, t):
    s = model.encode(s)
    t = model.encode(t)
    cosine_score = util.pytorch_cos_sim(s, t)
    return cosine_score > 0.73


compute_train_test_split(stsb_distilbert_is_trigger)



print()
print()

model = SentenceTransformer('nli-distilbert-base')
def nli_distilbert_is_trigger(s, t):
    s = model.encode(s)
    t = model.encode(t)
    cosine_score = util.pytorch_cos_sim(s, t)
    return cosine_score > 0.75


compute_train_test_split(nli_distilbert_is_trigger)