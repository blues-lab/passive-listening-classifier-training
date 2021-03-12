from tqdm import tqdm
import random

with open("samples.txt") as f:
    sentences = f.readlines()

with open("triggers.txt") as f:
    triggers = f.readlines()

with open("neg_distractors.txt") as f:
    distractors = f.readlines()

def compute_accuracy(sents, trigs, is_trigger):
    identified_trigger = 0
    random.shuffle(sents)
    for sent in tqdm(sents):
        for trig in trigs:
            if is_trigger(sent, trig):
                identified_trigger += 1
                break
    return identified_trigger, identified_trigger/len(sents)

def compute_train_test_split(is_trigger):
    print("Computing false positives:")
    random.shuffle(triggers)
    train_triggers = triggers[:int(len(triggers)*.8)]
    test_triggers = triggers[int(len(triggers)*.8): len(triggers)]
    print(compute_accuracy(sentences[:50], train_triggers, is_trigger))

    print("Compute true positives")
    print(compute_accuracy(test_triggers, train_triggers, is_trigger))

    print("Compute neg distractors rate")
    print(compute_accuracy(distractors, train_triggers, is_trigger))
