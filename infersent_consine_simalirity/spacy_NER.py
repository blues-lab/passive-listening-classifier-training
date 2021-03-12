import spacy

nlp = spacy.load("en_core_web_lg")


with open("context.txt") as f:
    context = f.readlines()

for i in range(len(context)):
    st = context[i]
    print()
    print(st)
    most_recent_GPE = ""
    for ent in nlp(st).ents:
        if ent.label_ == "GPE":
            #print(ent, end="")
            most_recent_GPE = ent
    print("=== Most recently named GPE:", most_recent_GPE)
    print()