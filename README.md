# Assistant Intent Classification

The classifer repo is split into multiple models:
- fuzzy_similarity: Tests fuzzy checking of words with text
- infersent_cosine_similarity: Test trigger similarity via HuggingFace and Infersent
- location_recognition: Location recognition for weather
- many_intent_classifier: Trying to classify lots of intents at a time
- qa_context_evaluation: Tests using qa models for context
- shopping_classifier: Finds shopping items mentioned

### Running the notebooks
Download Glove for word embeddings
```sh
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
```

and 
Download infersent for sentence embeddings
```sh
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```

Run the classifier notebook classifier.ipynb to load and view the data.

### Running the Server 
Current Accuracy:

Download deepspeech-0.7.3-models from deepspeech directory. The files that should be included are 
```
├── deepspeech-0.7.3-models.pbmm
├── lm.binary
├── output_graph.pb
├── output_graph.pbmm
├── output_graph.tflite
└── trie
```

## Download + Accessing Colabs
Copy the model files for the relevant classification system
https://drive.google.com/drive/folders/1gzlJk7iL88Fpe2OzW7NQI_9iSyIqx39B?usp=sharing
Download ffc_keras_model and infraset_model.torch from the notebook 
