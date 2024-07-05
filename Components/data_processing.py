import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np 

lemmatizer = WordNetLemmatizer()


words = []
classes = []
documents = []
ignore_words = ['?','@','!','$']

data_file = open('data.json').read()
intents = json.loads(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize 
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        #adding documents
        documents.append((w,intent['tag']))

        #adding classes to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


pickle.dump(words, open('words.pkl','wb'))

pickle.dump(classes, open('classes.pkl','wb'))

pickle.dump(documents, open('documents.pkl','wb'))

print("pickle files created successfully")




