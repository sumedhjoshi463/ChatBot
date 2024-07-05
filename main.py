from flask import Flask, jsonify
from tensorflow.keras.models import load_model
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
model = load_model("Model/chatbot_model.h5")

intents = json.loads(open('Data/data.json', encoding='utf8').read())
words = pickle.load(open('Data/words.pkl', 'rb'))
classes = pickle.load(open('Data/classes.pkl', 'rb'))

app = Flask(__name__)

def clean_sentence(sent):
    sent_words = nltk.word_tokenize(sent)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]
    return sent_words

def create_bag_of_words(sentence, words, show_details=True):
    sentence = clean_sentence(sentence) 
    bag = [0] * len(words)
    for s in sentence:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("word found in bag")
    return np.array(bag)

def predict_class(sent, model):
    p = create_bag_of_words(sent, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    error_threshold = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return result_list

def get_response(cls, intents):
    tag = cls[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def decrypt(msg):
    return msg.replace('+', ' ')

def chatbot_response(msg):
    cls = predict_class(msg, model)
    res = get_response(cls, intents)
    print("final response =", res)
    return res

@app.route("/query/<sentence>")
def query_chatbot(sentence):
    decrypted_msg = decrypt(sentence)
    response = chatbot_response(decrypted_msg)
    return jsonify({"top": {"res": response}})

if __name__ == "__main__":
    print("Starting Flask server...")

    app.run()
