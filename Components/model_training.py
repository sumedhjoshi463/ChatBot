import pickle
import os
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random

import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD




lemmatizer = WordNetLemmatizer()


# Data preperation for training


words = pickle.load(open('Data/words.pkl', 'rb'))
classes = pickle.load(open('Data/classes.pkl', 'rb'))
documents = pickle.load(open('Data/documents.pkl','rb'))

training = []
output_empty = [0]*len(classes)

for doc in documents:
    #initializing the bag of words 
    bag = []
    # list of tokenized words for the patterns
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]


    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1


    training.append([bag,output_row])

random.shuffle(training)

# Convert the lists to NumPy arrays separately
train_x = np.array([np.array(pair[0]) for pair in training])
train_y = np.array([np.array(pair[1]) for pair in training]) # Convert each 'bag' and 'output_row' to NumPy arrays individually



print("length of training sample x is: ", len(train_x[0]) )
print("length of training sample y is: ", len(train_y[0]) )

print("train x")
print(train_x[0:5])
print("train y")
print(train_y[0:5])




model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov =True)
model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=10, verbose=True)
model.save('chatbot_model.h5',hist)

print("model created!!")

