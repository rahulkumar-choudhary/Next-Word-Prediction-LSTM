import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
import matplotlib


from numpy.core.multiarray import dtype
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from pylab import rcParams

matplotlib.use('agg')
np.random.seed(42)

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5


#Loading the data
path = '1661-0.txt'
text = open(path, "r", encoding='utf-8').read().lower()
print ('Corpus length: ',len(text))

#Preprocessing
#Finding all the unique characters in the corpus
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print ("unique chars: ",len(chars))


#Cutting the corpus into chunks of 39 chars, spacing the sequences by 3 characters
#We will additionally store the next character (the one we need to predict) for every sequence

SEQUENCE_LENGTH = 39
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i:i+SEQUENCE_LENGTH])
    next_chars.append(text[i+SEQUENCE_LENGTH])
print ('num training examples: ',len(sentences))



#Generating features and labels.
#Using previously generated sequences and characters that need to be predicted to create one-hot encoded vectors

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
    
    
#Building the model

model = Sequential();
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))




#Training
optimizer = RMSprop(lr= 0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=20, shuffle=True).history


#Saving
model.save('keras_model'+str(SEQUENCE_LENGTH)+'.h5')
pickle.dump(history, open('history'+str(SEQUENCE_LENGTH)+'.p', 'wb'))



#Loading back the saved weights and history

model = load_model('keras_model'+str(SEQUENCE_LENGTH)+'.h5')
history = pickle.load(open('history'+str(SEQUENCE_LENGTH)+'.p', 'rb'))


#Evaluation
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc= 'upper left')

plt.savefig("01.Accuracy.png")

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc= 'upper left')

plt.savefig("02.Loss.png")



#Testing
def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1
    return x
#The sequences must be 40 chars long and the tensor is of the shape (1, 40, 57)



#The sample function
#This function allows us to ask our model what are the next probable characters (The heap simplifies the job)
def sample(preds, top_n = 3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)
  
  
  
#Prediction function
def predict_completion(text):
    original_text = text
    generalised = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]

        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion
          
          
          
#This methods wraps everything and allows us to predict multiple completions
def predict_completions(text, n = 3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]


for q in quotes:
    seq = q[:SEQUENCE_LENGTH].lower()
    print (seq)
    print (predict_completions(seq, 5))
    print ()
    
    
    
