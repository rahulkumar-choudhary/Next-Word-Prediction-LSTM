from IPython.display import Image
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow import keras
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os


# insert data

"""
    Dataset: http://www.gutenberg.org/cache/epub/5200/pg5200.txt
    Remove all the unnecessary data and label it as Metamorphosis-clean.
    The starting and ending lines should be as follows.

"""


file = open("1661-0.txt", "r", encoding="utf8")
lines = []

for i in file:
    lines.append(i)

print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])


# data pre-processing

data = ""

for i in lines:
    data = ' '. join(lines)

data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
data[:360]


# map punctuation to space
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
new_data = data.translate(translator)

new_data[:500]

z = []

for i in data.split():
    if i not in z:
        z.append(i)

data = ' '.join(z)
data[:500]


# Tokenization

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function.
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:10]


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)

print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]


X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])

X = np.array(X)
y = np.array(y)


print("The Data is: ", X[:5])
print("The responses are: ", y[:5])


y = to_categorical(y, num_classes=vocab_size)
y[:5]


# Creating the Model

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))


model.summary()


# Plot The Model


keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)


# Callbacks


checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2,
                           patience=3, min_lr=0.0001, verbose=1)

logdir = 'logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)


# Compile The Model


model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))


# Fit The Model

# epochs = 150 , batch size = 64

model.fit(X, y, epochs=30, batch_size=25, callbacks=[
          checkpoint, reduce, tensorboard_Visualization])


# visiual

# https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
# tensorboard --logdir="./logsnextword1"
# http://DESKTOP-U3TSCVT:6006/

pil_img = Image(filename='graph1.png')
display(pil_img)
