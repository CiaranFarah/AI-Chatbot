import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from tkinter import *
from PIL import ImageTk, Image


def makeentry(parent, caption, width=None, **options):
    Label(parent, text=caption).pack(side=LEFT)
    entry = Entry(parent, **options)
    if width:
        entry.config(width=width)
    entry.pack(side=LEFT)
    return entry


def bag_of_words(s, words_list):
    words_bag = [0 for _ in range(len(words_list))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words_list):
            if w == se:
                words_bag[i] = 1

    return np.array(words_bag)


def chat():

    user_input = e.get()
    results = model.predict([bag_of_words(user_input, words)])[0]
    results_index = np.argmax(results)  # Index of greatest node value
    tag = labels[results_index]

    if results[results_index] > .7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg["responses"]

        L2.config(text=random.choice(responses))
        # print(random.choice(responses))

    else:
        L2.config(text="Sorry I don't understand")
        # print("Sorry I don't understand")


#nltk.download()

stemmer = LancasterStemmer()

with open('training_data.json') as file:
    data = json.load(file)


try:
    x
    with open("data.pickle", "rb") as f:  # Read data as bytes
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)  # Tokenize splits up by word
            words.extend(wrds)  # Put the tokenized words into our words list
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # Turn words into all lowercase
    words = sorted(list(set(words)))  # Removes all duplicate words

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]  # List of 0 corresponding with length of sentence

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]  # Stem each words in docs_x

        for w in words:
            if w in wrds:
                bag.append(1)  # Means word exists
            else:
                bag.append(0)

        output_row = out_empty[:]  # Makes copy of out_empty
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:  # Write data as bytes
        pickle.dump((words, labels, training, output), f)

#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])  # Define input shape we are expecting for our model
net = tflearn.fully_connected(net, 8)  # Add fully connected layer to neural network and have 8 neurons for hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # Allows us to get probabilities for each output
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#     model.load("model.tflearn")
# except:

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)  # Show the model the same data 1000 times
model.save("model.tflearn")


master = Tk()
master.configure(background="white")
master.geometry("1000x500+700+300")

b = Button(master, text="Enter", width=20, height=2, command=chat)
b.pack()
b.place(x=0, y=30)

e = Entry(master, width=50)
e.pack()
e.place(x=0, y=5)

e.focus_set()


canvas = Canvas(master, width=500, height=400)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("girl.png"))
# img = PhotoImage(file="zoey.jpg")
canvas.create_image(20,0, anchor=NW, image=img)

L1 = Label(master, text="Zoey:", bg="white", font=("Helvetica", 20))
L1.pack()
L1.place(x=400, y=450)

L2 = Label(master, text="", bg="white", font=("Helvetica", 20))
L2.pack()
L2.config(bg="white")
L2.place(x=475, y=450)

mainloop()
e = Entry(master, width=50)
e.pack()

text = e.get()