import inltk
import pandas
import numpy
import tensorflow
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tflearn
import json
import random
import speech_recognition as sr
import pyttsx3
import pyaudio
import csv
import pandas as pd
from inltk.inltk import tokenize
import pickle

tags =[]
words=[]

with open("intents.json",encoding="utf-8-sig") as file:
    data = json.load(file)

my_stopwords = ['.',',','!','@','$','#','૦','૧','૨','૩','૪','૫','૬','૭','૮','૯','૧૦','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #tokenizer = RegexpTokenizer(r'\w+')
        #wrds2 = tokenizer.tokenize(pattern,'gu')
        wrds0 = tokenize(pattern,'gu')
        #wrds3 = [w.lower() for w in wrds2]
        wrds = list(filter(lambda i:i not in my_stopwords,wrds0))
        words.extend(wrds)
    tags.append(intent["tag"])

#words = [w.lower() for w in words]
words = sorted(list(set(words)))
tags = sorted(list(set(tags)))

#making the training and training_tags dataset-
training=[] #Actual Training Dataset for the model
training_tags=[]
for intent in data["intents"]:
    tag = intent["tag"]
    tag_ind = tags.index(tag)
    
    for pattern in intent["patterns"]:
        training_bin = [0 for x in range(0,len(words))]
        training_tags_bin = [0 for x in range(0,len(tags))]

        wrds0 = tokenize(pattern,'gu')
        wrds = list(filter(lambda i:i not in my_stopwords,wrds0))
        
        for w in wrds:
            ind = words.index(w)
            training_bin[ind] = 1

        training_tags_bin[tag_ind] = 1
        training.append(training_bin)
        training_tags.append(training_tags_bin)
    
training = numpy.array(training)
training_tags = numpy.array(training_tags)


#making of the model-
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(training_tags[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training,training_tags,n_epoch=10,batch_size=8,show_metric=True)
model.save("model.tflearn")
   

#Converting words to bag-
def bag(inp,words):
        user_bag=[0 for x in range(0,len(words))]
        swrds=[]
        wrds = tokenize(inp,'gu')
        swrds.extend(wrds)
        wrds = [w.lower() for w in inp]
        for w in swrds:
            if(w in words):
                ind = words.index(w)
                user_bag[ind] = 1

        #print("words :",[words[x] for x in range(0,len(user_bag)) if user_bag[x]==1])
        #print(user_bag.count(1))
                
        if(user_bag.count(1)== 0):return -1
        return numpy.array(user_bag)


def reply(tag,data):
    for intents in data["intents"]:
        if(tag == intents["tag"]):
            responses = intents["responses"]
    return tag
                

def start(model,data,words,tags):

    none_val = 'NAN'
    File = pd.read_csv("Test_unicode.csv",encoding="utf-8-sig")
    z = File.iloc[:,0].values
    u = File.iloc[:,1].values
    writeheader = True
    for i in range(len(z)):
        x=[]
        tags0=[]
        x=z[i]
        #print(x)
        results=[bag(x,words)]

        try:
            if(results[0]==-1):
                tags0.append(none_val)
        except:
            results = model.predict(results)
            ind = numpy.argmax(results)
            tags0.append(reply(tags[ind],data))

        data1 = {'Headline':[z[i]],'Actual Tag':[u[i]],'Predicted Tag':[tags0]}
        df = pd.DataFrame(data1)
        if writeheader is True:
            df.to_csv("AutomatedTags_NewsClassification.csv", mode="a", header=True, index=False)
            writeheader=False
        else:
            df.to_csv("AutomatedTags_NewsClassification.csv", mode="a", header=False, index=False)

        
start(model,data,words,tags)
