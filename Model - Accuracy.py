import csv
import pandas as pd
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

File = pd.read_csv("AutomatedTags_NewsClassification.csv",encoding="ISO-8859-1")
x = File.iloc[:,1].values  
y = File.iloc[:,2].values


training=[]
for i in range(len(x)):
    training.append(x[i])

training = numpy.array(training)

        
testing=[]
for j in range(len(y)):
    testing.append(y[j])

testing = numpy.array(testing)

print("\n")

str = 'Accuracy: {}'
print(str.format(accuracy_score(training,testing) * 100))
print("\n")

print("Confusion Matrix:")
print("\n")
cmt = confusion_matrix(training,testing)
#print(cmt)
index = ['Business','Entertainment','Tech']
columns = ['Business','Entertainment','Tech']

cm_df = pd.DataFrame(cmt,columns,index)   
print(cm_df)
