#import libaries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

#read data.
data = pd.read_csv('data/heart.csv')

#pre-process categorical features
catagorialList = ['sex','cp','fbs','restecg','exang','ca','thal']
for item in catagorialList:
    data[item] = data[item].astype('object')
data = pd.get_dummies(data, drop_first=True)

#normalize training features
y = data['target'].values
y = y.reshape(y.shape[0],1)
x = data.drop(['target'],axis=1)
minx = np.min(x)
maxx = np.max(x)
x = (x - minx) / (maxx - minx)
x.head()

#split training set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#set up neural network. Structure is 21->12->1
model = Sequential()
model.add(Dense(12, input_dim=21, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#Compile and train neural network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
output = model.fit(x_train, y_train, epochs=500, batch_size=x_train.shape[0])
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#plot accuracy-epoch figure
plt.plot(output.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy.png',dpi=100)
plt.show()

#plot loss-epoch figure
plt.plot(output.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss.png',dpi=100)
plt.show()
