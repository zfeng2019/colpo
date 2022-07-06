import csv
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten,Dense,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import codecs
from sklearn.metrics import confusion_matrix, roc_curve, auc

def baseline_model():
    model = Sequential()    
    model.add(Conv1D(64, 3, input_shape = (10,1),activation = 'relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    print(model.summary()) # Display network structure
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate = 1e-4), metrics = ['accuracy'])
    return model

size = 10
x_train = []
y_train = []
x_test = []
y_test = []

# Obtain salient features of training data set
with open('./split/10mixtrainfeatureandcolor.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        newrow = []
        if i == 0:
            i = i+1
            continue
        for i in range(size):                
            newrow.append(float(row[i]))

        x_train.append(newrow)
        y_train.append(int(row[size]))
# Obtain salient features of the test dataset
with open('./split_test/10mixtestfeatureandcolor.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        newrow = []
        if i == 0:
            i = i+1
            continue
        for i in range(size):
            newrow.append(float(row[i]))

        x_test.append(newrow)
        y_test.append(int(row[size]))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_label = []
y_label = y_test
x_train = x_train.reshape((np.shape(x_train)[0],np.shape(x_train)[1],1))
x_test = x_test.reshape((np.shape(x_test)[0],np.shape(x_test)[1],1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = baseline_model()

history = model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs = 100,batch_size = 32,verbose = 1)
pre_pro = model.predict(x_test)
result = model.predict_classes(x_test)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#Confusion matrix
cm = confusion_matrix(y_label, result)
print(cm)
plt.imshow(cm,cmap = 'Blues',)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(2), ('HISL', 'Norm'))
plt.yticks(np.arange(2), ('HISL', 'Norm'))
plt.title('Confusion matrix')
plt.colorbar()
# Control output name
k = 5 
plt.savefig('./featureSVM/neuraloutput/matrixoupt{}.jpeg'.format(k))
plt.clf()

# The output value
y_label = np.reshape(y_label,(np.shape(y_label)[0],1))
rows  = []
rows = list(result)+list(y_label)
f = codecs.open('./featureSVM/neuraloutput/output{}.csv'.format(k),'w','gbk')
writer = csv.writer(f)
writer.writerow(['predict','trueanswer'])
for i in range(np.shape(y_test)[0]):
    writer.writerow([result[i],y_label[i][0]])
f.close()

# orc
pre = []
for i in range(np.shape(x_test)[0]):
    pre.append(pre_pro[i][y_label[i]])
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(y_label, pre)
roc_auc = auc(fpr, tpr)
lw = 2
plt.plot(fpr, tpr, color = 'darkorange',lw = lw, label = 'ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc = "lower right")
plt.savefig('./featureSVM/neuraloutput/roc{}.jpeg'.format(k))