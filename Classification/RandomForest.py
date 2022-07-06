from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np
import matplotlib.pyplot as plt
import codecs
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve

x_train = []
y_train = []
x_test = []
y_test = []
# The input features are 10
size = 10
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

clf = RandomForestClassifier(n_estimators = 250,oob_score = True,criterion = 'gini')
# Returns the prediction result of the validation set by the model trained by the training set
result = clf.fit(x_train,y_train).predict(x_test)
pre_pro = clf.predict_proba(x_test)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Confusion matrix
cm = confusion_matrix(y_test, result)
print(cm)
plt.imshow(cm,cmap = 'Blues',)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(2), ('HISL', 'Norm'))
plt.yticks(np.arange(2), ('HISL', 'Norm'))
plt.title('Confusion matrix')
plt.colorbar()
# Control output name
k = 7
plt.savefig('./featureSVM/randomforestoutput/matrixoupt{}.jpeg'.format(k))
plt.clf()

# The output
y_test = np.reshape(y_test,(np.shape(y_test)[0],1))
rows  = []
rows = list(result)+list(y_test)
f = codecs.open('./featureSVM/randomforestoutput/output{}.csv'.format(k),'w','gbk')
writer = csv.writer(f)
writer.writerow(['predict','trueanswer'])
for i in range(np.shape(y_test)[0]):
    writer.writerow([result[i],y_test[i][0]])
f.close()

# orc
pre = []
for i in range(np.shape(x_test)[0]):
    pre.append(pre_pro[i][y_test[i]])
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(y_test, pre)
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
plt.savefig('./featureSVM/randomforestoutput/roc{}.jpeg'.format(k))
plt.clf()

precision, recall, thresholds = precision_recall_curve(y_test, pre)
lw = 2
plt.plot(precision, recall, color = 'darkorange',lw = lw)
plt.savefig('./featureSVM/randomforestoutput/recall{}.jpeg'.format(k))