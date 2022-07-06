import csv
import numpy as np
from sklearn.model_selection import  cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
import codecs
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
x_train = []
y_train = []
x_test = []
y_test = []

def evaluate_model(x_train, y_train, model,x_test,y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    pre_pro = model.predict_proba(x_test)
    scores = cross_val_score(model, x_train, y_train, cv = 5, scoring = "recall")
    diff = scores.mean() - model.score(x_test, y_test)
    SD = diff / scores.std()
    
    print(f"Training Score:{model.score(x_train, y_train)}")
    print(f"Cross V Score: {scores.mean()} +/- {scores.std()}")
    print(f"Testing Score: {model.score(x_test, y_test)}")
    print(f"Cross & Test Diff: {diff}")
    print(f"Standard Deviations Away: {SD}")
    # joblib.dump(model, './featureSVM/smoteVIA.pkl') # save
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    print(cm)
    plt.imshow(cm,cmap = 'Blues',)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(np.arange(2), ('HISL', 'Norm'))
    plt.yticks(np.arange(2), ('HISL', 'Norm'))
    plt.title('Confusion matrix')
    plt.colorbar()
    # Control the output name
    k = 6
    plt.savefig('./featureSVM/smoteVIA/matrixoupt{}.jpeg'.format(k))
    plt.clf()

    y_test = np.reshape(y_test,(np.shape(y_test)[0],1))
    f = codecs.open('./featureSVM/smoteVIA/output{}.csv'.format(k),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['predict','trueanswer'])
    for i in range(np.shape(y_test)[0]):
        writer.writerow([preds[i],y_test[i][0]])
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
    plt.savefig('./featureSVM/smoteVIA/roc{}.jpeg'.format(k))

    plt.clf()
    
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, pre)
    lw = 2
    plt.plot(precision, recall, color = 'darkorange',lw = lw)
    plt.savefig('./featureSVM/smoteVIA/recall{}.jpeg'.format(k))


with open('./split/10VIAtrainfeatureandcolor.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        newrow = []
        if i == 0:
            i = i+1
            continue
        for i in range(10):
            newrow.append(float(row[i]))

        x_train.append(newrow)
        y_train.append(int(row[10]))


with open('./split_test/10VIAtestfeatureandcolor.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        newrow = []
        if i == 0:
            i = i+1
            continue
        for i in range(10):
            newrow.append(float(row[i]))

        x_test.append(newrow)
        y_test.append(int(row[10]))

clf = svm.SVC(C = 2,kernel = 'rbf', gamma = 0.5,decision_function_shape = 'ovr',probability = 
True)
  
from imblearn.over_sampling import SMOTE
smt = SMOTE(k_neighbors = 7)
x_train_SMOTE, y_train_SMOTE = smt.fit_sample(x_train, y_train)

from imblearn.pipeline import make_pipeline
pipeline = make_pipeline(smt, clf)
evaluate_model(x_train_SMOTE, y_train_SMOTE, pipeline,x_test,y_test)