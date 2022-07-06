import csv
import numpy as np
from sklearn import svm
import joblib
import codecs
from sklearn.metrics import  roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
from mixed_naive_bayes import MixedNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

x_train = []
y_train = []
x_test = []
y_test = []
size = 10
def evaluate_model(x_train_SMOTE, y_train_SMOTE, model):
    model.fit(x_train_SMOTE, y_train_SMOTE)
    # Control output file name
    m = 1
    joblib.dump(model,'./featureSVM/output/smoteVIA_VILI.pkl') # save

    # Training section
    preds = model.predict(x_train)
    # Save smote model train prediction answer
    rows = list(preds)
    f = codecs.open('./featureSVM/output/smotetrainoutput{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['predict','label'])
    for i in range(len(rows)):
        writer.writerow([rows[i],y_train_SMOTE[i]])
    f.close()

    result = []
    for i in preds:
        result.append(i)
    train_x = []
    train_y = []
    with open('./traintext.txt', 'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        k = 0
        rows  = []
        # Get input(output of last model, age, HPV, TCT)
        for line in lines:
            row =  []
            row.append(result[k])
            row.append(int(line.split(',')[0].split('_')[1]))
            if line.split(',')[1].split('  ')[0] ==  '1':
                row.append(1)
            else:
                row.append(0)
            if line.split(',')[1].split('  ')[1] ==  '1':
                row.append(1)
            else:
                row.append(0)
            if line.split(',')[1].split('  ')[2] ==  '1':
                row.append(1)
            else:
                row.append(0)

            # tct
            for j in range(6):
                if int(line.split(",")[2].split("  ")[j]) == 1:
                    row.append(j)
            
            k = k+1
            rows.append(row)
            if line.split(",")[3] == '1  0  0':
                train_y.append(0)
            if line.split(",")[3] == '0  1  0' or line.split(",")[3] == '0  0  1':
                train_y.append(1)
        train_x = rows

    # The test part
    preds = model.predict(x_test)
    # Save the predictable value of smote model test
    rows = list(preds)
    f = codecs.open('./featureSVM/output/smotetestoutput{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['predict','label'])
    for i in range(len(rows)):
        writer.writerow([rows[i],y_test[i]])
    f.close()

    result = []
    for i in preds:
        result.append(i)
    test_x = []

    with open('./testtext.txt', 'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        k = 0
        rows  = []
        # Get input(output of last model, age, HPV, TCT)
        for line in lines:
            row =  []
            row.append(result[k])
            row.append(int(line.split(',')[0].split('_')[1]))
            if line.split(',')[1].split('  ')[0] ==  '1':
                row.append(1)
            else:
                row.append(0)
            if line.split(',')[1].split('  ')[1] ==  '1':
                row.append(1)
            else:
                row.append(0)
            if line.split(',')[1].split('  ')[2] ==  '1':
                row.append(1)
            else:
                row.append(0)
            # tct
            for j in range(6):
                if int(line.split(",")[2].split("  ")[j]) == 1:
                    row.append(j)
            
            k = k+1
            rows.append(row)
        test_x = rows
        
    classifier = MixedNB(alpha = 0, priors = [0.1,0.9], categorical_features = [0,2,3,4,5])
    classifier.fit(train_x, y_train)
    joblib.dump(classifier, './featureSVM/output/BayesVIAVILI.pkl') # save

    # Predictive test data sets
    prediction = classifier.predict(test_x)
    prediction_train = classifier.predict(train_x) 
    pre_pro = classifier.predict_proba(test_x)
    classifier.score(test_x,y_test)

    # Bayesian model test output
    rows = list(prediction)
    f = codecs.open('./featureSVM/output/Bayestestpredict{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['predict'])
    for i in rows:
        writer.writerow([int(i)])
    f.close()

    # Bayesian model training output
    f = codecs.open('./featureSVM/output/Bayestrainpredict{}.csv'.format(m),'w','gbk')
    rows = list(prediction_train)
    writer = csv.writer(f)
    writer.writerow(['predict'])
    for i in rows:
        writer.writerow([int(i)])
    f.close()   

    # Output result
    print(f"Training Set Accuracy : {accuracy_score(train_y, prediction_train) * 100} %\n") 
    print(f"Test Set Accuracy : {accuracy_score(y_test, prediction) * 100} % \n") 
    print(f"Classifier Report : \n {classification_report(y_test, prediction)}")


    # Confusion matrix
    cm = confusion_matrix(y_test, prediction)
    print(cm)
    plt.imshow(cm,cmap = 'Blues',)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(np.arange(2), ('HISL', 'Norm'))
    plt.yticks(np.arange(2), ('HISL', 'Norm'))
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.savefig('./featureSVM/output/matrixoupt{}.jpeg'.format(m))
    plt.clf()
    # Compute ROC curve and ROC area for each class
    pre = []
    for i in range(np.shape(test_x)[0]):
        pre.append(pre_pro[i][y_test[i]])
    fpr = dict()
    tpr = dict()
    fpr, tpr, thresholds  = roc_curve(y_test, pre)
    f = codecs.open('./featureSVM/output/fpr{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['fpr'])
    for i in fpr:
        writer.writerow([i])
    f.close() 

    f = codecs.open('./featureSVM/output/tpr{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['tpr'])
    for i in tpr:
        writer.writerow([i])
    f.close() 

    f = codecs.open('./featureSVM/output/thresholds{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['thresholds'])
    for i in thresholds:
        writer.writerow([i])
    f.close() 
    
    roc_auc = auc(fpr, tpr)
    print("roc_auc:",roc_auc)
    lw = 2
    plt.plot(fpr, tpr, color = 'darkorange',lw = lw, label = 'ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic examplme')
    plt.legend(loc = "lower right")
    plt.savefig('./featureSVM/output/roc{}.jpeg'.format(m))
    plt.clf()
    

    precision, recall, thresholds = precision_recall_curve(y_test, pre)
    auc_score = auc(recall, precision)
    print(auc_score)
    # print('No Skill PR AUC: %.3f' % auc_score)
    lw = 2
    plt.plot(precision, recall, color = 'darkorange',lw = lw)
    plt.savefig('./featureSVM/output/recall{}.jpeg'.format(m))

    f = codecs.open('./featureSVM/output/precision{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['precision'])
    for i in precision:
        writer.writerow([i])
    f.close() 

    f = codecs.open('./featureSVM/output/recall{}.csv'.format(m),'w','gbk')
    writer = csv.writer(f)
    writer.writerow(['recall'])
    for i in recall:
        writer.writerow([i])
    f.close() 

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

# The preliminary training
clf = svm.SVC(C = 0.02,kernel = 'rbf',gamma = 0.3,decision_function_shape = 'ovr')
# smote
smt = SMOTE(k_neighbors = 4,random_state = 10)
x_train_SMOTE, y_train_SMOTE = smt.fit_sample(x_train, y_train)

# predict
pipeline = make_pipeline(smt, clf)
# Bayes model
evaluate_model(x_train_SMOTE, y_train_SMOTE, pipeline)
