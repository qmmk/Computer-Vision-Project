from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import torch

def setup():
    bankdata = pd.read_csv("./dataset/dataset_SVM_NEW.csv")

    X = bankdata.drop('class', axis=1)
    y = bankdata['class']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    svclassifier = SVC(kernel='poly')
    svclassifier.fit(X, y)
    return svclassifier

def check(svclassifier_,v):
    v = v.squeeze(2)
    v = v.squeeze(0)
    vec = []

    for i in range(512):
        vec.append(v[i][0].item())
    y_pred = svclassifier_.predict([vec])
    if y_pred == 1:
        return True
    else:
        return False



# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))