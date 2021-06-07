import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt  
import numpy as np

def svm_confusion_method(k, X_train, y_train, X_test, y_test):
    #   Initialize SVM with desired kernel
    svm = SVC(kernel=k)

    #   Training
    svm.fit(X_train, y_train)

    #   Plot our confusion matrix
    plot_confusion_matrix(svm, X_test, y_test, display_labels=["Not Deceased", "Deceased"])
    plt.suptitle('SVM kernel=' + k)

    #   Print Classification report to get percentages of accuracy
    print('SVM kernel=' + k)
    print(classification_report(y_test, svm.predict(X_test), zero_division=0))
    print('\n')
    

def mlp_confusion_method(a, X_train, y_train, X_test, y_test):
    #   Initialize MLP with desired activation function
    mlp = MLPClassifier(activation=a)

    #   Training
    mlp.fit(X_train, y_train)

    #   Plot our confusion matrix
    plot_confusion_matrix(mlp, X_test, y_test, display_labels=["Not Deceased", "Deceased"])
    plt.suptitle('MLP activation=' + a)

    #   Print Classification report to get percentages of accuracy
    print('MLP activation=' + a)
    print(classification_report(y_test, mlp.predict(X_test), zero_division=0))
    print('\n')
    

