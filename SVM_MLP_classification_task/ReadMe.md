# SVM and MLP Classification Task
Practice task for machine learning which implement Support Vector Machine and Multilayer Perception 



# Intro 
Support Vector Machine (SVM) and Multilayer Perception (MLP) are both fundamental approaches to data classification. Both algorithms incorporate functions for improving accuracy in there classifications. The type of kernel/activation functions, parameters, and processing vary and can be determined by the user of the program.

This task uses a Heart failure clinical records dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records). The goal of this task is to learn binary classification models that distinguish the two classes (in this case mortality event or not).

# Implementation 
The program will use the built-in functions in python, Scikit-Learn for both SVM and MLP. The accuracy of thses algorithms is measured using 10-fold cross validation. The findings are plotted in confusion matrices, using the plot_confusion_matrix function. 

I took the liberty of separating the ploting and training algorithms in separate files. The main.py initially reads the dataset, splits the variables, initializes 10-fold CV, and performs train_test_split. The file then calls the methods from plot_method.py which run the SVM or MLP approaches with the given kernel or activation functions and training and test sets. The plot_methods will plot the results in separate image figures and print the classification report the the console.

# Outcomes
The first SVM uses linear kernel. 

![Alt text](https://github.com/eolt/A.I._Fundametals/blob/c6a794980f3aac24103ecb1f196bb1eca706eaed/SVM_MLP_classification_task/plotted_confusion_matrix/SVM_linear.png)

                     precision    recall  f1-score   support
              
               0       0.95      0.88      0.91        24
               1       0.57      0.80      0.67         5

        accuracy                           0.86        29
       macro avg       0.76      0.84      0.79        29
    weighted avg       0.89      0.86      0.87        29


Second SVM uses Radial Basis Function (RBF).

![Alt text](https://github.com/eolt/A.I._Fundametals/blob/c6a794980f3aac24103ecb1f196bb1eca706eaed/SVM_MLP_classification_task/plotted_confusion_matrix/SVM_rbf.png)

                     precision    recall  f1-score   support

               0       1.00      0.92      0.96        24
               1       0.71      1.00      0.83         5

        accuracy                           0.93        29
       macro avg       0.86      0.96      0.89        29
    weighted avg       0.95      0.93      0.94        29



The first MLP uses logistic/sigmoid activation.

![Alt text](https://github.com/eolt/A.I._Fundametals/blob/c6a794980f3aac24103ecb1f196bb1eca706eaed/SVM_MLP_classification_task/plotted_confusion_matrix/MLP_logistic.png)

                     precision    recall  f1-score   support

               0       0.95      0.88      0.91        24
               1       0.57      0.80      0.67         5

        accuracy                           0.86        29
       macro avg       0.76      0.84      0.79        29
    weighted avg       0.89      0.86      0.87        29


Second MLP uses Tanh activation.

![Alt text](https://github.com/eolt/A.I._Fundametals/blob/61347fd7e6cb769e51cd2a932f9e1f281d584e9d/SVM_MLP_classification_task/plotted_confusion_matrix/MLP_tanh.png)

                    precision    recall  f1-score   support

              -1       0.95      0.83      0.89        24
               1       0.50      0.80      0.62         5

        accuracy                           0.83        29
       macro avg       0.73      0.82      0.75        29
    weighted avg       0.87      0.83      0.84        29
