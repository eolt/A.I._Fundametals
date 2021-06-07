# SVM and MLP Classification Task
Practice task for machine learning which implement Support Vector Machine and Multilayer Perception 



# Intro 
Support Vector Machine (SVM) and Multilayer Perception (MLP) are both fundamental approaches to data classification. Both algorithms incorporate  kernel functions for improving accuracy in there classifications. The type of kernel functions, parameters, and processing vary and can be determined by the user of the program.

This task uses a Heart failure clinical records dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records). The goal of this task is to learn binary classification models that distinguish the two classes (in this case mortality event or not).

# Implementation 
The program will use the built-in functions in python, Scikit-Learn for both SVM and MLP. The accuracy of thses algorithms is measured using 10-fold cross validation. The findings are plotted in confusion matrices, using the plot_confusion_matrix function. I took the liberty of separating the ploting and training algorithms in separate files. The main.py initially reads the dataset, splits the variables, initializes 10-fold CV, and performs train_test_split. The file then calls the methods from plot_method.py which run the SVM or MLP approaches with the given kernel functions and training and test sets. The plot_methods will polt the results in a figure. 





