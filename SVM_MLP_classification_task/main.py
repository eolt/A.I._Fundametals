import plot_methods as plm

def main():

    df = plm.pd.read_csv('heart_failure_clinical_records_dataset.csv')
  
    # Extract features a X variable
    X = df.drop('DEATH_EVENT', axis=1)
    
    # Extract 0 or 1 value from 'DEATH_EVENT' as our y variable
    y = df['DEATH_EVENT']

    # 10-fold cross-validation
    # the last, 267 - 298 entries have DEATH_EVENT = 0 
    # thus, we should shuffle before splitting 
    kf = plm.KFold(n_splits=10, shuffle=True, random_state=42)

    #   Split the training and testing sample from data
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

    # Scale the data before doing SVM and MLP for better accuracy 
    X_train_scaled = plm.scale(X_train)
    X_test_scaled = plm.scale(X_test)

    #   Call SVM with Linear and RBF kernel. 
    #   The method plots a confusion matrix and prints a classification report
    plm.svm_confusion_method('linear', X_train_scaled, y_train, X_test_scaled, y_test)
    plm.svm_confusion_method('rbf', X_train_scaled, y_train, X_test_scaled, y_test)

    #   Call MLP with logistic and Tanh activation. 
    #   The method plots a confusion matrix and prints a classification report
    plm.mlp_confusion_method('logistic', X_train_scaled, y_train, X_test_scaled, y_test)

    # Tanh must be implemented in ranges (-1, 1)
    y_train_tanh = plm.np.where(y_train == 0, -1, y_train)
    y_test_tanh = plm.np.where(y_test == 0, -1, y_test)

    plm.mlp_confusion_method('tanh', X_train_scaled, y_train_tanh, X_test_scaled, y_test_tanh)

    #   Show plotted graphs with matplotlib
    plm.plt.show()

if __name__ == '__main__':
    main()
