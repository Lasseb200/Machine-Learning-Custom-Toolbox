function Y_pred = LinearSVM_1vRest(X_training,Y_training,X_test,C)
    [W, b, Yu] = LinearSVM_training1vRest(X_training,Y_training,C);
    Y_pred = LinearSVM_1vRestClassifier(W,b,X_test,Yu);
end