function Y_pred = LinearSVM_1v1(X_training,Y_training,X_test,C)
    [W, b, Yu] = LinearSVM_training1v1(X_training,Y_training,C);
    Y_pred = LinearSVM_1v1classifier(X_test,W,b,Yu);
end