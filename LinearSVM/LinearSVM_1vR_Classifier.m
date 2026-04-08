function Y_pred = LinearSVM_1vR_Classifier(W,b,X_test,Yu)
    n_test = size(X_test,1);
    F = X_test*W + ones(n_test,1)*b';
    [~, idx_pred] = max(F,[],2);
    Y_pred = Yu(idx_pred);
end