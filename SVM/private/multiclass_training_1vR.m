function [alpha, b, X_set, y_set] = multiclass_training_1vR(X_training,R,K_tot,C,j)
    n = size(X_training,1);
    idx_j = R(:,j) == 1;
    y_set = -ones(n,1);
    y_set(idx_j) = 1;
    X_set = X_training;
    idx_pair = true(n,1);
    [alpha, b] = binary_SVM_training(X_set,y_set,K_tot,C,idx_pair);
end
    