function [alpha, b, X_training_pair, Y_training_pair, A_Platt, B_Platt] = multiclass_training_pairwise(X_training,R,K_tot,C,pairs,p,multiclassModel)
    n1 = pairs(p,1); n2 = pairs(p,2);
    idx1 = R(:,n1) == 1; idx2 = R(:,n2) == 1;
    idx_pair = idx1 | idx2;
    X_training_pair = X_training(idx_pair,:);
    Y_training_pair = [ones(sum(idx1),1); -ones(sum(idx2),1)];
    [alpha, b, K] = binary_SVM_training(X_training_pair,Y_training_pair,K_tot,C,idx_pair); 

    if strcmpi(multiclassModel, 'soft1v1') | strcmpi(multiclassModel, 'pairwiseCoupling')
        F_training = ((alpha.*Y_training_pair)'*K)' + b;
        [A_Platt, B_Platt] = Platt_coefficients(F_training,Y_training_pair);
    else
        A_Platt = NaN; B_Platt = NaN;
    end
end