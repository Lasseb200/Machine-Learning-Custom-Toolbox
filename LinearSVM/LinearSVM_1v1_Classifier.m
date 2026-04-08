function Y_pred = LinearSVM_1v1_Classifier(X_test,W,b,Yu)
    k = length(Yu);
    pairs = nchoosek(1:k,2); nPairs = size(pairs,1);
    n_test = size(X_test,1);
    F = X_test*W + ones(n_test,1)*b';
    votes = zeros(size(X_test,1),k);
    for p = 1:nPairs
        n1 = pairs(p,1); n2 = pairs(p,2);
        votes(:,n1) = votes(:,n1) + (F(:,p) > 0);
        votes(:,n2) = votes(:,n2) + (F(:,p) < 0);
    end
    [~, idx_pred] = max(votes,[],2);
    Y_pred = Yu(idx_pred);
end