function [Y_pred, confidence] = kNN_wconf(X_training,Y_training,X_test,f)
    Yu = unique(Y_training,'stable');
    n_test = size(X_test,1);
    Delta = pdist2(X_test,X_training,'euclidean');
    trainingClasses = repmat(Y_training,1,n_test);
    [Delta, idx] = sort(Delta, 2);
    trainingClasses = trainingClasses(idx);
    Delta = Delta(:,1:f);
    trainingClasses = trainingClasses(:,1:f);
    W = ones(n_test,f)./(Delta+eps*ones(n_test,f));
    [uniqueStrings, ~, numericVector] = unique(trainingClasses,'stable');
    trainingClasses = reshape(numericVector, size(trainingClasses));
    k = length(uniqueStrings);
    R = zeros(n_test,f,k);
    idx = sub2ind([n_test, f, k], repmat((1:n_test)',1,f), repmat(1:f,n_test,1), trainingClasses);
    R(idx) = 1;
    W_k = reshape(W,[n_test, f, 1]); W_k = repmat(W_k,[1, 1, k]);
    S = sum(W_k.*R,2);
    S = squeeze(S);
    [~, idx_pred] = max(S,[],2);
    Y_pred = Yu(idx_pred);
    totalWeighting = sum(S,2);
    confidence = S(sub2ind(size(S), (1:n_test)', idx_pred)) ./ totalWeighting;
end